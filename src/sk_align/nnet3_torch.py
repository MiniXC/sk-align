"""
PyTorch implementation of nnet3 forward pass for forced alignment.

Loads weights from a parsed ``Nnet3Model`` and implements the full
computation graph (TDNN-F architecture) in PyTorch, producing
log-likelihoods compatible with the Viterbi decoder.

This module is the "scorer" that replaces pykaldi's nnet3 computation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from sk_align.nnet3_model import (
        BatchNormParams,
        ComponentNode,
        GeneralDropoutParams,
        InputNode,
        LinearParams,
        LogSoftmaxParams,
        NaturalGradientAffineParams,
        Nnet3Model,
        NoOpParams,
        OutputNode,
        RectifiedLinearParams,
        TdnnParams,
    )


# ---------------------------------------------------------------------------
# Descriptor parsing
# ---------------------------------------------------------------------------

def parse_descriptor(desc: str) -> dict:
    """Parse an nnet3 descriptor string into a nested dict.

    Supported descriptors:
    - ``"name"`` → ``{"type": "ref", "name": "name"}``
    - ``"Sum(a, b)"`` → ``{"type": "sum", "args": [parsed_a, parsed_b]}``
    - ``"Scale(factor, a)"`` → ``{"type": "scale", "factor": float, "arg": parsed_a}``
    - ``"Offset(a, t)"`` → ``{"type": "offset", "arg": parsed_a, "t": int}``
    - ``"Append(a, b, ...)"`` → ``{"type": "append", "args": [...]}``
    """
    desc = desc.strip()

    # Function-like descriptors
    for func_name in ("Sum", "Scale", "Offset", "Append", "ReplaceIndex",
                       "IfDefined", "Round", "Switch", "Failover"):
        if desc.startswith(func_name + "(") and desc.endswith(")"):
            inner = desc[len(func_name) + 1:-1]
            args = _split_args(inner)

            if func_name == "Sum":
                return {
                    "type": "sum",
                    "args": [parse_descriptor(a) for a in args],
                }
            elif func_name == "Scale":
                return {
                    "type": "scale",
                    "factor": float(args[0]),
                    "arg": parse_descriptor(args[1]),
                }
            elif func_name == "Offset":
                return {
                    "type": "offset",
                    "arg": parse_descriptor(args[0]),
                    "t": int(args[1]),
                }
            elif func_name == "Append":
                return {
                    "type": "append",
                    "args": [parse_descriptor(a) for a in args],
                }
            elif func_name == "ReplaceIndex":
                return {
                    "type": "replace_index",
                    "arg": parse_descriptor(args[0]),
                    "var": args[1],
                    "value": int(args[2]),
                }
            else:
                # Generic function
                return {
                    "type": func_name.lower(),
                    "args": [parse_descriptor(a) for a in args],
                }

    # Simple reference
    return {"type": "ref", "name": desc}


def _split_args(s: str) -> list[str]:
    """Split comma-separated arguments respecting balanced parentheses."""
    args = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            args.append(s[start:i].strip())
            start = i + 1
    args.append(s[start:].strip())
    return args


# ---------------------------------------------------------------------------
# PyTorch component modules
# ---------------------------------------------------------------------------

class KaldiBatchNorm(nn.Module):
    """Kaldi BatchNormComponent (test-mode only).

    Applies: ``y = (x - mean) / sqrt(var + eps) * target_rms``
    """

    def __init__(self, params: BatchNormParams):
        super().__init__()
        self.dim = params.dim
        self.block_dim = params.block_dim
        self.epsilon = params.epsilon
        self.target_rms = params.target_rms

        # The stored stats are means and variances
        self.register_buffer('mean', torch.from_numpy(params.stats_mean.copy()))
        self.register_buffer('var', torch.from_numpy(params.stats_var.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D) or (B, T, D)
        # Normalize: (x - mean) / sqrt(var + eps) * target_rms
        scale = self.target_rms / torch.sqrt(self.var + self.epsilon)
        offset = -self.mean * scale
        return x * scale + offset


class KaldiAffine(nn.Module):
    """Affine transform: y = x @ W^T + b.

    Used for NaturalGradientAffineComponent (at test time, identical to
    a plain affine).
    """

    def __init__(self, params: NaturalGradientAffineParams):
        super().__init__()
        # linear_params: (out_dim, in_dim)
        self.register_buffer('weight', torch.from_numpy(params.linear_params.copy()))
        self.register_buffer('bias', torch.from_numpy(params.bias_params.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class KaldiLinear(nn.Module):
    """Linear transform (no bias): y = x @ W^T."""

    def __init__(self, params: LinearParams):
        super().__init__()
        # params: (out_dim, in_dim)
        self.register_buffer('weight', torch.from_numpy(params.params.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class KaldiTdnn(nn.Module):
    """TDNN (Time-Delay Neural Network) component.

    Gathers frames at specified time offsets, concatenates them,
    then applies an affine transform.
    """

    def __init__(self, params: TdnnParams):
        super().__init__()
        self.time_offsets = params.time_offsets
        # linear_params: (out_dim, in_dim * len(time_offsets))
        self.register_buffer('weight', torch.from_numpy(params.linear_params.copy()))
        has_bias = params.bias_params.size > 0
        if has_bias:
            self.register_buffer('bias', torch.from_numpy(params.bias_params.copy()))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        T, D = x.shape
        if len(self.time_offsets) == 1 and self.time_offsets[0] == 0:
            # No splicing needed
            return F.linear(x, self.weight, self.bias)

        # Gather frames at each offset and concatenate
        parts = []
        for offset in self.time_offsets:
            if offset == 0:
                parts.append(x)
            else:
                # Shift: positive offset = future frames, negative = past frames
                # Use padding to handle boundaries
                if offset > 0:
                    # Take frames starting from 'offset'
                    shifted = torch.zeros_like(x)
                    if offset < T:
                        shifted[:T - offset] = x[offset:]
                    # Pad end with last frame
                    shifted[T - offset:] = x[-1]
                    parts.append(shifted)
                else:
                    # offset < 0, take frames ending at T + offset
                    shifted = torch.zeros_like(x)
                    abs_off = -offset
                    if abs_off < T:
                        shifted[abs_off:] = x[:T - abs_off]
                    # Pad beginning with first frame
                    shifted[:abs_off] = x[0]
                    parts.append(shifted)

        spliced = torch.cat(parts, dim=-1)  # (T, D * num_offsets)
        return F.linear(spliced, self.weight, self.bias)


class KaldiRelu(nn.Module):
    """RectifiedLinearComponent — plain ReLU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class KaldiLogSoftmax(nn.Module):
    """LogSoftmaxComponent."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=-1)


class KaldiDropout(nn.Module):
    """GeneralDropoutComponent — identity at test time."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class KaldiNoOp(nn.Module):
    """NoOpComponent — identity (used for skip connections)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# Full nnet3 computation graph as a PyTorch module
# ---------------------------------------------------------------------------

def _build_component_module(comp_type: str, params) -> nn.Module:
    """Create a PyTorch module from component type and params."""
    from sk_align.nnet3_model import (
        BatchNormParams,
        GeneralDropoutParams,
        LinearParams,
        LogSoftmaxParams,
        NaturalGradientAffineParams,
        NoOpParams,
        RectifiedLinearParams,
        TdnnParams,
    )

    if isinstance(params, BatchNormParams):
        return KaldiBatchNorm(params)
    elif isinstance(params, NaturalGradientAffineParams):
        return KaldiAffine(params)
    elif isinstance(params, LinearParams):
        return KaldiLinear(params)
    elif isinstance(params, TdnnParams):
        return KaldiTdnn(params)
    elif isinstance(params, RectifiedLinearParams):
        return KaldiRelu()
    elif isinstance(params, LogSoftmaxParams):
        return KaldiLogSoftmax()
    elif isinstance(params, GeneralDropoutParams):
        return KaldiDropout()
    elif isinstance(params, NoOpParams):
        return KaldiNoOp()
    else:
        raise ValueError(f"Unknown component type: {comp_type}")


class Nnet3Network(nn.Module):
    """Complete nnet3 computation graph as a PyTorch module.

    Evaluates component nodes in topological order, following
    the descriptor-based data flow.
    """

    def __init__(self, model: Nnet3Model, output_name: str = "output"):
        super().__init__()
        from sk_align.nnet3_model import ComponentNode, InputNode, OutputNode

        self.left_context = model.left_context
        self.right_context = model.right_context
        self.priors = model.priors  # may be empty for chain models

        # Build PyTorch modules for each component
        self.component_modules = nn.ModuleDict()
        for name, comp_def in model.components.items():
            # ModuleDict requires valid Python identifiers
            safe_name = name.replace(".", "_").replace("-", "_")
            self.component_modules[safe_name] = _build_component_module(
                comp_def.component_type, comp_def.params
            )

        # Build evaluation order: topological sort of component nodes
        # that lead to the target output
        self._output_name = output_name
        self._node_map: dict[str, InputNode | OutputNode | ComponentNode] = {}
        for node in model.all_nodes:
            self._node_map[node.name] = node

        # Find the output node and trace back to get needed component nodes
        output_node = None
        for n in model.outputs:
            if n.name == output_name:
                output_node = n
                break
        assert output_node is not None, f"Output '{output_name}' not found"

        # Get ordered component nodes (they're already in topological order
        # in the config)
        self._eval_order: list[str] = []
        needed = self._trace_needed_nodes(output_node.input_descriptor)
        for n in model.all_nodes:
            if isinstance(n, ComponentNode) and n.name in needed:
                self._eval_order.append(n.name)

        # Store descriptors and component mappings
        self._descriptors: dict[str, dict] = {}
        self._comp_map: dict[str, str] = {}  # node_name → component_name
        for n in model.all_nodes:
            if isinstance(n, ComponentNode):
                self._descriptors[n.name] = parse_descriptor(n.input_descriptor)
                self._comp_map[n.name] = n.component

        self._output_descriptor = parse_descriptor(output_node.input_descriptor)

    def _trace_needed_nodes(self, descriptor_str: str) -> set[str]:
        """Find all component node names needed to evaluate a descriptor."""
        from sk_align.nnet3_model import ComponentNode
        needed: set[str] = set()
        self._trace_needed_recursive(parse_descriptor(descriptor_str), needed)
        return needed

    def _trace_needed_recursive(self, desc: dict, needed: set[str]):
        from sk_align.nnet3_model import ComponentNode

        if desc["type"] == "ref":
            name = desc["name"]
            if name in needed:
                return
            node = self._node_map.get(name)
            if node is not None and isinstance(node, ComponentNode):
                needed.add(name)
                self._trace_needed_recursive(
                    parse_descriptor(node.input_descriptor), needed
                )
        elif desc["type"] == "sum":
            for arg in desc["args"]:
                self._trace_needed_recursive(arg, needed)
        elif desc["type"] == "scale":
            self._trace_needed_recursive(desc["arg"], needed)
        elif desc["type"] == "offset":
            self._trace_needed_recursive(desc["arg"], needed)
        elif desc["type"] == "append":
            for arg in desc["args"]:
                self._trace_needed_recursive(arg, needed)

    def _evaluate_descriptor(
        self,
        desc: dict,
        activations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate a descriptor tree using cached activations."""
        if desc["type"] == "ref":
            return activations[desc["name"]]
        elif desc["type"] == "sum":
            result = self._evaluate_descriptor(desc["args"][0], activations)
            for arg in desc["args"][1:]:
                result = result + self._evaluate_descriptor(arg, activations)
            return result
        elif desc["type"] == "scale":
            return desc["factor"] * self._evaluate_descriptor(desc["arg"], activations)
        elif desc["type"] == "offset":
            x = self._evaluate_descriptor(desc["arg"], activations)
            t = desc["t"]
            T = x.shape[0]
            result = torch.zeros_like(x)
            if t > 0:
                if t < T:
                    result[:T - t] = x[t:]
                result[T - t:] = x[-1]
            elif t < 0:
                abs_t = -t
                if abs_t < T:
                    result[abs_t:] = x[:T - abs_t]
                result[:abs_t] = x[0]
            else:
                result = x
            return result
        elif desc["type"] == "append":
            parts = [self._evaluate_descriptor(a, activations) for a in desc["args"]]
            return torch.cat(parts, dim=-1)
        else:
            raise ValueError(f"Unknown descriptor type: {desc['type']}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the nnet3 network.

        Parameters
        ----------
        features : torch.Tensor
            MFCC features, shape ``(T, D)`` where ``T`` includes
            left/right context padding.

        Returns
        -------
        torch.Tensor
            Raw nnet output, shape ``(T, num_pdfs)``.
        """
        # Initialize activations with input
        activations: dict[str, torch.Tensor] = {"input": features}

        # Evaluate component nodes in order
        for node_name in self._eval_order:
            desc = self._descriptors[node_name]
            comp_name = self._comp_map[node_name]
            safe_name = comp_name.replace(".", "_").replace("-", "_")

            # Get input from descriptor
            input_tensor = self._evaluate_descriptor(desc, activations)

            # Apply component
            module = self.component_modules[safe_name]
            activations[node_name] = module(input_tensor)

        # Evaluate output descriptor
        output = self._evaluate_descriptor(self._output_descriptor, activations)
        return output


# ---------------------------------------------------------------------------
# NnetScorer: high-level interface for the aligner
# ---------------------------------------------------------------------------

class TorchNnetScorer:
    """Acoustic scorer using a PyTorch nnet3 model.

    Implements the ``NnetScorer`` protocol from ``sk_align.aligner``.
    """

    def __init__(
        self,
        model: Nnet3Model,
        frame_subsampling_factor: int = 3,
        device: str = "cpu",
    ):
        self.network = Nnet3Network(model, output_name="output")
        self.network.eval()
        self.network.to(device)
        self.device = device
        self.frame_subsampling_factor = frame_subsampling_factor
        self.left_context = model.left_context
        self.right_context = model.right_context
        self.priors = model.priors

    @classmethod
    def from_model_file(
        cls,
        path: str | Path,
        frame_subsampling_factor: int = 3,
        device: str = "cpu",
    ) -> TorchNnetScorer:
        """Load a scorer from a Kaldi ``final.mdl`` file."""
        from sk_align.nnet3_model import read_nnet3_model
        model = read_nnet3_model(path)
        return cls(model, frame_subsampling_factor, device)

    def compute_log_likelihoods(self, features: np.ndarray) -> np.ndarray:
        """Compute log-likelihoods from MFCC features.

        Parameters
        ----------
        features : np.ndarray
            MFCC features, shape ``(T, D)`` e.g. ``(num_frames, 40)``.

        Returns
        -------
        np.ndarray
            Log-likelihoods, shape ``(T_sub, num_pdfs)`` where
            ``T_sub ≈ T / frame_subsampling_factor``.
        """
        T, D = features.shape

        # Pad with left and right context by repeating edge frames
        left_pad = np.repeat(features[:1], self.left_context, axis=0)
        right_pad = np.repeat(features[-1:], self.right_context, axis=0)
        padded = np.concatenate([left_pad, features, right_pad], axis=0)

        # Convert to torch
        x = torch.from_numpy(padded.astype(np.float32)).to(self.device)

        with torch.no_grad():
            output = self.network(x)  # (T_padded, num_pdfs)

        # Remove context padding from output
        output = output[self.left_context: self.left_context + T]

        # Subsample
        output = output[::self.frame_subsampling_factor]

        # Convert to numpy
        loglikes = output.cpu().numpy()

        # Subtract log-priors if available (for non-chain models)
        if self.priors is not None and self.priors.size > 0:
            log_priors = np.log(self.priors + 1e-20)
            loglikes -= log_priors

        return loglikes
