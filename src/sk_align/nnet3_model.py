"""
Reader for Kaldi nnet3 binary model format.

Parses the neural network portion of ``final.mdl`` (after the TransitionModel)
to extract:
- Network config lines (computation graph topology)
- Component weights and parameters
- AmNnetSimple metadata (left_context, right_context, priors)

Reference: kaldi/src/nnet3/nnet-nnet.cc, am-nnet-simple.cc,
           nnet-component-itf.cc, nnet-simple-component.cc,
           nnet-normalize-component.cc, nnet-tdnn-component.cc
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from sk_align.kaldi_io import (
    expect_token,
    read_binary_header,
    read_bool,
    read_float32,
    read_float_matrix,
    read_float_vector,
    read_int32,
    read_integer_vector,
    read_token,
)


# ---------------------------------------------------------------------------
# Config line (computation graph node) dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InputNode:
    name: str
    dim: int


@dataclass
class OutputNode:
    name: str
    input_descriptor: str
    objective: str = "linear"


@dataclass
class ComponentNode:
    """A node that applies a named component to its input descriptor."""
    name: str
    component: str  # name of the component (matches a ComponentDef)
    input_descriptor: str


# ---------------------------------------------------------------------------
# Component dataclasses (weights/parameters only — no forward logic)
# ---------------------------------------------------------------------------

@dataclass
class BatchNormParams:
    dim: int
    block_dim: int
    epsilon: float
    target_rms: float
    test_mode: bool
    count: float
    stats_mean: np.ndarray  # shape (dim,)
    stats_var: np.ndarray   # shape (dim,)


@dataclass
class NaturalGradientAffineParams:
    linear_params: np.ndarray  # shape (out_dim, in_dim)
    bias_params: np.ndarray    # shape (out_dim,)


@dataclass
class LinearParams:
    params: np.ndarray  # shape (out_dim, in_dim)


@dataclass
class TdnnParams:
    time_offsets: list[int]
    linear_params: np.ndarray  # shape (out_dim, in_dim * len(time_offsets))
    bias_params: np.ndarray    # shape (out_dim,)


@dataclass
class RectifiedLinearParams:
    dim: int


@dataclass
class LogSoftmaxParams:
    dim: int


@dataclass
class GeneralDropoutParams:
    dim: int
    block_dim: int
    dropout_proportion: float
    test_mode: bool = False
    continuous: bool = False


@dataclass
class NoOpParams:
    dim: int


# Union type for all component parameter types
ComponentParams = (
    BatchNormParams
    | NaturalGradientAffineParams
    | LinearParams
    | TdnnParams
    | RectifiedLinearParams
    | LogSoftmaxParams
    | GeneralDropoutParams
    | NoOpParams
)


@dataclass
class ComponentDef:
    """A named component with its type and parameters."""
    name: str
    component_type: str
    params: ComponentParams


# ---------------------------------------------------------------------------
# Full nnet3 model
# ---------------------------------------------------------------------------

@dataclass
class Nnet3Model:
    """Complete parsed nnet3 model."""
    inputs: list[InputNode]
    outputs: list[OutputNode]
    component_nodes: list[ComponentNode]
    components: dict[str, ComponentDef]
    left_context: int
    right_context: int
    priors: np.ndarray  # shape (num_pdfs,) — log-priors

    # Ordered list of all config nodes for topological evaluation
    all_nodes: list[InputNode | OutputNode | ComponentNode] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Config line parsing
# ---------------------------------------------------------------------------

def _parse_key_value(s: str) -> dict[str, str]:
    """Parse ``key=value`` pairs from a config line (after the node type)."""
    result: dict[str, str] = {}
    # Match key=value where value may contain nested parens
    i = 0
    while i < len(s):
        # Skip whitespace
        while i < len(s) and s[i] == ' ':
            i += 1
        if i >= len(s):
            break
        # Read key
        eq = s.index('=', i)
        key = s[i:eq].strip()
        i = eq + 1
        # Read value (may contain balanced parentheses)
        depth = 0
        start = i
        while i < len(s):
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            elif s[i] == ' ' and depth == 0:
                break
            i += 1
        value = s[start:i]
        result[key] = value
    return result


def parse_config_lines(lines: list[str]) -> list[InputNode | OutputNode | ComponentNode]:
    """Parse nnet3 config text lines into node objects."""
    nodes: list[InputNode | OutputNode | ComponentNode] = []
    for line in lines:
        if line.startswith("input-node"):
            kv = _parse_key_value(line[len("input-node"):])
            nodes.append(InputNode(name=kv["name"], dim=int(kv["dim"])))
        elif line.startswith("output-node"):
            kv = _parse_key_value(line[len("output-node"):])
            nodes.append(OutputNode(
                name=kv["name"],
                input_descriptor=kv["input"],
                objective=kv.get("objective", "linear"),
            ))
        elif line.startswith("component-node"):
            kv = _parse_key_value(line[len("component-node"):])
            nodes.append(ComponentNode(
                name=kv["name"],
                component=kv["component"],
                input_descriptor=kv["input"],
            ))
        # dim-range-node and other types can be added as needed
    return nodes


# ---------------------------------------------------------------------------
# Component binary readers
# ---------------------------------------------------------------------------

def _peek_token_char(f: BinaryIO) -> str:
    """Peek at the first letter of the next token (after '<').

    Kaldi's PeekToken reads past '<' and returns the character after it,
    then ungets the '<'.  We simulate this by peeking.
    """
    pos = f.tell()
    b = f.read(1)
    while b in (b' ', b'\t', b'\n', b'\r'):
        b = f.read(1)
    if b == b'<':
        char = f.read(1)
        f.seek(pos)
        return char.decode('ascii')
    f.seek(pos)
    return b.decode('ascii')


def _read_double(f: BinaryIO) -> float:
    """Read a Kaldi binary double (8-byte prefix)."""
    import struct
    nbytes = struct.unpack("b", f.read(1))[0]
    if nbytes == 8:
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Expected 8-byte double, got {nbytes}")


def _skip_optional_tokens(f: BinaryIO, token: str, end_token: str) -> str:
    """Read tokens, skipping optional fields, until we hit end_token."""
    while token != end_token:
        # Skip unknown optional fields by reading their value
        token = read_token(f)
    return token


def _read_updatable_common(f: BinaryIO, component_type: str) -> str:
    """Read UpdatableComponent common fields.

    Returns the next unconsumed token (empty string if all consumed).
    """
    token = read_token(f)
    # May start with the component type tag (already consumed by ReadNew)
    if token == f"<{component_type}>":
        token = read_token(f)
    if token == "<LearningRateFactor>":
        _lrf = read_float32(f)
        token = read_token(f)
    if token == "<IsGradient>":
        _ig = read_bool(f)
        token = read_token(f)
    if token == "<MaxChange>":
        _mc = read_float32(f)
        token = read_token(f)
    if token == "<L2Regularize>":
        _l2 = read_float32(f)
        token = read_token(f)
    if token == "<LearningRate>":
        _lr = read_float32(f)
        return ""
    return token


def read_batch_norm(f: BinaryIO) -> BatchNormParams:
    """Read BatchNormComponent."""
    # Opening tag already consumed by ReadNew.
    # ExpectOneOrTwoTokens: may read "<BatchNormComponent>" then "<Dim>",
    # or just "<Dim>" if opening tag already consumed.
    token = read_token(f)
    if token == "<BatchNormComponent>":
        token = read_token(f)
    assert token == "<Dim>", f"Expected <Dim>, got {token!r}"
    dim = read_int32(f)

    expect_token(f, "<BlockDim>")
    block_dim = read_int32(f)

    expect_token(f, "<Epsilon>")
    epsilon = read_float32(f)

    expect_token(f, "<TargetRms>")
    target_rms = read_float32(f)

    expect_token(f, "<TestMode>")
    test_mode = read_bool(f)

    expect_token(f, "<Count>")
    count = read_float32(f)

    expect_token(f, "<StatsMean>")
    stats_mean = read_float_vector(f)

    expect_token(f, "<StatsVar>")
    stats_var = read_float_vector(f)

    expect_token(f, "</BatchNormComponent>")

    return BatchNormParams(
        dim=dim,
        block_dim=block_dim,
        epsilon=epsilon,
        target_rms=target_rms,
        test_mode=test_mode,
        count=count,
        stats_mean=stats_mean,
        stats_var=stats_var,
    )


def read_natural_gradient_affine(f: BinaryIO) -> NaturalGradientAffineParams:
    """Read NaturalGradientAffineComponent."""
    token = _read_updatable_common(f, "NaturalGradientAffineComponent")
    if token:
        assert token == "<LinearParams>", f"Expected <LinearParams>, got {token!r}"
    else:
        expect_token(f, "<LinearParams>")
    linear_params = read_float_matrix(f)

    expect_token(f, "<BiasParams>")
    bias_params = read_float_vector(f)

    expect_token(f, "<RankIn>")
    _rank_in = read_int32(f)

    expect_token(f, "<RankOut>")
    _rank_out = read_int32(f)

    # Optional fields
    c = _peek_token_char(f)
    if c == 'O':
        expect_token(f, "<OrthonormalConstraint>")
        _oc = read_float32(f)

    expect_token(f, "<UpdatePeriod>")
    _update_period = read_int32(f)

    expect_token(f, "<NumSamplesHistory>")
    _nsh = read_float32(f)

    expect_token(f, "<Alpha>")
    _alpha = read_float32(f)

    # Back-compat optional fields
    c = _peek_token_char(f)
    if c == 'M':
        expect_token(f, "<MaxChangePerSample>")
        _mcs = read_float32(f)
        c = _peek_token_char(f)
    if c == 'I':
        expect_token(f, "<IsGradient>")
        _ig = read_bool(f)
        c = _peek_token_char(f)
    if c == 'U':
        expect_token(f, "<UpdateCount>")
        _uc = _read_double(f)
        expect_token(f, "<ActiveScalingCount>")
        _asc = _read_double(f)
        expect_token(f, "<MaxChangeScaleStats>")
        _mcss = _read_double(f)

    # Read closing tag
    token = read_token(f)
    assert token.startswith("</"), f"Expected closing tag, got {token!r}"

    return NaturalGradientAffineParams(
        linear_params=linear_params,
        bias_params=bias_params,
    )


def read_linear_component(f: BinaryIO) -> LinearParams:
    """Read LinearComponent."""
    token = _read_updatable_common(f, "LinearComponent")
    if token:
        assert token == "<Params>", f"Expected <Params>, got {token!r}"
    else:
        expect_token(f, "<Params>")
    params = read_float_matrix(f)

    # Optional OrthonormalConstraint
    c = _peek_token_char(f)
    if c == 'O':
        expect_token(f, "<OrthonormalConstraint>")
        _oc = read_float32(f)

    expect_token(f, "<UseNaturalGradient>")
    _ung = read_bool(f)

    expect_token(f, "<RankInOut>")
    _rank_in = read_int32(f)
    _rank_out = read_int32(f)

    expect_token(f, "<Alpha>")
    _alpha = read_float32(f)

    expect_token(f, "<NumSamplesHistory>")
    _nsh = read_float32(f)

    expect_token(f, "<UpdatePeriod>")
    _up = read_int32(f)

    expect_token(f, "</LinearComponent>")

    return LinearParams(params=params)


def read_tdnn_component(f: BinaryIO) -> TdnnParams:
    """Read TdnnComponent."""
    token = _read_updatable_common(f, "TdnnComponent")
    if token:
        assert token == "<TimeOffsets>", f"Expected <TimeOffsets>, got {token!r}"
    else:
        expect_token(f, "<TimeOffsets>")
    time_offsets = read_integer_vector(f)

    expect_token(f, "<LinearParams>")
    linear_params = read_float_matrix(f)

    expect_token(f, "<BiasParams>")
    bias_params = read_float_vector(f)

    expect_token(f, "<OrthonormalConstraint>")
    _oc = read_float32(f)

    expect_token(f, "<UseNaturalGradient>")
    _ung = read_bool(f)

    expect_token(f, "<NumSamplesHistory>")
    _nsh = read_float32(f)

    token = read_token(f)
    if token == "<AlphaInOut>":
        _alpha_in = read_float32(f)
        _alpha_out = read_float32(f)
    elif token == "<Alpha>":
        _alpha = read_float32(f)
    else:
        raise ValueError(f"Expected <AlphaInOut> or <Alpha>, got {token!r}")

    expect_token(f, "<RankInOut>")
    _rank_in = read_int32(f)
    _rank_out = read_int32(f)

    expect_token(f, "</TdnnComponent>")

    return TdnnParams(
        time_offsets=time_offsets,
        linear_params=linear_params,
        bias_params=bias_params,
    )


def read_nonlinear_component(f: BinaryIO, component_type: str) -> RectifiedLinearParams | LogSoftmaxParams:
    """Read NonlinearComponent (ReLU, LogSoftmax, etc.)."""
    open_tag = f"<{component_type}>"
    close_tag = f"</{component_type}>"

    token = read_token(f)
    if token == open_tag:
        token = read_token(f)
    assert token == "<Dim>", f"Expected <Dim>, got {token!r}"
    dim = read_int32(f)

    # Optional BlockDim
    c = _peek_token_char(f)
    if c == 'B':
        expect_token(f, "<BlockDim>")
        _block_dim = read_int32(f)

    expect_token(f, "<ValueAvg>")
    _value_avg = read_float_vector(f)

    expect_token(f, "<DerivAvg>")
    _deriv_avg = read_float_vector(f)

    expect_token(f, "<Count>")
    _count = read_float32(f)

    # Optional OderivRms
    c = _peek_token_char(f)
    if c == 'O':
        expect_token(f, "<OderivRms>")
        _oderiv_rms = read_float_vector(f)
        expect_token(f, "<OderivCount>")
        _oderiv_count = read_float32(f)

    # Read remaining optional tokens until closing tag
    token = read_token(f)
    while token != close_tag:
        # Skip optional fields: NumDimsSelfRepaired, NumDimsProcessed,
        # SelfRepairLowerThreshold, SelfRepairUpperThreshold, SelfRepairScale
        if token.startswith("<") and not token.startswith("</"):
            _val = read_float32(f)
        token = read_token(f)

    if component_type == "RectifiedLinearComponent":
        return RectifiedLinearParams(dim=dim)
    elif component_type == "LogSoftmaxComponent":
        return LogSoftmaxParams(dim=dim)
    else:
        return RectifiedLinearParams(dim=dim)  # fallback


def read_general_dropout(f: BinaryIO) -> GeneralDropoutParams:
    """Read GeneralDropoutComponent."""
    token = read_token(f)
    if token == "<GeneralDropoutComponent>":
        token = read_token(f)
    assert token == "<Dim>", f"Expected <Dim>, got {token!r}"
    dim = read_int32(f)

    expect_token(f, "<BlockDim>")
    block_dim = read_int32(f)

    expect_token(f, "<TimePeriod>")
    _time_period = read_int32(f)

    expect_token(f, "<DropoutProportion>")
    dropout_proportion = read_float32(f)

    test_mode = False
    continuous = False

    # Optional SpecAugment fields
    c = _peek_token_char(f)
    if c == 'S':
        expect_token(f, "<SpecAugmentMaxProportion>")
        _samp = read_float32(f)
        c = _peek_token_char(f)
        if c == 'S':
            expect_token(f, "<SpecAugmentMaxRegions>")
            _samr = read_int32(f)
        c = _peek_token_char(f)

    # TestMode is just a flag token (no value)
    c = _peek_token_char(f)
    if c == 'T':
        expect_token(f, "<TestMode>")
        test_mode = True
        c = _peek_token_char(f)

    # Continuous is just a flag token (no value)
    if c == 'C':
        expect_token(f, "<Continuous>")
        continuous = True

    expect_token(f, "</GeneralDropoutComponent>")

    return GeneralDropoutParams(
        dim=dim,
        block_dim=block_dim,
        dropout_proportion=dropout_proportion,
        test_mode=test_mode,
        continuous=continuous,
    )


def read_noop_component(f: BinaryIO) -> NoOpParams:
    """Read NoOpComponent."""
    token = read_token(f)
    if token == "<NoOpComponent>":
        token = read_token(f)
    assert token == "<Dim>", f"Expected <Dim>, got {token!r}"
    dim = read_int32(f)

    # Check for old vs new format
    c = _peek_token_char(f)
    if c == 'V':
        # Old format (NonlinearComponent fields)
        expect_token(f, "<ValueAvg>")
        _va = read_float_vector(f)
        expect_token(f, "<DerivAvg>")
        _da = read_float_vector(f)
        expect_token(f, "<Count>")
        _count = read_float32(f)
        # Read optional fields and closing tag
        token = read_token(f)
        while token != "</NoOpComponent>":
            if token.startswith("<") and not token.startswith("</"):
                _val = read_float32(f)
            token = read_token(f)
    else:
        # New format
        expect_token(f, "<BackpropScale>")
        _bps = read_float32(f)
        expect_token(f, "</NoOpComponent>")

    return NoOpParams(dim=dim)


# ---------------------------------------------------------------------------
# Component dispatch
# ---------------------------------------------------------------------------

_COMPONENT_READERS: dict[str, Any] = {
    "BatchNormComponent": read_batch_norm,
    "NaturalGradientAffineComponent": read_natural_gradient_affine,
    "LinearComponent": read_linear_component,
    "TdnnComponent": read_tdnn_component,
    "RectifiedLinearComponent": lambda f: read_nonlinear_component(f, "RectifiedLinearComponent"),
    "LogSoftmaxComponent": lambda f: read_nonlinear_component(f, "LogSoftmaxComponent"),
    "GeneralDropoutComponent": read_general_dropout,
    "NoOpComponent": read_noop_component,
}


def read_component(f: BinaryIO) -> tuple[str, ComponentParams]:
    """Read a single Component via ReadNew dispatch.

    Returns (component_type, params).
    """
    # ReadNew: read <ComponentType> token
    token = read_token(f)
    # Strip angle brackets to get type name
    assert token.startswith("<") and token.endswith(">"), f"Expected <Type>, got {token!r}"
    component_type = token[1:-1]

    reader = _COMPONENT_READERS.get(component_type)
    if reader is None:
        raise NotImplementedError(f"Unsupported component type: {component_type}")

    params = reader(f)
    return component_type, params


# ---------------------------------------------------------------------------
# Full model reader
# ---------------------------------------------------------------------------

def read_nnet3_model(path: str | Path) -> Nnet3Model:
    """Read a complete nnet3 alignment model from ``final.mdl``.

    Parses both the TransitionModel (skipped) and the nnet3 acoustic model.
    """
    from sk_align.transition_model import TransitionModel

    with open(path, "rb") as f:
        read_binary_header(f)

        # 1. Skip TransitionModel
        _tm = TransitionModel.read(f)

        # 2. Read Nnet
        expect_token(f, "<Nnet3>")

        # Read newline after <Nnet3> token
        c = f.read(1)
        while c in (b'\r',):
            c = f.read(1)
        # c should be '\n' now

        # Read config lines (text, terminated by empty line)
        config_lines: list[str] = []
        while True:
            line_bytes = b''
            while True:
                c = f.read(1)
                if c == b'\n' or not c:
                    break
                line_bytes += c
            line = line_bytes.decode('ascii').strip()
            if not line:
                break
            config_lines.append(line)

        # Parse config lines
        all_nodes = parse_config_lines(config_lines)

        # Read components
        expect_token(f, "<NumComponents>")
        num_components = read_int32(f)

        components: dict[str, ComponentDef] = {}
        for _ in range(num_components):
            expect_token(f, "<ComponentName>")
            comp_name = read_token(f)
            comp_type, params = read_component(f)
            components[comp_name] = ComponentDef(
                name=comp_name,
                component_type=comp_type,
                params=params,
            )

        expect_token(f, "</Nnet3>")

        # 3. Read AmNnetSimple metadata
        expect_token(f, "<LeftContext>")
        left_context = read_int32(f)

        expect_token(f, "<RightContext>")
        right_context = read_int32(f)

        expect_token(f, "<Priors>")
        priors = read_float_vector(f)

    # Organize nodes
    inputs = [n for n in all_nodes if isinstance(n, InputNode)]
    outputs = [n for n in all_nodes if isinstance(n, OutputNode)]
    component_nodes = [n for n in all_nodes if isinstance(n, ComponentNode)]

    return Nnet3Model(
        inputs=inputs,
        outputs=outputs,
        component_nodes=component_nodes,
        components=components,
        left_context=left_context,
        right_context=right_context,
        priors=priors,
        all_nodes=all_nodes,
    )
