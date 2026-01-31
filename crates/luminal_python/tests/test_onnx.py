"""Tests for the luminal_python torch.compile backend."""

import luminal_python  # noqa: F401 - import registers torch.compile backends
import torch
from torch import nn


class BasicTransformerLM(nn.Module):
    """
    Minimal encoder-style Transformer for language-model-ish next-token logits.
    Input:  (B, T) token ids
    Output: (B, T, vocab_size) logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, C)
            norm_first=True,  # pre-norm = a bit more stable
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        x: LongTensor (B, T)
        """
        B, T = x.shape
        device = x.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)  # (B, T, C)

        # Padding mask: True where tokens should be ignored
        key_padding_mask = x == self.pad_id  # (B, T)

        # Causal mask so each position can't see the future (LM-style)
        # True/inf above diagonal => disallow attention
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        h = self.encoder(
            h,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        logits = self.lm_head(h)  # (B, T, vocab)
        return logits


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.linear_2 = nn.Linear(10, 10)

    def forward(self, x):
        first_output = self.linear(x)
        second_output = x + first_output
        return self.linear_2(second_output)


class MulDivModel(nn.Module):
    """Element-wise Mul and Div with broadcasting."""

    def __init__(self, features):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(features))
        self.divisor = nn.Parameter(torch.rand(1) + 0.5)

    def forward(self, x):
        scaled = x * self.scale
        return scaled / self.divisor


class SqrtDivModel(nn.Module):
    """Sqrt and Div in a chain (ensures positive input to sqrt)."""

    def __init__(self, features):
        super().__init__()
        self.offset = nn.Parameter(torch.rand(features) + 0.1)

    def forward(self, x):
        x_pos = x * x + self.offset
        root = torch.sqrt(x_pos)
        return x / root


class SoftmaxModel(nn.Module):
    """Softmax on the last dimension."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


class LayerNormModel(nn.Module):
    """LayerNorm applied to linear output."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        return self.norm(self.linear(x))


class ErfModel(nn.Module):
    """torch.erf applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.erf(self.linear(x))


class GeluModel(nn.Module):
    """GELU activation (decomposes to Erf ops in ONNX)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.linear(x))


class ConcatModel(nn.Module):
    """Two parallel branches concatenated along feature dim."""

    def __init__(self, in_features, hidden):
        super().__init__()
        self.branch_a = nn.Linear(in_features, hidden)
        self.branch_b = nn.Linear(in_features, hidden)

    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)
        return torch.cat([a, b], dim=-1)


class SliceModel(nn.Module):
    """Slices first half of features, then applies linear."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.half = in_features // 2
        self.linear = nn.Linear(self.half, out_features)

    def forward(self, x):
        sliced = x[:, : self.half]
        return self.linear(sliced)


class ConcatSliceModel(nn.Module):
    """Concat two branches then slice back to original hidden size."""

    def __init__(self, features, hidden):
        super().__init__()
        self.hidden = hidden
        self.proj_a = nn.Linear(features, hidden)
        self.proj_b = nn.Linear(features, hidden)
        self.final_proj = nn.Linear(hidden, features)

    def forward(self, x):
        a = self.proj_a(x)
        b = self.proj_b(x)
        combined = torch.cat([a, b], dim=-1)
        sliced = combined[:, : self.hidden]
        return self.final_proj(sliced)


class ResidualBlock(nn.Module):
    """Residual Add + LayerNorm pattern."""

    def __init__(self, features):
        super().__init__()
        self.linear1 = nn.Linear(features, features)
        self.linear2 = nn.Linear(features, features)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = x + residual
        x = self.norm(x)
        return self.linear2(x)


class SimpleAttention(nn.Module):
    """Single-head attention: Q@K^T/sqrt(d), softmax, @V."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model**0.5

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        k_t = k.transpose(-2, -1)
        scores = torch.matmul(q, k_t) / self.scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class MultiLayerMLP(nn.Module):
    """3-layer MLP with GELU activations."""

    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_features)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def assert_tensors_close(result, expected, tol):
    """Compare result vs expected and assert within tolerance."""
    diff = (result - expected).abs().max().item()
    assert result.shape == expected.shape, (
        f"Shape mismatch: {result.shape} vs {expected.shape}"
    )
    assert diff < tol, f"Max difference {diff:.2e} exceeds tolerance {tol}"


def test_simple_linear(backend):
    """Test simple linear model with residual connection."""
    model = SimpleModel()
    model.eval()
    x = torch.randn(2, 10)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_elementwise_mul_div(backend):
    """Test Mul and Div with broadcasting."""
    model = MulDivModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_sqrt_div_model(backend):
    """Test Sqrt and Div in a chain."""
    model = SqrtDivModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_softmax_standalone(backend):
    """Test Softmax applied to linear output."""
    model = SoftmaxModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_layer_norm_standalone(backend):
    """Test LayerNorm in isolation."""
    model = LayerNormModel(features=32)
    model.eval()
    x = torch.randn(4, 32)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-4)


def test_erf_standalone(backend):
    """Test torch.erf (approximated in Rust)."""
    model = ErfModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=5e-3)


def test_gelu_activation(backend):
    """Test GELU activation (Erf-based decomposition in ONNX)."""
    model = GeluModel(in_features=16, out_features=32)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=5e-3)


def test_concat_model(backend):
    """Test Concat along feature dimension."""
    model = ConcatModel(in_features=16, hidden=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_slice_model(backend):
    """Test Slice on feature dimension."""
    model = SliceModel(in_features=16, out_features=6)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_concat_slice_roundtrip(backend):
    """Test Concat followed by Slice."""
    model = ConcatSliceModel(features=16, hidden=12)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_residual_block(backend):
    """Test residual Add + LayerNorm pattern."""
    model = ResidualBlock(features=32)
    model.eval()
    x = torch.randn(4, 32)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-4)


def test_matmul_transpose_attention(backend):
    """Test MatMul, Transpose, Div, Softmax via single-head attention."""
    model = SimpleAttention(d_model=32)
    model.eval()
    x = torch.randn(2, 8, 32)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-4)


def test_multi_layer_mlp(backend):
    """Test 3-layer MLP with GELU activations."""
    model = MultiLayerMLP(in_features=16, hidden=32, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=5e-3)


def test_simple_transformer(backend):
    return
    """Test Simple transformer."""
    vocab_size = 10_000
    model = BasicTransformerLM(
        vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2, max_len=256
    )
    model.eval()
    x = torch.randint(0, vocab_size, (2, 32))  # (batch=2, seq=32)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


# --- New ONNX Op Test Models ---


class TanhModel(nn.Module):
    """Tanh activation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class CosModel(nn.Module):
    """Cos activation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.cos(self.linear(x))


class SinModel(nn.Module):
    """Sin activation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.sin(self.linear(x))


class PowModel(nn.Module):
    """Pow operation: base raised to exponent power."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Use abs to avoid NaN from negative bases with fractional exponents
        base = torch.abs(self.linear(x)) + 0.1
        return torch.pow(base, 2.0)


class ReduceMeanModel(nn.Module):
    """ReduceMean along last dimension."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return torch.mean(x, dim=-1, keepdim=True)


class NegModel(nn.Module):
    """Negation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.neg(self.linear(x))


class SigmoidModel(nn.Module):
    """Sigmoid activation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class SplitEqualModel(nn.Module):
    """Split tensor into equal parts along feature dimension."""

    def __init__(self, in_features, num_splits):
        super().__init__()
        self.num_splits = num_splits
        self.linear = nn.Linear(in_features, in_features)

    def forward(self, x):
        x = self.linear(x)
        splits = torch.split(x, x.shape[-1] // self.num_splits, dim=-1)
        return splits[0] + splits[1]


class SplitUnequalModel(nn.Module):
    """Split tensor into unequal parts."""

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)

    def forward(self, x):
        x = self.linear(x)
        # Split into [4, 8, 4] for 16 features
        splits = torch.split(x, [4, 8, 4], dim=-1)
        return splits[1]  # Return middle chunk


class GatherEmbeddingModel(nn.Module):
    """Gather elements using indices (embedding lookup pattern)."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, indices):
        return self.embedding(indices)


class ExpandBroadcastModel(nn.Module):
    """Expand tensor via broadcasting addition."""

    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(1, features))

    def forward(self, x):
        # x: (batch, seq, features), bias: (1, features)
        # This triggers Expand to broadcast bias
        return x + self.bias


class ReshapeViewModel(nn.Module):
    """Reshape tensor to different dimensions via view."""

    def __init__(self, in_features, hidden):
        super().__init__()
        self.hidden = hidden
        self.linear = nn.Linear(in_features, hidden * 4)

    def forward(self, x):
        batch = x.shape[0]
        x = self.linear(x)
        return x.view(batch, 4, self.hidden)


class ReshapeFlattenModel(nn.Module):
    """Flatten tensor dimensions using view."""

    def __init__(self, seq_len, features):
        super().__init__()
        self.seq_len = seq_len
        self.features = features
        self.linear = nn.Linear(features, features)

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, seq * features)
        batch = x.shape[0]
        x = self.linear(x)
        return x.view(batch, self.seq_len * self.features)


class UnsqueezeModel(nn.Module):
    """Add dimension via unsqueeze."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, x):
        x = self.linear(x)
        # (batch, features) -> (batch, 1, features)
        return x.unsqueeze(1)


class SqueezeModel(nn.Module):
    """Remove dimension via squeeze."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, x):
        x = self.linear(x)
        # (batch, 1, features) -> (batch, features)
        return x.squeeze(1)


class EqualModel(nn.Module):
    """Equality comparison."""

    def __init__(self):
        super().__init__()
        self.register_buffer("target", torch.tensor([0.0]))

    def forward(self, x):
        return (x == self.target).float()


class ModModel(nn.Module):
    """Element-wise modulo operation."""

    def __init__(self, divisor):
        super().__init__()
        self.register_buffer("divisor", torch.tensor([float(divisor)]))

    def forward(self, x):
        return torch.fmod(x, self.divisor)


class CastModel(nn.Module):
    """Type casting operations (float -> int -> float)."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, x):
        x = self.linear(x)
        x_int = x.int()
        return x_int.float()


# --- Trilu (Triangular) Models ---


class TriluUpperModel(nn.Module):
    """Extract upper triangular part of matrix."""

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        x = self.linear(x)
        # Reshape to square matrix for triu
        batch = x.shape[0]
        x = x.view(batch, int(self.size**0.5), int(self.size**0.5))
        return torch.triu(x)


class TriluLowerModel(nn.Module):
    """Extract lower triangular part of matrix."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return torch.tril(x)


class TriluDiagonalOffsetModel(nn.Module):
    """Trilu with diagonal offset k."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # k=1 means one diagonal above main
        return torch.triu(x, diagonal=1)


# --- Where (Conditional) Models ---


class WhereEqualModel(nn.Module):
    """Where with Equal condition (selecting based on zero elements)."""

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.register_buffer("zero", torch.tensor([0.0]))

    def forward(self, x):
        # Condition: where x equals 0, use replacement value
        condition = x == self.zero
        replacement = torch.ones_like(x) * -1.0
        return torch.where(condition, replacement, x)


class WhereBroadcastModel(nn.Module):
    """Where with broadcasting boolean mask."""

    def __init__(self, features):
        super().__init__()
        self.features = features
        # Create a mask that broadcasts
        self.register_buffer("mask", torch.tensor([True, False] * (features // 2)))
        # Alternative value
        self.register_buffer("alt_value", torch.zeros(features))

    def forward(self, x):
        # mask: (features,) broadcasts to (batch, features)
        return torch.where(self.mask, x, self.alt_value)


class WhereTwoTensorModel(nn.Module):
    """Where selecting between two tensors based on Equal condition."""

    def __init__(self, features):
        super().__init__()
        self.linear_a = nn.Linear(features, features)
        self.linear_b = nn.Linear(features, features)
        self.register_buffer("zero", torch.tensor([0.0]))

    def forward(self, x):
        a = self.linear_a(x)
        b = self.linear_b(x)
        # Condition based on Equal
        condition = x == self.zero
        return torch.where(condition, a, b)


# --- ConstantOfShape Models ---


class ConstantOfShapeZerosModel(nn.Module):
    """Creates zeros tensor with computed shape."""

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        batch = x.shape[0]
        # Create zeros and add to input (tests ConstantOfShape with value=0)
        zeros = torch.zeros(batch, self.features, device=x.device, dtype=x.dtype)
        return x + zeros


class ConstantOfShapeOnesModel(nn.Module):
    """Creates ones tensor and multiplies."""

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        batch = x.shape[0]
        # Create ones and multiply (tests ConstantOfShape with value=1)
        ones = torch.ones(batch, self.features, device=x.device, dtype=x.dtype)
        return x * ones


class ConstantOfShapeFullModel(nn.Module):
    """Creates tensor filled with arbitrary constant."""

    def __init__(self, features, fill_value):
        super().__init__()
        self.features = features
        self.fill_value = fill_value

    def forward(self, x):
        batch = x.shape[0]
        # Create filled tensor (tests ConstantOfShape with custom value)
        filled = torch.full(
            (batch, self.features), self.fill_value, device=x.device, dtype=x.dtype
        )
        return x + filled


# --- Shape-based Model ---


class ShapeBasedReshapeModel(nn.Module):
    """Model that uses tensor shape for reshaping."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features * 2)

    def forward(self, x):
        # Uses shape internally: reshape to (batch, 2, features)
        batch_size = x.shape[0]
        x = self.linear(x)
        return x.view(batch_size, 2, -1)


# --- Gather Edge Case Models ---


class GatherLargeVocabModel(nn.Module):
    """Gather with large vocabulary (tests index precision)."""

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, indices):
        return self.embedding(indices)


# --- Expand Edge Case Models ---


class ExpandMultiDimModel(nn.Module):
    """Expand tensor across multiple dimensions."""

    def __init__(self, features):
        super().__init__()
        # Scalar-like parameter that expands to full tensor
        self.scale = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # scale: (1,) expands to x's shape (batch, seq, features)
        return x * self.scale


class ExpandPrependDimsModel(nn.Module):
    """Expand by prepending dimensions."""

    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(features))

    def forward(self, x):
        # bias: (features,) needs batch and seq dims prepended
        # x: (batch, seq, features)
        return x + self.bias


# --- Split Edge Case Models ---


class SplitBatchDimModel(nn.Module):
    """Split along axis 0 (batch dimension)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split batch in half
        splits = torch.split(x, x.shape[0] // 2, dim=0)
        return splits[0] + splits[1]


class SplitMultipleChunksModel(nn.Module):
    """Split into 3 chunks and concatenate first two."""

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)
        # in_features should be divisible by 3

    def forward(self, x):
        x = self.linear(x)
        # Split into 3 equal chunks
        chunk_size = x.shape[-1] // 3
        chunks = torch.split(x, chunk_size, dim=-1)
        # Concatenate first two chunks
        return torch.cat([chunks[0], chunks[1]], dim=-1)


# --- New ONNX Op Tests ---


def test_tanh_standalone(backend):
    """Test torch.tanh activation."""
    model = TanhModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-4)


def test_cos_standalone(backend):
    """Test torch.cos function."""
    model = CosModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-4)


def test_sin_standalone(backend):
    """Test torch.sin function."""
    model = SinModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-4)


def test_pow_standalone(backend):
    """Test torch.pow function."""
    model = PowModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-4)


def test_reduce_mean_standalone(backend):
    """Test torch.mean reduction."""
    model = ReduceMeanModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-4)


def test_neg_standalone(backend):
    """Test torch.neg function."""
    model = NegModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_sigmoid_standalone(backend):
    """Test torch.sigmoid function."""
    model = SigmoidModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-4)


def test_split_equal(backend):
    """Test Split with equal division."""
    model = SplitEqualModel(in_features=16, num_splits=4)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_split_unequal(backend):
    """Test Split with unequal split sizes."""
    model = SplitUnequalModel(in_features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_gather_embedding(backend):
    """Test Gather via embedding lookup (axis=0)."""
    model = GatherEmbeddingModel(num_embeddings=100, embedding_dim=32)
    model.eval()
    indices = torch.randint(0, 100, (4, 8))
    compiled = torch.compile(model, backend=backend)
    result = compiled(indices)
    with torch.no_grad():
        expected = model(indices)
    assert_tensors_close(result, expected, tol=1e-5)


def test_expand_broadcast(backend):
    """Test Expand via broadcasting."""
    model = ExpandBroadcastModel(features=32)
    model.eval()
    x = torch.randn(4, 8, 32)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_reshape_view(backend):
    """Test Reshape via view operation."""
    model = ReshapeViewModel(in_features=16, hidden=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_reshape_flatten(backend):
    """Test Reshape via view to flatten."""
    model = ReshapeFlattenModel(seq_len=8, features=16)
    model.eval()
    x = torch.randn(4, 8, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_unsqueeze(backend):
    """Test Unsqueeze dimension insertion."""
    model = UnsqueezeModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_squeeze(backend):
    """Test Squeeze dimension removal."""
    model = SqueezeModel(features=16)
    model.eval()
    x = torch.randn(4, 1, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_equal(backend):
    """Test Equal comparison."""
    model = EqualModel()
    model.eval()
    x = torch.tensor([[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0]])
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_mod(backend):
    """Test Mod element-wise modulo."""
    model = ModModel(divisor=3.0)
    model.eval()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_cast(backend):
    """Test Cast type conversion."""
    model = CastModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


# --- Trilu Tests ---


def test_trilu_upper(backend):
    """Test Trilu upper triangular extraction."""
    model = TriluUpperModel(size=16)
    model.eval()
    x = torch.randn(2, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_trilu_lower(backend):
    """Test Trilu lower triangular extraction."""
    model = TriluLowerModel(size=8)
    model.eval()
    x = torch.randn(4, 8, 8)  # 3D with square last two dims
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_trilu_diagonal_offset(backend):
    """Test Trilu with diagonal offset."""
    model = TriluDiagonalOffsetModel()
    model.eval()
    x = torch.randn(2, 6, 6)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


# --- Where Tests ---


def test_where_equal(backend):
    """Test Where with Equal condition."""
    model = WhereEqualModel(features=16)
    model.eval()
    # Use tensor with some zeros to trigger the condition
    x = torch.tensor(
        [
            [
                0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0,
                0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0,
            ]
        ]
    )
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_where_broadcast(backend):
    """Test Where with broadcasting condition."""
    model = WhereBroadcastModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_where_two_tensors(backend):
    """Test Where selecting between two computed tensors."""
    model = WhereTwoTensorModel(features=16)
    model.eval()
    # Use tensor with some zeros to trigger the condition
    x = torch.tensor(
        [
            [
                0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0,
                0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0,
            ],
            [
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0,
                5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
            ],
        ]
    )
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


# --- ConstantOfShape Tests ---


def test_constant_of_shape_zeros(backend):
    """Test ConstantOfShape with zeros."""
    model = ConstantOfShapeZerosModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_constant_of_shape_ones(backend):
    """Test ConstantOfShape with ones."""
    model = ConstantOfShapeOnesModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_constant_of_shape_full(backend):
    """Test ConstantOfShape with custom fill value."""
    model = ConstantOfShapeFullModel(features=16, fill_value=0.5)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


# --- Shape Test ---


def test_shape_based_reshape(backend):
    """Test Shape operation via reshape with dynamic batch."""
    model = ShapeBasedReshapeModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


# --- Gather Edge Case Tests ---


def test_gather_large_vocab(backend):
    """Test Gather with large vocabulary (tests index precision)."""
    vocab_size = 50304  # GPT-2 vocab size
    model = GatherLargeVocabModel(vocab_size=vocab_size, embedding_dim=32)
    model.eval()
    # Include indices near the end of vocab to stress precision
    indices = torch.randint(0, vocab_size, (2, 16))
    compiled = torch.compile(model, backend=backend)
    result = compiled(indices)
    with torch.no_grad():
        expected = model(indices)
    assert_tensors_close(result, expected, tol=1e-5)


# --- Expand Edge Case Tests ---


def test_expand_multi_dim(backend):
    """Test Expand across multiple dimensions."""
    model = ExpandMultiDimModel(features=16)
    model.eval()
    x = torch.randn(2, 8, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_expand_prepend_dims(backend):
    """Test Expand with prepended dimensions."""
    model = ExpandPrependDimsModel(features=16)
    model.eval()
    x = torch.randn(2, 8, 16)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


# --- Split Edge Case Tests ---


def test_split_batch_dim(backend):
    """Test Split along batch dimension."""
    model = SplitBatchDimModel()
    model.eval()
    x = torch.randn(4, 16)  # 4 batches, split into 2+2
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)


def test_split_multiple_chunks(backend):
    """Test Split into multiple chunks."""
    model = SplitMultipleChunksModel(in_features=24)
    model.eval()
    x = torch.randn(4, 24)
    compiled = torch.compile(model, backend=backend)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    assert_tensors_close(result, expected, tol=1e-5)
