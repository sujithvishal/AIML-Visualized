import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ── Code blocks ────────────────────────────────────────────────────────────────

const CODE_FFN = `import tensorflow as tf

# ── Position-wise Feed-Forward Network ──
# Applied identically at every position: FFN(x) = max(0, xW₁+b₁)W₂+b₂
# d_model=512, d_ff=2048 in "Attention is All You Need"

class PositionwiseFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.dense1   = tf.keras.layers.Dense(d_ff,     activation='relu')
        self.dense2   = tf.keras.layers.Dense(d_model)
        self.dropout  = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # x: (batch, T, d_model)
        out = self.dense1(x)          # (batch, T, d_ff)   — expand
        out = self.dropout(out, training=training)
        out = self.dense2(out)        # (batch, T, d_model) — project back
        return out

# Quick check
ffn = PositionwiseFFN(d_model=64, d_ff=256)
x   = tf.random.normal((2, 10, 64))   # batch=2, T=10, d_model=64
print("FFN output:", ffn(x).shape)    # (2, 10, 64) — shape preserved
print("d_ff acts as a bottleneck expansion (×4) then compression")`

const CODE_LAYERNORM = `import tensorflow as tf
import numpy as np

# ── Layer Normalisation vs Batch Normalisation ──

# Batch Norm: normalises across the batch dimension (problematic for variable-len seqs)
# Layer Norm: normalises across the feature dimension (each token independently)

class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = self.add_weight("gamma", shape=(d_model,), initializer="ones")
        self.beta  = self.add_weight("beta",  shape=(d_model,), initializer="zeros")
        self.eps   = eps

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta   # learnable scale + shift

# Residual + Layer Norm sublayer wrapper
class SublayerConnection(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm    = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, sublayer_fn, training=False):
        # Pre-norm variant: norm(x) is passed to sublayer, then add residual
        return x + self.dropout(sublayer_fn(self.norm(x)), training=training)

# Demonstrate: residual keeps gradient magnitude healthy
x   = tf.random.normal((2, 10, 64))
ln  = tf.keras.layers.LayerNormalization(epsilon=1e-6)
out = ln(x)
print("LayerNorm mean (≈0):", tf.reduce_mean(out).numpy().round(4))
print("LayerNorm std  (≈1):", tf.math.reduce_std(out).numpy().round(4))`

const CODE_ENCODER_BLOCK = `import tensorflow as tf

# ── Single Transformer Encoder Block ──
# Sub-layer 1: Multi-Head Self-Attention  + Residual + LayerNorm
# Sub-layer 2: Position-wise FFN          + Residual + LayerNorm

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha     = tf.keras.layers.MultiHeadAttention(
                            num_heads=num_heads, key_dim=d_model // num_heads,
                            dropout=dropout)
        self.ffn1    = tf.keras.layers.Dense(d_ff,     activation='relu')
        self.ffn2    = tf.keras.layers.Dense(d_model)
        self.norm1   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1   = tf.keras.layers.Dropout(dropout)
        self.drop2   = tf.keras.layers.Dropout(dropout)

    def call(self, x, padding_mask=None, training=False):
        # Sub-layer 1: self-attention (Q=K=V=x)
        attn_out = self.mha(x, x, x, attention_mask=padding_mask,
                            training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))

        # Sub-layer 2: FFN
        ffn_out = self.ffn2(tf.nn.relu(self.ffn1(x)))
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x

# Stack N encoder blocks → Transformer Encoder
def build_encoder(vocab_size, max_len, d_model, num_heads, d_ff, N, dropout=0.1):
    inp     = tf.keras.Input(shape=(None,), dtype=tf.int32)
    tok_emb = tf.keras.layers.Embedding(vocab_size, d_model)(inp)
    pos_emb = tf.keras.layers.Embedding(max_len,    d_model)(tf.range(tf.shape(inp)[1]))
    x = tok_emb + pos_emb
    x = tf.keras.layers.Dropout(dropout)(x)
    for _ in range(N):
        x = EncoderBlock(d_model, num_heads, d_ff, dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return tf.keras.Model(inp, x)

enc = build_encoder(vocab_size=1000, max_len=128, d_model=64,
                    num_heads=4, d_ff=256, N=2)
enc.summary()`

const CODE_DECODER_BLOCK = `import tensorflow as tf

# ── Single Transformer Decoder Block ──
# Sub-layer 1: Masked Multi-Head SELF-Attention  + Residual + LayerNorm
# Sub-layer 2: Multi-Head CROSS-Attention        + Residual + LayerNorm
# Sub-layer 3: Position-wise FFN                 + Residual + LayerNorm

def make_causal_mask(seq_len):
    """Upper-triangular mask: token i cannot attend to j > i."""
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask * -1e9   # large negative → softmax ≈ 0

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        kd = d_model // num_heads
        self.self_attn  = tf.keras.layers.MultiHeadAttention(
                              num_heads=num_heads, key_dim=kd, dropout=dropout)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
                              num_heads=num_heads, key_dim=kd, dropout=dropout)
        self.ffn1  = tf.keras.layers.Dense(d_ff,     activation='relu')
        self.ffn2  = tf.keras.layers.Dense(d_model)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop  = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_out, causal_mask=None, training=False):
        T = tf.shape(x)[1]

        # 1. Masked self-attention (causal — no peeking at future tokens)
        sa = self.self_attn(x, x, x, attention_mask=causal_mask, training=training)
        x  = self.norm1(x + self.drop(sa, training=training))

        # 2. Cross-attention: Q from decoder, K/V from encoder
        ca = self.cross_attn(x, enc_out, enc_out, training=training)
        x  = self.norm2(x + self.drop(ca, training=training))

        # 3. FFN
        ff = self.ffn2(tf.nn.relu(self.ffn1(x)))
        x  = self.norm3(x + self.drop(ff, training=training))
        return x

# Quick shape check
B, T_src, T_tgt, D = 2, 8, 6, 64
block   = DecoderBlock(d_model=D, num_heads=4, d_ff=256)
enc_out = tf.random.normal((B, T_src, D))
dec_in  = tf.random.normal((B, T_tgt, D))
mask    = make_causal_mask(T_tgt)
out     = block(dec_in, enc_out, causal_mask=mask)
print("Decoder block output:", out.shape)  # (2, 6, 64)`

const CODE_FULL_TRANSFORMER = `import tensorflow as tf
import numpy as np

# ── Full Encoder-Decoder Transformer (mini) ──

def get_pos_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, None]
    dims      = np.arange(d_model)[None, :]
    rates     = 1 / (10000 ** (2 * (dims // 2) / d_model))
    angles    = positions * rates
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(angles[None], dtype=tf.float32)   # (1, max_len, d_model)

class MiniTransformer(tf.keras.Model):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, num_heads=4,
                 d_ff=256, N=2, max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = tf.keras.layers.Embedding(src_vocab, d_model)
        self.tgt_emb = tf.keras.layers.Embedding(tgt_vocab, d_model)
        self.pos_enc = get_pos_encoding(max_len, d_model)

        kd = d_model // num_heads
        self.enc_blocks = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=kd)
            for _ in range(N)
        ]
        self.dec_sa = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=kd)
            for _ in range(N)
        ]
        self.dec_ca = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=kd)
            for _ in range(N)
        ]
        self.final = tf.keras.layers.Dense(tgt_vocab)

    def call(self, src, tgt, training=False):
        T_src = tf.shape(src)[1]; T_tgt = tf.shape(tgt)[1]
        # Embed + positional
        x = self.src_emb(src) + self.pos_enc[:, :T_src, :]
        y = self.tgt_emb(tgt) + self.pos_enc[:, :T_tgt, :]
        # Encoder
        for enc in self.enc_blocks:
            x = x + enc(x, x, x, training=training)
        # Decoder
        causal = 1 - tf.linalg.band_part(tf.ones((T_tgt, T_tgt)), -1, 0)
        causal_mask = causal * -1e9
        for sa, ca in zip(self.dec_sa, self.dec_ca):
            y = y + sa(y, y, y, attention_mask=causal_mask, training=training)
            y = y + ca(y, x, x, training=training)
        return self.final(y)   # (batch, T_tgt, tgt_vocab)

model = MiniTransformer(src_vocab=200, tgt_vocab=150)
src = tf.constant([[5, 12, 8, 3, 0]])
tgt = tf.constant([[1, 7, 4, 2]])
out = model(src, tgt)
print("Transformer output:", out.shape)   # (1, 4, 150)`

const CODE_ENCODER_ONLY = `import tensorflow as tf

# ── Encoder-Only model (BERT-style) ──
# Task: sentence-pair classification (e.g. entailment)

def build_bert_style(vocab_size=30522, max_len=512,
                     d_model=768, num_heads=12, d_ff=3072, N=12):
    inp     = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    seg     = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    tok_emb = tf.keras.layers.Embedding(vocab_size, d_model)(inp)
    pos_emb = tf.keras.layers.Embedding(max_len,    d_model)(tf.range(max_len))
    seg_emb = tf.keras.layers.Embedding(2,           d_model)(seg)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-12)(tok_emb + pos_emb + seg_emb)
    x = tf.keras.layers.Dropout(0.1)(x)

    kd = d_model // num_heads
    for _ in range(N):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=kd)(x, x)
        x    = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff   = tf.keras.layers.Dense(d_ff, activation='gelu')(x)
        ff   = tf.keras.layers.Dense(d_model)(ff)
        x    = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

    # [CLS] token representation → classification head
    cls_out = x[:, 0, :]                            # (batch, d_model)
    logits  = tf.keras.layers.Dense(3)(cls_out)      # 3-class NLI
    return tf.keras.Model([inp, seg], logits)

# BERT-base parameter count ≈ 110M  (too large to instantiate fully here)
# Mini version for shape check:
mini = build_bert_style(vocab_size=1000, max_len=32, d_model=64, num_heads=4, d_ff=256, N=2)
mini.summary()`

const CODE_DECODER_ONLY = `import tensorflow as tf

# ── Decoder-Only model (GPT-style) ──
# Autoregressive language model: predicts next token given all previous tokens

def build_gpt_style(vocab_size=50257, max_len=1024,
                    d_model=768, num_heads=12, d_ff=3072, N=12):
    inp = tf.keras.Input(shape=(None,), dtype=tf.int32)
    T   = tf.shape(inp)[1]

    tok_emb = tf.keras.layers.Embedding(vocab_size, d_model)(inp)
    pos_emb = tf.keras.layers.Embedding(max_len,    d_model)(tf.range(T))
    x = tok_emb + pos_emb

    kd = d_model // num_heads
    for _ in range(N):
        # Causal (masked) self-attention only — NO cross-attention
        attn = tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=kd, use_causal_mask=True)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = tf.keras.layers.Dense(d_ff, activation='gelu')(x)
        ff = tf.keras.layers.Dense(d_model)(ff)
        x  = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

    # Project back to vocabulary
    logits = tf.keras.layers.Dense(vocab_size)(x)   # (batch, T, vocab_size)
    return tf.keras.Model(inp, logits)

mini_gpt = build_gpt_style(vocab_size=1000, max_len=64, d_model=64, num_heads=4, d_ff=256, N=2)
mini_gpt.summary()

# Generation loop (greedy)
def generate(model, prompt_ids, max_new=10):
    ids = list(prompt_ids)
    for _ in range(max_new):
        inp  = tf.constant([ids])
        logi = model(inp)[0, -1]          # logits for last position
        next_id = int(tf.argmax(logi))
        ids.append(next_id)
        if next_id == 2:                  # EOS
            break
    return ids

print("Generated:", generate(mini_gpt, [5, 12, 8], max_new=5))`

const CODE_RESIDUAL_STREAM = `import tensorflow as tf
import numpy as np

# ── Residual stream view of the Transformer ──
# Every sub-layer *adds* to a shared residual stream x.
# x starts as the token+positional embedding and accumulates information.

# Visualise how the residual stream norm grows across layers

np.random.seed(0)
d_model, T, N_layers = 64, 10, 6

x = tf.random.normal((1, T, d_model))   # initial stream: embeddings
norms = [float(tf.reduce_mean(tf.norm(x, axis=-1)))]

for layer in range(N_layers):
    # Simulate attention residual (small random update)
    delta_attn = tf.random.normal((1, T, d_model)) * 0.3
    x = x + delta_attn                   # residual add
    # Simulate FFN residual
    delta_ffn  = tf.random.normal((1, T, d_model)) * 0.3
    x = x + delta_ffn                    # residual add
    norms.append(float(tf.reduce_mean(tf.norm(x, axis=-1))))

print("Residual stream L2 norm per layer:")
for i, n in enumerate(norms):
    bar = '█' * int(n * 2)
    print(f"  Layer {i}: {n:.2f}  {bar}")

# Key insight: because every sub-layer is ADDITIVE, gradients flow directly
# from the loss back to the embedding layer without vanishing.
# ∂L/∂x₀ = ∂L/∂x_N (plus contributions from each sublayer)
print("\\nGradient highway: ∂L/∂x₀ always has a direct path via residuals")`

// ── Quiz ──────────────────────────────────────────────────────────────────────

const QUIZ_QUESTIONS = [
  {
    question: 'What is the purpose of the position-wise FFN in a Transformer block?',
    options: [
      'To compute attention weights between tokens',
      'To add non-linearity and increase model capacity — applied independently at each position',
      'To normalise the activations across the batch',
      'To encode absolute position information',
    ],
    correct: 1,
    explanation: 'Multi-head attention mixes information across positions but is linear within each position. The FFN (with ReLU) adds non-linearity and expands/compresses the representation at each position independently.',
  },
  {
    question: 'Why is Layer Normalisation preferred over Batch Normalisation in Transformers?',
    options: [
      'Layer Norm is faster to compute',
      'Batch Norm requires a fixed batch size',
      'Layer Norm normalises over feature dimensions independently per token, handling variable-length sequences correctly',
      'Batch Norm cannot be used with residual connections',
    ],
    correct: 2,
    explanation: 'Batch Norm computes statistics across the batch dimension, which becomes problematic with variable-length padded sequences and small batch sizes. Layer Norm operates per-token across the feature dimension, independent of batch size and sequence length.',
  },
  {
    question: 'What does the causal (look-ahead) mask do in the Transformer decoder\'s self-attention?',
    options: [
      'Masks padding tokens in the source sequence',
      'Prevents each decoder position from attending to future positions it should not yet have seen',
      'Scales the attention scores by √d_k',
      'Prevents attention heads from attending to the same positions',
    ],
    correct: 1,
    explanation: 'During training, the full target sequence is fed to the decoder at once (teacher forcing). The causal mask sets future positions to −∞ before softmax so each position can only attend to itself and earlier tokens.',
  },
  {
    question: 'In an Encoder-Only Transformer (BERT-style), the [CLS] token is used to:',
    options: [
      'Indicate the start of decoding',
      'Separate the two sentences in a pair',
      'Aggregate a sentence-level representation for classification tasks',
      'Encode absolute position 0',
    ],
    correct: 2,
    explanation: '[CLS] (class token) is prepended to the input. After N encoder layers it has attended to the entire sequence, so its final representation is used as the sentence-level vector for classification.',
  },
  {
    question: 'A Decoder-Only Transformer (GPT-style) differs from an Encoder-Decoder by:',
    options: [
      'Using additive attention instead of dot-product attention',
      'Having no cross-attention sublayer — only causal self-attention',
      'Processing the source and target in separate forward passes',
      'Using a separate positional encoding for each layer',
    ],
    correct: 1,
    explanation: 'GPT-style models have only masked self-attention (no cross-attention, no encoder). The model attends to its own previous tokens to predict the next one — making it purely autoregressive.',
  },
  {
    question: 'In the residual stream view, what does every sublayer (attention + FFN) do to the shared stream?',
    options: [
      'Replaces the stream with its output',
      'Concatenates its output to the stream',
      'Adds its output to the stream — leaving the original information intact',
      'Multiplies the stream by a learned gate',
    ],
    correct: 2,
    explanation: 'Each sublayer computes a delta and adds it: x ← x + sublayer(x). This means the original information is never destroyed — it persists through all layers, and gradients flow back to the embedding directly through the skip connections.',
  },
]

// ── Transformer architecture overview SVG ────────────────────────────────────

function TransformerOverview({ theme }) {
  const dark = theme === 'dark'

  const box = (x, y, w, h, fill, stroke, label, sub, fg) => (
    <g key={label}>
      <rect x={x} y={y} width={w} height={h} rx="8"
        fill={fill} stroke={stroke} strokeWidth="1.5" />
      <text x={x + w / 2} y={y + h / 2 - (sub ? 6 : 0)}
        textAnchor="middle" fontSize="10" fontWeight="700" fill={fg}>{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 10}
        textAnchor="middle" fontSize="9" fill={fg} opacity="0.75">{sub}</text>}
    </g>
  )

  const arrow = (x1, y1, x2, y2, color) => (
    <line x1={x1} y1={y1} x2={x2} y2={y2}
      stroke={color || (dark ? '#475569' : '#94a3b8')}
      strokeWidth="1.3" markerEnd="url(#trarr)" />
  )

  const indB = dark ? '#312e81' : '#eef2ff'
  const indS = dark ? '#6366f1' : '#6366f1'
  const indF = dark ? '#c7d2fe' : '#4338ca'
  const vioB = dark ? '#3b0764' : '#fdf4ff'
  const vioS = dark ? '#a855f7' : '#a855f7'
  const vioF = dark ? '#e9d5ff' : '#7e22ce'
  const cynB = dark ? '#083344' : '#ecfeff'
  const cynS = dark ? '#22d3ee' : '#0891b2'
  const cynF = dark ? '#67e8f9' : '#0e7490'
  const grnB = dark ? '#052e16' : '#f0fdf4'
  const grnS = dark ? '#22c55e' : '#16a34a'
  const grnF = dark ? '#86efac' : '#166534'
  const slB  = dark ? '#1e293b' : '#f8fafc'
  const slS  = dark ? '#475569' : '#cbd5e1'
  const slF  = dark ? '#94a3b8' : '#64748b'

  return (
    <div className="overflow-x-auto my-4">
      <svg viewBox="0 0 560 310" className="w-full max-w-2xl mx-auto block" style={{ minWidth: 360 }}>

        {/* ── Encoder column ── */}
        <text x="130" y="16" textAnchor="middle" fontSize="11" fontWeight="800"
          fill={indF}>ENCODER</text>

        {box(50,  25, 160, 32, slB, slS, 'Input Embedding', '+ Pos Encoding', slF)}
        {arrow(130, 57, 130, 72)}
        {box(50,  72, 160, 32, indB, indS, 'Multi-Head Self-Attention', null, indF)}
        {arrow(130, 104, 130, 116)}
        {box(50, 116, 160, 22, cynB, cynS, 'Add & Layer Norm', null, cynF)}
        {arrow(130, 138, 130, 150)}
        {box(50, 150, 160, 32, vioB, vioS, 'Position-wise FFN', null, vioF)}
        {arrow(130, 182, 130, 194)}
        {box(50, 194, 160, 22, cynB, cynS, 'Add & Layer Norm', null, cynF)}

        {/* Nx label */}
        <rect x="220" y="68" width="28" height="160" rx="6"
          fill="none" stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" strokeDasharray="4 2" />
        <text x="234" y="153" textAnchor="middle" fontSize="10" fontWeight="700"
          fill={dark ? '#475569' : '#94a3b8'} transform="rotate(-90 234 153)">× N</text>

        {/* ── Arrow encoder → decoder ── */}
        {arrow(210, 205, 330, 175)}
        {arrow(210, 205, 330, 205)}

        {/* ── Decoder column ── */}
        <text x="430" y="16" textAnchor="middle" fontSize="11" fontWeight="800"
          fill={vioF}>DECODER</text>

        {box(330, 25,  160, 32, slB, slS, 'Target Embedding', '+ Pos Encoding', slF)}
        {arrow(410, 57, 410, 72)}
        {box(330, 72,  160, 32, indB, indS, 'Masked Self-Attention', '(causal mask)', indF)}
        {arrow(410, 104, 410, 116)}
        {box(330, 116, 160, 22, cynB, cynS, 'Add & Layer Norm', null, cynF)}
        {arrow(410, 138, 410, 150)}
        {box(330, 150, 160, 32, grnB, grnS, 'Cross-Attention', 'Q←dec, K/V←enc', grnF)}
        {arrow(410, 182, 410, 194)}
        {box(330, 194, 160, 22, cynB, cynS, 'Add & Layer Norm', null, cynF)}
        {arrow(410, 216, 410, 228)}
        {box(330, 228, 160, 32, vioB, vioS, 'Position-wise FFN', null, vioF)}
        {arrow(410, 260, 410, 272)}
        {box(330, 272, 160, 22, cynB, cynS, 'Add & Layer Norm', null, cynF)}

        {/* Nx label decoder */}
        <rect x="500" y="68" width="28" height="232" rx="6"
          fill="none" stroke={dark ? '#334155' : '#e2e8f0'} strokeWidth="1" strokeDasharray="4 2" />
        <text x="514" y="187" textAnchor="middle" fontSize="10" fontWeight="700"
          fill={dark ? '#475569' : '#94a3b8'} transform="rotate(-90 514 187)">× N</text>

        {/* output below */}
        {arrow(410, 294, 410, 306)}
        <text x="410" y="310" textAnchor="middle" fontSize="10" fontWeight="700"
          fill={grnF}>Linear + Softmax → output</text>

        <defs>
          <marker id="trarr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill={dark ? '#475569' : '#94a3b8'} />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

// ── Residual stream diagram ───────────────────────────────────────────────────

function ResidualStreamDiagram({ theme }) {
  const dark = theme === 'dark'
  const layers = ['Embed', 'Attn 1', 'FFN 1', 'Attn 2', 'FFN 2', 'Output']
  const W = 440, boxW = 56, boxH = 34, gap = (W - 40 - boxW * layers.length) / (layers.length - 1)

  const colors = [
    [dark ? '#312e81' : '#eef2ff', dark ? '#6366f1' : '#4338ca'],
    [dark ? '#3b0764' : '#fdf4ff', dark ? '#a855f7' : '#7e22ce'],
    [dark ? '#083344' : '#ecfeff', dark ? '#22d3ee' : '#0e7490'],
    [dark ? '#3b0764' : '#fdf4ff', dark ? '#a855f7' : '#7e22ce'],
    [dark ? '#083344' : '#ecfeff', dark ? '#22d3ee' : '#0e7490'],
    [dark ? '#052e16' : '#f0fdf4', dark ? '#22c55e' : '#166534'],
  ]

  return (
    <div className="overflow-x-auto my-4">
      <svg viewBox={`0 0 ${W} 90`} className="w-full max-w-xl mx-auto block">
        {/* Main stream line */}
        <line x1="20" y1="52" x2={W - 20} y2="52"
          stroke={dark ? '#6366f1' : '#4f46e5'} strokeWidth="2.5" strokeDasharray="none" />

        {layers.map((lbl, i) => {
          const cx = 20 + i * (boxW + gap)
          const [bg, fg] = colors[i]
          return (
            <g key={i}>
              {/* Skip arrow for attn/ffn layers */}
              {i > 0 && i < layers.length - 1 && (
                <>
                  <path d={`M ${cx - gap * 0.4} 35 Q ${cx + boxW / 2} 10 ${cx + boxW + gap * 0.4} 35`}
                    fill="none" stroke={dark ? '#64748b' : '#94a3b8'}
                    strokeWidth="1" strokeDasharray="3 2" markerEnd="url(#skiparr)" />
                  <text x={cx + boxW / 2} y="9" textAnchor="middle" fontSize="8"
                    fill={dark ? '#64748b' : '#94a3b8'}>+</text>
                </>
              )}
              {/* Box */}
              <rect x={cx} y="35" width={boxW} height={boxH} rx="6"
                fill={bg} stroke={fg} strokeWidth="1.5" />
              <text x={cx + boxW / 2} y={35 + boxH / 2 + 4}
                textAnchor="middle" fontSize="9" fontWeight="700" fill={fg}>{lbl}</text>
            </g>
          )
        })}

        {/* stream label */}
        <text x="20" y="75" fontSize="9" fill={dark ? '#6366f1' : '#4f46e5'} fontWeight="700">
          ← residual stream (shared vector updated by each sublayer) →
        </text>

        <defs>
          <marker id="skiparr" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
            <path d="M0,0 L0,5 L5,2.5 z" fill={dark ? '#64748b' : '#94a3b8'} />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

// ── Variant cards ─────────────────────────────────────────────────────────────

function VariantCard({ icon, name, example, architecture, usecases, dark }) {
  return (
    <div className={`rounded-xl border p-4 ${dark ? 'bg-slate-900/60 border-slate-700' : 'bg-white border-slate-200'}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">{icon}</span>
        <div>
          <p className={`font-bold text-sm ${dark ? 'text-white' : 'text-slate-900'}`}>{name}</p>
          <p className={`text-xs font-mono ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>{example}</p>
        </div>
      </div>
      <p className={`text-xs mb-2 ${dark ? 'text-slate-400' : 'text-slate-500'}`}>{architecture}</p>
      <ul className="space-y-1">
        {usecases.map((u, i) => (
          <li key={i} className={`text-xs flex gap-1.5 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
            <span className={dark ? 'text-violet-400' : 'text-violet-500'}>▸</span>{u}
          </li>
        ))}
      </ul>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function Transformer() {
  const { theme, markTopicComplete } = useApp()
  const dark = theme === 'dark'

  return (
    <div className="space-y-10">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-4xl">⚡</span>
          <div>
            <p className={`text-sm font-semibold uppercase tracking-widest ${dark ? 'text-violet-400' : 'text-violet-600'}`}>
              Session 12 · Deep Learning
            </p>
            <h1 className={`text-3xl font-black ${dark ? 'text-white' : 'text-slate-900'}`}>
              Transformer Architecture
            </h1>
          </div>
        </div>
        <p className={`text-lg leading-relaxed ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The Transformer (Vaswani et al., 2017) replaced recurrence entirely with
          <strong> self-attention</strong>. Its fully parallelisable architecture became the foundation
          of every modern large language model — BERT, GPT, T5, and beyond.
        </p>

        {/* Prerequisites */}
        <div className={`mt-4 rounded-xl p-4 border text-sm ${dark ? 'bg-amber-900/20 border-amber-700/40 text-amber-200' : 'bg-amber-50 border-amber-200 text-amber-800'}`}>
          <p className="font-bold mb-1">📚 Prerequisites covered in earlier sessions</p>
          <ul className="list-disc list-inside space-y-0.5 text-xs">
            <li><strong>Session 9</strong> — Encoder–decoder, teacher forcing, greedy decoding, masked loss</li>
            <li><strong>Session 11</strong> — Scaled dot-product attention, multi-head attention, Q/K/V, positional encoding, cross-attention, self-attention</li>
          </ul>
        </div>
      </motion.div>

      {/* ── Section 12.1 ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 12.1</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Core Transformer Components</h2>
        </div>

        {/* 12.1.1 Architecture overview */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          12.1.1 — Transformer Architecture Overview
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The original Transformer is an <strong>encoder–decoder</strong> model. Both sides are stacks
          of N identical blocks. The encoder reads the source sequence in parallel; the decoder generates
          the target sequence one token at a time, attending to both its own previous outputs and the
          encoder's output.
        </p>

        <TransformerOverview theme={theme} />

        <TheoryBlock items={[
          { title: 'No recurrence', content: 'The entire source sequence is processed in a single parallel forward pass through the encoder. Sequence order is injected via positional encoding, not via sequential computation.' },
          { title: 'N stacked blocks', content: 'Original paper used N=6 for both encoder and decoder. Each block is identical in structure; weights are NOT shared across blocks.' },
          { title: 'Residual + LayerNorm', content: 'Every sublayer (attention, FFN) wraps x ← LayerNorm(x + sublayer(x)). This enables deep stacks by keeping gradients healthy through all N layers.' },
          { title: 'Linear + Softmax output', content: 'The decoder\'s final hidden state is projected to vocabulary size and softmaxed to produce a probability distribution over the next token.' },
        ]} />

        {/* 12.1.2 FFN */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          12.1.2 — Position-wise Feed-Forward Networks (FFN)
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          After multi-head attention mixes information <em>across</em> positions, the FFN processes each
          position <em>independently</em> through a two-layer MLP. It is the same MLP for every
          position — hence "position-wise". In practice it is the most parameter-heavy component
          (d_ff = 4 × d_model in the original paper).
        </p>

        <Callout type="formula" title="Position-wise FFN">
          FFN(x) = max(0, x W₁ + b₁) W₂ + b₂ &nbsp;&nbsp; where W₁ ∈ R^{'d_model × d_ff'}, W₂ ∈ R^{'d_ff × d_model'}
        </Callout>

        <CodeBlock code={CODE_FFN} language="python" title="positionwise_ffn.py" />

        <DeepDive title="Why d_ff = 4 × d_model?">
          <TheoryBlock items={[
            { title: '', content: 'The expansion ratio ×4 is an empirical sweet spot. The FFN acts like a key-value memory: neurons that "fire" on specific input patterns retrieve stored knowledge. Larger d_ff = more stored patterns. GPT-3 uses d_model=12288, d_ff=49152 (×4).' },
            { title: 'Activation variants', content: 'Original used ReLU. BERT uses GELU (smoother, slightly better). GPT-2 also uses GELU. Modern LLMs use SwiGLU (two gates) which requires d_ff = (8/3) × d_model instead of ×4.' },
          ]} />
        </DeepDive>

        {/* 12.1.3 Residual + LayerNorm */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          12.1.3 — Residual Connections and Layer Normalisation
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Every sublayer is wrapped in a <strong>residual connection</strong> followed by
          <strong> Layer Normalisation</strong>. Together they solve the two main training problems
          of deep networks: vanishing gradients (residuals) and internal covariate shift (LayerNorm).
        </p>

        <TheoryBlock cols={2} items={[
          {
            title: '➕ Residual connection',
            content: 'x ← x + sublayer(x). The identity path means ∂L/∂x_in = ∂L/∂x_out + gradient-from-sublayer. Gradient always has a direct highway back — no vanishing.',
          },
          {
            title: '📏 Layer Normalisation',
            content: 'Normalises each token\'s d_model-dimensional vector to mean 0, variance 1 (then rescales with learned γ, β). Independent of batch size and sequence length.',
          },
          {
            title: 'Pre-norm vs Post-norm',
            content: 'Original paper used Post-LN: LayerNorm(x + sublayer(x)). Modern models (GPT-2 onward) use Pre-LN: x + sublayer(LayerNorm(x)), which is more training-stable for deep models.',
          },
          {
            title: 'Dropout',
            content: 'Applied to sublayer output before addition: x ← x + Dropout(sublayer(x)). Typical rate 0.1 for base models, 0.0 for very large models.',
          },
        ]} />

        <CodeBlock code={CODE_LAYERNORM} language="python" title="residual_layernorm.py" />

        {/* 12.1.4 Encoder block */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          12.1.4 — Transformer Encoder Block
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Each encoder block has exactly <strong>two sublayers</strong>: multi-head self-attention and
          the position-wise FFN. N identical blocks are stacked; the output of block l is the input
          to block l+1. The full encoder maps a token sequence to a same-shape sequence of
          contextualised representations.
        </p>

        <Callout type="info" title="Encoder block data flow">
          x → MHA(x,x,x) → [Add &amp; Norm] → FFN → [Add &amp; Norm] → x_out
        </Callout>

        <CodeBlock code={CODE_ENCODER_BLOCK} language="python" title="encoder_block.py" />

        {/* 12.1.5 Decoder block */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          12.1.5 — Transformer Decoder Block (with Masked Self-Attention)
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The decoder block has <strong>three sublayers</strong>. The first is a masked self-attention
          that prevents each position from seeing future tokens (essential for autoregressive
          generation). The second is cross-attention connecting the decoder to the encoder output.
          The third is the position-wise FFN.
        </p>

        <TheoryBlock items={[
          { title: 'Sublayer 1 — Masked self-attention', content: 'Q=K=V=decoder hidden states. Upper-triangular mask sets future positions to −∞. Allows the decoder to condition on its own previous outputs.' },
          { title: 'Sublayer 2 — Cross-attention', content: 'Q=decoder hidden, K=V=encoder output. Same as Bahdanau attention but now implemented via scaled dot-product. Connects target to source.' },
          { title: 'Sublayer 3 — FFN', content: 'Identical to the encoder FFN. Applied position-wise after cross-attention to transform the attended representation.' },
          { title: 'Why three instead of two?', content: 'The extra sublayer (cross-attention) is the join point between encoder and decoder. Encoder-only / decoder-only models have only two sublayers per block.' },
        ]} />

        <Callout type="info" title="Decoder block data flow">
          y → MaskedMHA(y,y,y) → [Add &amp; Norm] → CrossMHA(y,enc,enc) → [Add &amp; Norm] → FFN → [Add &amp; Norm] → y_out
        </Callout>

        <CodeBlock code={CODE_DECODER_BLOCK} language="python" title="decoder_block.py" />

        <div className="mt-4">
          <CodeBlock code={CODE_FULL_TRANSFORMER} language="python" title="full_transformer.py" />
        </div>
      </motion.div>

      {/* ── Section 12.2 ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-violet-500/5 border-violet-500/20' : 'bg-violet-50 border-violet-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-violet-400' : 'text-violet-600'}`}>Section 12.2</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Transformer Variants and Residual View</h2>
        </div>

        {/* 12.2.1–12.2.3 Variants */}
        <h3 className={`text-lg font-bold mb-4 ${dark ? 'text-white' : 'text-slate-800'}`}>
          12.2.1–12.2.3 — Encoder-Only, Encoder–Decoder, Decoder-Only
        </h3>
        <p className={`mb-5 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Modern pre-trained models keep the Transformer block but drop one half of the architecture
          depending on the task family. Each variant specialises to a different class of problems.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <VariantCard dark={dark}
            icon="🔎" name="Encoder-Only" example="BERT, RoBERTa, DeBERTa"
            architecture="Stack of N encoder blocks only. Full bidirectional attention over the entire input."
            usecases={[
              'Text classification (sentiment, NLI)',
              'Named entity recognition (NER)',
              'Question answering (extractive)',
              'Sentence embeddings / semantic search',
            ]}
          />
          <VariantCard dark={dark}
            icon="🔄" name="Encoder–Decoder" example="T5, BART, mT5, MarianMT"
            architecture="Full encoder + decoder. Cross-attention connects them. Handles variable source/target lengths."
            usecases={[
              'Machine translation',
              'Abstractive summarisation',
              'Question generation',
              'Any seq2seq task',
            ]}
          />
          <VariantCard dark={dark}
            icon="✍️" name="Decoder-Only" example="GPT-2/3/4, LLaMA, Falcon, Mistral"
            architecture="Causal self-attention only — no encoder, no cross-attention. Autoregressive left-to-right generation."
            usecases={[
              'Text generation & completion',
              'In-context few-shot learning',
              'Code generation (Codex)',
              'Chat / instruction following',
            ]}
          />
        </div>

        <CodeBlock code={CODE_ENCODER_ONLY} language="python" title="encoder_only_bert.py" />
        <div className="mt-4">
          <CodeBlock code={CODE_DECODER_ONLY} language="python" title="decoder_only_gpt.py" />
        </div>

        <DeepDive title="Pre-training objectives">
          <TheoryBlock items={[
            { title: 'Encoder-Only: MLM', content: 'BERT uses Masked Language Modelling — randomly mask 15% of tokens and predict them. Bidirectional context makes representations rich for classification.' },
            { title: 'Encoder-Decoder: span denoising', content: 'T5 uses "text-to-text": every task (translation, summarisation, classification) is framed as a string-in → string-out problem. Pre-trains with span corruption (mask contiguous spans, not single tokens).' },
            { title: 'Decoder-Only: CLM', content: 'GPT uses Causal Language Modelling — predict the next token given all previous. Scales extremely well: larger model + more data = better few-shot performance (emergent abilities).' },
          ]} />
        </DeepDive>

        {/* 12.2.4 Residual view */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          12.2.4 — Residual View of the Transformer (Residual Stream)
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          A powerful conceptual lens: think of the Transformer as a single
          <strong> residual stream</strong> — a d_model-dimensional vector per token that is
          <em> initialised</em> as the token + positional embedding, then <em>updated additively</em>
          by every attention and FFN sublayer. No sublayer ever replaces the stream — it only adds to it.
        </p>

        <ResidualStreamDiagram theme={theme} />

        <TheoryBlock items={[
          { title: 'Additive updates only', content: 'Every sublayer computes a delta Δ and does x ← x + Δ. The original token embedding is always present in the stream and can be read at any later layer.' },
          { title: 'Gradient highway', content: '∂L/∂x_0 = ∂L/∂x_N × (chain) + direct identity path. The direct path prevents vanishing gradients regardless of depth N.' },
          { title: 'Layer superposition', content: 'Different features can coexist in the same residual stream at different layers. Early layers store token-level features; later layers store more abstract, contextual features.' },
          { title: 'Mechanistic interpretability', content: 'Research (Elhage et al., 2021 "A Mathematical Framework for Transformer Circuits") uses the residual stream lens to reverse-engineer what specific attention heads compute.' },
        ]} />

        <CodeBlock code={CODE_RESIDUAL_STREAM} language="python" title="residual_stream.py" />

        <Callout type="analogy" title="Residual stream analogy">
          Think of each token's residual stream as a shared whiteboard. The embedding layer writes the
          initial word meaning. Each attention head reads from the board, computes something, and writes
          an additive note. Each FFN reads and writes another note. The final Linear layer reads the
          completed board and predicts the next token.
        </Callout>
      </motion.div>

      {/* ── Quiz ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-cyan-500/5 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-cyan-400' : 'text-cyan-600'}`}>Knowledge Check</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Quiz — Sessions 12.1 &amp; 12.2</h2>
        </div>
        <Quiz
          questions={QUIZ_QUESTIONS}
          onComplete={() => markTopicComplete('transformer')}
        />
      </motion.div>
    </div>
  )
}
