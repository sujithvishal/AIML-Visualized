import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ── Code blocks ────────────────────────────────────────────────────────────────

const CODE_TOKENISE = `import tensorflow as tf
import numpy as np

# Raw text corpus
sentences = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great pets",
]

# 1. Build a vocabulary via the TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=20,          # vocabulary size (including [PAD]=0, [UNK]=1)
    output_mode='int',      # return integer indices
    output_sequence_length=8,  # pad / truncate to this length
)
vectorizer.adapt(sentences)

vocab = vectorizer.get_vocabulary()
print("Vocabulary:", vocab)
# ['', '[UNK]', 'the', 'sat', 'on', 'cats', 'cat', ...]

# 2. Encode a sentence → integer sequence
seq = vectorizer(["the cat sat on the mat"])
print("Encoded:", seq.numpy())  # e.g. [[2, 6, 3, 4, 2, 5, 0, 0]]

# 3. Reverse-lookup: index → word
idx2word = np.array(vocab)
print("Decoded:", ' '.join(idx2word[seq.numpy()[0]]))`

const CODE_FORWARD = `import tensorflow as tf
import numpy as np

# ── Single-step RNN forward pass (manual, for illustration) ──

np.random.seed(0)
H, D = 4, 3            # hidden_size=4, input_dim=3
T    = 5               # sequence length

# Learnable weight matrices
Wh = np.random.randn(H, H) * 0.1   # hidden-to-hidden
Wx = np.random.randn(H, D) * 0.1   # input-to-hidden
bh = np.zeros(H)                   # bias

def tanh(x):
    return np.tanh(x)

# Simulate one forward pass over a sequence (batch=1)
x_seq = np.random.randn(T, D)       # (T, D)
h     = np.zeros(H)                 # initial hidden state

hidden_states = []
for t in range(T):
    h = tanh(Wx @ x_seq[t] + Wh @ h + bh)
    hidden_states.append(h.copy())
    print(f"t={t}  h={np.round(h, 3)}")

# Using Keras SimpleRNN instead
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(4, return_sequences=True,
                              input_shape=(T, D))
])
out = model(x_seq[None].astype('float32'))   # add batch dim
print("Keras output shape:", out.shape)       # (1, T, H)`

const CODE_BPTT = `import tensorflow as tf
import numpy as np

# ── BPTT via tf.GradientTape (language-model toy example) ──
# Task: predict next token in a sequence of 10 vocabulary items

VOCAB = 10; SEQ_LEN = 6; HIDDEN = 16; BATCH = 4

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB, 8, input_length=SEQ_LEN),
    tf.keras.layers.SimpleRNN(HIDDEN),
    tf.keras.layers.Dense(VOCAB),          # logits over vocab
])

optimizer = tf.keras.optimizers.Adam(1e-2)
loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Random toy batch: inputs (B,T), targets (B,) — predict last token
x_batch = tf.random.uniform((BATCH, SEQ_LEN), 0, VOCAB, dtype=tf.int32)
y_batch = tf.random.uniform((BATCH,),          0, VOCAB, dtype=tf.int32)

for step in range(5):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)   # (B, VOCAB)
        loss   = loss_fn(y_batch, logits)

    grads = tape.gradient(loss, model.trainable_weights)
    # grads flow back through the unrolled RNN — this IS BPTT
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print(f"Step {step+1}  Loss: {loss.numpy():.4f}")`

const CODE_LM = `import tensorflow as tf
import numpy as np

# ── Character-level language model with RNN ──

text = "hello world hello rnn world"
chars = sorted(set(text))
c2i   = {c: i for i, c in enumerate(chars)}
i2c   = np.array(chars)

SEQ_LEN = 5; VOCAB = len(chars); HIDDEN = 32

# Build (input, target) pairs: predict next char
encoded = [c2i[c] for c in text]
X = [encoded[i:i+SEQ_LEN]     for i in range(len(encoded)-SEQ_LEN)]
Y = [encoded[i+SEQ_LEN]        for i in range(len(encoded)-SEQ_LEN)]
X = tf.constant(X); Y = tf.constant(Y)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB, 8,      input_length=SEQ_LEN),
    tf.keras.layers.SimpleRNN(HIDDEN),
    tf.keras.layers.Dense(VOCAB, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, Y, epochs=30, verbose=0)

# ── Generation: greedy decoding ──
seed = "hello"
for _ in range(10):
    seq = [c2i[c] for c in seed[-SEQ_LEN:]]
    inp = tf.constant([seq])
    probs = model(inp)[0].numpy()
    next_char = i2c[np.argmax(probs)]
    seed += next_char
print("Generated:", seed)`

const CODE_ENCDEC = `import tensorflow as tf

# ── Encoder–Decoder (seq2seq) for translation toy task ──
# Vocabulary sizes (source / target languages)
SRC_VOCAB = 50; TGT_VOCAB = 40
EMBED = 16; HIDDEN = 32; MAX_LEN = 10

# ── Encoder ──
enc_input = tf.keras.Input(shape=(None,), name='encoder_input')
enc_emb   = tf.keras.layers.Embedding(SRC_VOCAB, EMBED)(enc_input)
# return_state=True gives us the final hidden state
enc_out, enc_h = tf.keras.layers.SimpleRNN(
    HIDDEN, return_state=True, name='encoder')(enc_emb)
encoder_states = enc_h  # single state for SimpleRNN

# ── Decoder (training mode) ──
dec_input  = tf.keras.Input(shape=(None,), name='decoder_input')
dec_emb    = tf.keras.layers.Embedding(TGT_VOCAB, EMBED)(dec_input)
dec_rnn    = tf.keras.layers.SimpleRNN(HIDDEN, return_sequences=True,
                                        return_state=True, name='decoder')
# Pass encoder final state as decoder initial state
dec_out, _ = dec_rnn(dec_emb, initial_state=encoder_states)
dec_dense  = tf.keras.layers.Dense(TGT_VOCAB, activation='softmax')(dec_out)

model = tf.keras.Model([enc_input, dec_input], dec_dense)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()`

const CODE_TEACHER_FORCE = `import tensorflow as tf
import numpy as np

# ── Teacher forcing vs free-running illustration ──
# During TRAINING: decoder input  = ground-truth shifted target  (<BOS> + y[:-1])
# During INFERENCE: decoder input = model's own previous prediction

SRC_VOCAB = 30; TGT_VOCAB = 25; EMBED = 12; HIDDEN = 24; MAX_OUT = 8

# Build encoder
enc_in  = tf.keras.Input(shape=(None,))
enc_emb = tf.keras.layers.Embedding(SRC_VOCAB, EMBED)(enc_in)
_, enc_state = tf.keras.layers.SimpleRNN(HIDDEN, return_state=True)(enc_emb)

# Build decoder
dec_in  = tf.keras.Input(shape=(None,))
dec_emb = tf.keras.layers.Embedding(TGT_VOCAB, EMBED)(dec_in)
dec_rnn = tf.keras.layers.SimpleRNN(HIDDEN, return_sequences=True, return_state=True)
dec_out_seq, _ = dec_rnn(dec_emb, initial_state=enc_state)
logits  = tf.keras.layers.Dense(TGT_VOCAB)(dec_out_seq)

train_model = tf.keras.Model([enc_in, dec_in], logits)

# Teacher forcing: feed correct target tokens shifted by 1
BOS, EOS, PAD = 1, 2, 0
batch = 4; src_len = 6; tgt_len = 7
src = np.random.randint(3, SRC_VOCAB, (batch, src_len))
tgt = np.random.randint(3, TGT_VOCAB, (batch, tgt_len))

dec_input_tf  = np.hstack([np.full((batch,1), BOS), tgt[:,:-1]])  # BOS + y[:-1]
dec_target_tf = tgt                                                  # y

train_model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
train_model.fit([src, dec_input_tf], dec_target_tf, epochs=2, verbose=1)`

const CODE_MASKED_LOSS = `import tensorflow as tf
import numpy as np

# ── Masked cross-entropy loss (ignore PAD=0 tokens) ──

PAD = 0

def masked_sparse_ce(y_true, y_pred):
    """Compute mean CE loss over non-padding positions only."""
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True)             # (B, T)
    mask = tf.cast(tf.not_equal(y_true, PAD), tf.float32)  # 1 where real, 0 at PAD
    loss *= mask                                       # zero out PAD positions
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)  # mean over real tokens

# Example: batch of 3 target sequences, length 5, vocab=10
BATCH, T, VOCAB = 3, 5, 10
y_true = tf.constant([[3, 4, 2, 0, 0],    # last 2 are PAD
                       [5, 1, 3, 6, 0],    # last 1 is PAD
                       [2, 7, 1, 4, 9]])   # no PAD
y_pred = tf.random.normal((BATCH, T, VOCAB))

loss_masked   = masked_sparse_ce(y_true, y_pred)
loss_unmasked = tf.keras.losses.sparse_categorical_crossentropy(
    y_true, y_pred, from_logits=True)
print(f"Masked loss  : {loss_masked.numpy():.4f}")
print(f"Unmasked mean: {tf.reduce_mean(loss_unmasked).numpy():.4f}")`

const CODE_GREEDY = `import tensorflow as tf
import numpy as np

# ── Greedy decoding (inference) for a trained encoder–decoder ──

def greedy_decode(encoder_model, decoder_model, src_seq,
                  bos_idx, eos_idx, max_len=20):
    """
    encoder_model: src → enc_state
    decoder_model: (token, state) → (probs, new_state)
    """
    # 1. Encode source
    state = encoder_model(src_seq)               # final hidden state

    # 2. Seed decoder with BOS token
    dec_tok   = np.array([[bos_idx]])            # (1, 1)
    output_ids = []

    for _ in range(max_len):
        # 3. One decoder step
        probs, state = decoder_model([dec_tok, state])
        # 4. Pick the highest-probability token
        next_id = int(np.argmax(probs[0, -1]))
        if next_id == eos_idx:
            break
        output_ids.append(next_id)
        dec_tok = np.array([[next_id]])          # feed prediction as next input

    return output_ids

# Minimal end-to-end check (random weights)
SRC_V, TGT_V, EMB, H = 30, 25, 12, 24
BOS, EOS = 1, 2

src_in = tf.keras.Input(shape=(None,))
s_emb  = tf.keras.layers.Embedding(SRC_V, EMB)(src_in)
_, s_h = tf.keras.layers.SimpleRNN(H, return_state=True)(s_emb)
enc_m  = tf.keras.Model(src_in, s_h)

dec_tok_in = tf.keras.Input(shape=(None,))
dec_st_in  = tf.keras.Input(shape=(H,))
d_emb      = tf.keras.layers.Embedding(TGT_V, EMB)(dec_tok_in)
d_rnn      = tf.keras.layers.SimpleRNN(H, return_sequences=True, return_state=True)
d_out, d_h = d_rnn(d_emb, initial_state=dec_st_in)
d_probs    = tf.keras.layers.Dense(TGT_V, activation='softmax')(d_out)
dec_m      = tf.keras.Model([dec_tok_in, dec_st_in], [d_probs, d_h])

result = greedy_decode(enc_m, dec_m, np.array([[5, 12, 8, 0]]), BOS, EOS)
print("Decoded token ids:", result)`

// ── Quiz ──────────────────────────────────────────────────────────────────────

const QUIZ_QUESTIONS = [
  {
    question: 'Which value is carried from one time step to the next in a vanilla RNN?',
    options: ['The input embedding', 'The hidden state h_t', 'The output logit', 'The gradient'],
    correct: 1,
    explanation: 'The hidden state h_t encodes the running "memory" of all previous inputs and is passed to the next time step.',
  },
  {
    question: 'BPTT stands for:',
    options: ['Batch Propagation Through Trees', 'Backpropagation Through Time', 'Backward Pass for Temporal Training', 'Batch-Parallel Training Technique'],
    correct: 1,
    explanation: 'BPTT unrolls the RNN across time steps and applies standard backpropagation through the unrolled computation graph.',
  },
  {
    question: 'In teacher forcing, the decoder at step t receives:',
    options: ['The predicted token from step t−1', 'The ground-truth token from step t−1', 'A random token from the vocabulary', 'The encoder hidden state again'],
    correct: 1,
    explanation: 'Teacher forcing feeds the correct target token (not the model prediction) as the decoder input at each step, stabilising training.',
  },
  {
    question: 'Why do we apply a mask when computing the loss over padded sequences?',
    options: ['To speed up matrix multiplication', 'To prevent PAD tokens from contributing to the loss', 'To zero out the encoder output', 'To clip gradients'],
    correct: 1,
    explanation: 'PAD tokens are placeholder positions with no real meaning. Including them in the loss would push the model to predict PAD, skewing learning.',
  },
  {
    question: 'Greedy decoding at inference time picks the token with:',
    options: ['The lowest index', 'The highest probability at each step', 'A random sample from the distribution', 'The median probability'],
    correct: 1,
    explanation: 'Greedy decoding takes argmax over the output probability distribution at every step — simple but can miss globally better sequences.',
  },
]

// ── RNN unroll animation ──────────────────────────────────────────────────────

function RNNUnrollVis({ theme }) {
  const dark = theme === 'dark'
  const steps = ['x₀', 'x₁', 'x₂', 'x₃']
  const hidden = ['h₀', 'h₁', 'h₂', 'h₃']

  return (
    <div className="overflow-x-auto py-2">
      <div className="flex items-end gap-0 min-w-max mx-auto w-fit">
        {steps.map((xt, i) => (
          <motion.div
            key={i}
            className="flex flex-col items-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.15 }}
          >
            {/* hidden state box */}
            <div className={`w-16 h-12 rounded-lg flex items-center justify-center text-sm font-bold border-2
              ${dark ? 'bg-indigo-900/50 border-indigo-500 text-indigo-200' : 'bg-indigo-50 border-indigo-400 text-indigo-700'}`}>
              {hidden[i]}
            </div>

            {/* arrow up from RNN cell */}
            <div className={`w-0.5 h-4 ${dark ? 'bg-slate-500' : 'bg-slate-400'}`} />

            {/* RNN cell */}
            <div className={`w-16 h-12 rounded-lg flex items-center justify-center text-xs font-semibold border
              ${dark ? 'bg-violet-900/50 border-violet-500 text-violet-200' : 'bg-violet-50 border-violet-400 text-violet-700'}`}>
              RNN
            </div>

            {/* arrow down from input */}
            <div className={`w-0.5 h-4 ${dark ? 'bg-slate-500' : 'bg-slate-400'}`} />

            {/* input token */}
            <div className={`w-16 h-10 rounded-lg flex items-center justify-center text-sm font-bold border
              ${dark ? 'bg-cyan-900/40 border-cyan-500 text-cyan-200' : 'bg-cyan-50 border-cyan-400 text-cyan-700'}`}>
              {xt}
            </div>

            {/* horizontal arrow between cells (except last) */}
            {i < steps.length - 1 && (
              <div className="absolute" style={{ display: 'none' }} />
            )}
          </motion.div>
        ))}
      </div>
      {/* recurrent arrows */}
      <div className="flex items-center gap-0 min-w-max mx-auto w-fit mt-1">
        {steps.slice(0, -1).map((_, i) => (
          <div key={i} className="flex items-center" style={{ width: 64 }}>
            <div className={`flex-1 h-0.5 ${dark ? 'bg-violet-500' : 'bg-violet-400'}`} />
            <svg width="12" height="10" viewBox="0 0 12 10">
              <polygon points="0,0 12,5 0,10"
                fill={dark ? '#8b5cf6' : '#7c3aed'} />
            </svg>
            <div style={{ width: 0 }} />
          </div>
        ))}
      </div>
      <p className={`text-center text-xs mt-2 ${dark ? 'text-slate-400' : 'text-slate-500'}`}>
        Unrolled RNN — hidden state flows left to right through time
      </p>
    </div>
  )
}

// ── Encoder–Decoder diagram ───────────────────────────────────────────────────

function EncDecDiagram({ theme }) {
  const dark = theme === 'dark'
  return (
    <div className="overflow-x-auto">
      <svg viewBox="0 0 520 130" className="w-full max-w-xl mx-auto block" style={{ minWidth: 340 }}>
        {/* Encoder box */}
        <rect x="10" y="35" width="150" height="60" rx="10"
          fill={dark ? '#312e81' : '#eef2ff'} stroke={dark ? '#6366f1' : '#6366f1'} strokeWidth="1.5" />
        <text x="85" y="62" textAnchor="middle" fontSize="12" fontWeight="700"
          fill={dark ? '#c7d2fe' : '#4338ca'}>Encoder</text>
        <text x="85" y="80" textAnchor="middle" fontSize="10"
          fill={dark ? '#a5b4fc' : '#6366f1'}>x₁ … xₙ → hₙ</text>

        {/* Arrow encoder → context */}
        <line x1="160" y1="65" x2="210" y2="65" stroke={dark ? '#94a3b8' : '#64748b'} strokeWidth="1.5" markerEnd="url(#arr)" />

        {/* Context vector */}
        <rect x="210" y="48" width="90" height="34" rx="8"
          fill={dark ? '#1e3a5f' : '#f0f9ff'} stroke={dark ? '#38bdf8' : '#0ea5e9'} strokeWidth="1.5" />
        <text x="255" y="70" textAnchor="middle" fontSize="11" fontWeight="700"
          fill={dark ? '#7dd3fc' : '#0369a1'}>context c</text>

        {/* Arrow context → decoder */}
        <line x1="300" y1="65" x2="350" y2="65" stroke={dark ? '#94a3b8' : '#64748b'} strokeWidth="1.5" markerEnd="url(#arr)" />

        {/* Decoder box */}
        <rect x="350" y="35" width="155" height="60" rx="10"
          fill={dark ? '#3b0764' : '#fdf4ff'} stroke={dark ? '#a855f7' : '#a855f7'} strokeWidth="1.5" />
        <text x="427" y="62" textAnchor="middle" fontSize="12" fontWeight="700"
          fill={dark ? '#e9d5ff' : '#7e22ce'}>Decoder</text>
        <text x="427" y="80" textAnchor="middle" fontSize="10"
          fill={dark ? '#d8b4fe' : '#9333ea'}>c + y_{`{t-1}`} → y_t</text>

        {/* Source label */}
        <text x="85" y="115" textAnchor="middle" fontSize="10" fill={dark ? '#64748b' : '#94a3b8'}>source sequence</text>
        {/* Target label */}
        <text x="427" y="115" textAnchor="middle" fontSize="10" fill={dark ? '#64748b' : '#94a3b8'}>target sequence</text>

        <defs>
          <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill={dark ? '#94a3b8' : '#64748b'} />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function RNN() {
  const { theme, markTopicComplete } = useApp()
  const dark = theme === 'dark'

  return (
    <div className="space-y-10">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-4xl">🔄</span>
          <div>
            <p className={`text-sm font-semibold uppercase tracking-widest ${dark ? 'text-violet-400' : 'text-violet-600'}`}>
              Session 9 · Deep Learning
            </p>
            <h1 className={`text-3xl font-black ${dark ? 'text-white' : 'text-slate-900'}`}>
              Recurrent Neural Networks
            </h1>
          </div>
        </div>
        <p className={`text-lg leading-relaxed ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Standard feedforward networks see one fixed-size input at a time. RNNs process <strong>sequences</strong> by
          maintaining a hidden state that evolves with each new token — making them the foundation of language models,
          machine translation, and time-series analysis.
        </p>
      </motion.div>

      {/* ── Section 9.1 ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 9.1</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>RNN Architecture</h2>
        </div>

        {/* 9.1.1 Sequence data */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          9.1.1 — Sequence Data: Raw Text to Numerical Sequences
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Neural networks operate on numbers, not words. Converting text to sequences of integers involves two steps:
          building a <strong>vocabulary</strong> (word → index mapping) and <strong>encoding</strong> each sentence as a
          fixed-length integer array, padding shorter sentences with a special PAD token.
        </p>

        <TheoryBlock items={[
          { title: 'Tokenisation', content: 'Split raw text into tokens (words or sub-words). Each unique token gets an integer ID.' },
          { title: 'Vocabulary', content: 'A look-up table mapping token → integer. Special tokens: [PAD]=0 (padding), [UNK]=1 (unknown words).' },
          { title: 'Encoding', content: 'Replace each token with its integer ID. Short sequences are padded; long ones are truncated to a fixed length.' },
          { title: 'Embedding', content: 'Map each integer to a dense vector via a learned Embedding layer — capturing semantic similarity.' },
        ]} />

        <div className="mt-4">
          <CodeBlock code={CODE_TOKENISE} language="python" title="text_to_sequences.py" />
        </div>

        {/* 9.1.2 Recurrent connection */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          9.1.2 — Recurrent Connection and Hidden State
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          A vanilla RNN has one extra connection compared to a feedforward layer: the output (hidden state) is fed
          back as an additional input at the next time step. This creates a <strong>loop</strong> — the network has
          &quot;memory&quot; of previous inputs.
        </p>

        <RNNUnrollVis theme={theme} />

        <Callout type="formula" title="RNN update rule">
          h_t = tanh(W_h · h_{'t−1'} + W_x · x_t + b_h) <br />
          y_t = W_y · h_t + b_y
        </Callout>

        <TheoryBlock items={[
          { title: 'W_x (input weights)', content: 'Project input x_t (dimension D) into the hidden space (dimension H). Shape: H × D.' },
          { title: 'W_h (recurrent weights)', content: 'Mix the previous hidden state h_{t−1} into the current state. Shape: H × H. This is what makes it recurrent.' },
          { title: 'h_t (hidden state)', content: 'The "memory" vector. It summarises all tokens seen up to position t. Dimension H.' },
          { title: 'tanh non-linearity', content: 'Squashes values to (−1, 1), preventing unbounded growth of the hidden state over long sequences.' },
        ]} />

        {/* 9.1.3 Forward pass */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          9.1.3 — Forward Pass Formulation with Hidden States
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          The forward pass unrolls the loop across T time steps. We start with h₀ = <strong>0</strong>, then apply the
          same set of weights at every step — this <em>weight sharing</em> allows RNNs to handle sequences of variable
          length.
        </p>

        <CodeBlock code={CODE_FORWARD} language="python" title="rnn_forward.py" />

        <DeepDive title="Many-to-one vs many-to-many">
          <TheoryBlock items={[
            { title: 'Many-to-one', content: 'Use only the final hidden state h_T as the sequence representation. Common for classification (sentiment, spam).' },
            { title: 'Many-to-many (same length)', content: 'Return h_t at every step (return_sequences=True). Used for tagging tasks: POS, NER.' },
            { title: 'Many-to-many (diff length)', content: 'Encoder–decoder: encode source → pass final state → decode to produce variable-length output.' },
            { title: 'One-to-many', content: 'Single input seeds a sequence output — e.g. image captioning (image vector → sentence).' },
          ]} />
        </DeepDive>

        {/* 9.1.4 BPTT */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          9.1.4 — Backpropagation Through Time (BPTT)
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Because the hidden state at step t depends on h_{'t−1'}, which in turn depends on h_{'t−2'}, and so on,
          gradients must flow backward through all unrolled time steps. This is <strong>BPTT</strong>.
          The chain rule applied over T steps can make gradients vanishingly small (<em>vanishing gradient</em>) or
          explosively large (<em>exploding gradient</em>).
        </p>

        <Callout type="warning" title="Vanishing / Exploding Gradients">
          During BPTT the gradient is multiplied by W_h at every step. If the largest singular value of W_h is
          &lt; 1, gradients shrink exponentially. If &gt; 1, they explode. <strong>Gradient clipping</strong> bounds
          the norm; <strong>LSTMs/GRUs</strong> add gating to mitigate vanishing.
        </Callout>

        <CodeBlock code={CODE_BPTT} language="python" title="bptt_tape.py" />

        <DeepDive title="Truncated BPTT">
          <TheoryBlock items={[
            { title: 'Full BPTT', content: 'Unroll the entire sequence, then backprop. Exact but memory-intensive for long sequences.' },
            { title: 'Truncated BPTT', content: 'Unroll only k steps (e.g. 35), carry hidden state forward, backprop only through k steps. Memory O(k) instead of O(T).' },
            { title: 'Gradient clipping', content: 'If ||g|| > threshold, scale g ← threshold × g / ||g||. Prevents exploding gradients without stopping learning.' },
          ]} />
        </DeepDive>

        {/* 9.1.5 RNNs for NLP */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          9.1.5 — RNNs for NLP: Language Modelling &amp; Classification
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Two archetypal NLP tasks illustrate how RNN hidden states are used:
        </p>

        <TheoryBlock cols={2} items={[
          {
            title: '📖 Language Modelling',
            content: 'Predict the next token given all previous tokens: P(w_t | w_1 … w_{t−1}). Train by teacher forcing — at each step, the input is the ground-truth previous token and the target is the current token.',
          },
          {
            title: '🏷️ Text Classification',
            content: 'Encode the full sequence with an RNN, take the final hidden state h_T, and pass it through a Dense + Softmax head. The entire pipeline is end-to-end differentiable.',
          },
        ]} />

        <div className="mt-4">
          <CodeBlock code={CODE_LM} language="python" title="char_lm.py" />
        </div>
      </motion.div>

      {/* ── Section 9.2 ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-violet-500/5 border-violet-500/20' : 'bg-violet-50 border-violet-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-violet-400' : 'text-violet-600'}`}>Section 9.2</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Encoder-Decoder and Training</h2>
        </div>

        {/* 9.2.1 Enc-Dec */}
        <h3 className={`text-lg font-bold mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          9.2.1 — Encoder–Decoder Architecture
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Sequence-to-sequence tasks (translation, summarisation) produce output sequences of a <em>different</em> length
          than the input. The encoder–decoder pattern solves this by first compressing the source into a
          <strong> context vector</strong>, then generating the target token-by-token conditioned on that vector.
        </p>

        <EncDecDiagram theme={theme} />

        <div className="mt-4">
          <TheoryBlock items={[
            { title: 'Encoder', content: 'Reads the source sequence x₁…xₙ with an RNN. The final hidden state h_n becomes the context vector c that summarises the entire source.' },
            { title: 'Context vector c', content: 'The encoder\'s final hidden state passed to the decoder as its initial state. It is the only channel of information between the two RNNs.' },
            { title: 'Decoder', content: 'A separate RNN seeded with c. At each step it consumes its previous output (or the ground-truth token during training) and c to predict the next token.' },
            { title: 'Bottleneck limitation', content: 'Compressing the entire source into a single fixed-size vector is a bottleneck for long sequences. Attention mechanisms (next session) address this.' },
          ]} />
        </div>

        <div className="mt-4">
          <CodeBlock code={CODE_ENCDEC} language="python" title="encoder_decoder.py" />
        </div>

        {/* 9.2.2 Teacher forcing */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          9.2.2 — Teacher Forcing During Training
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          At training time we know the correct target sequence. Instead of feeding the decoder's own (possibly wrong)
          prediction at step t as input to step t+1, we feed the <strong>ground-truth token</strong>. This is
          <em> teacher forcing</em> — it stabilises training by preventing error accumulation.
        </p>

        <Callout type="info" title="Exposure bias">
          Teacher forcing creates a train–test mismatch: at inference the decoder receives its own predictions, which
          may differ from the ground-truth distribution seen during training. Techniques like <strong>scheduled
          sampling</strong> (gradually replacing teacher tokens with model predictions) reduce this gap.
        </Callout>

        <CodeBlock code={CODE_TEACHER_FORCE} language="python" title="teacher_forcing.py" />

        <DeepDive title="Scheduled Sampling">
          <TheoryBlock items={[
            { title: '', content: 'Start training with full teacher forcing (p=1.0). Anneal p toward 0 over training epochs. At p=0 the decoder is entirely self-fed — matching inference conditions.' },
            { title: '', content: 'A common schedule: p(epoch) = k / (k + exp(epoch / k)). The hyperparameter k controls how fast exposure to the model\'s own outputs increases.' },
          ]} />
        </DeepDive>

        {/* 9.2.3 Masked loss */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          9.2.3 — Loss Function with Masking (Padding Tokens)
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Sequences in a batch are padded to the same length with a PAD token (index 0). If the loss is computed
          naively over all positions, the model wastes capacity learning to predict PAD at padding positions.
          A <strong>mask</strong> zeroes out those contributions.
        </p>

        <Callout type="formula" title="Masked cross-entropy">
          L = Σ_{'{t : y_t ≠ PAD}'} CE(ŷ_t, y_t) / |{'{t : y_t ≠ PAD}'}|
        </Callout>

        <CodeBlock code={CODE_MASKED_LOSS} language="python" title="masked_loss.py" />

        <TheoryBlock items={[
          { title: 'Padding mask', content: 'Boolean tensor: True where y_t ≠ PAD. Multiply element-wise with the per-token loss vector before averaging.' },
          { title: 'Length normalisation', content: 'Divide by the number of non-pad tokens (not the sequence length) so that longer real content does not produce an artificially larger loss.' },
          { title: 'Keras integration', content: 'Pass sample_weight=mask or use model.compile(loss=masked_sparse_ce) to plug in the custom masked loss seamlessly.' },
        ]} />

        {/* 9.2.4 Training & greedy decoding */}
        <h3 className={`text-lg font-bold mt-8 mb-3 ${dark ? 'text-white' : 'text-slate-800'}`}>
          9.2.4 — Training and Prediction (Greedy Decoding)
        </h3>
        <p className={`mb-4 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>
          Training and inference use <em>different</em> computational graphs. During training the full target sequence
          is available; during inference we generate one token at a time, feeding each prediction back into the decoder
          until an EOS token or maximum length is reached. <strong>Greedy decoding</strong> picks the highest-probability
          token at each step.
        </p>

        <TheoryBlock cols={2} items={[
          {
            title: '🏋️ Training graph',
            content: 'Encoder reads source, passes state to decoder. Decoder receives [BOS, y₁, …, y_{T−1}] (teacher forced) and outputs logits for [y₁, …, y_T]. Loss + BPTT.',
          },
          {
            title: '🔮 Inference graph',
            content: 'Encoder runs once. Decoder runs step by step: feed BOS → get y₁ → feed y₁ → get y₂ → … until EOS or max length. Argmax at every step = greedy decoding.',
          },
        ]} />

        <div className="mt-4">
          <CodeBlock code={CODE_GREEDY} language="python" title="greedy_decode.py" />
        </div>

        <DeepDive title="Beyond Greedy: Beam Search">
          <TheoryBlock items={[
            { title: 'Problem with greedy', content: 'Greedy picks the locally best token — but the globally best sequence may require a suboptimal early choice (e.g. "I am" vs "I\'m").' },
            { title: 'Beam search (k beams)', content: 'Keep the k highest-scoring partial sequences at each step. At the end, return the beam with the highest total log-probability.' },
            { title: 'Length penalty', content: 'Longer sequences accumulate more log-prob terms. Divide total score by length^α (α ≈ 0.6–0.7) to avoid preference for short outputs.' },
          ]} />
        </DeepDive>
      </motion.div>

      {/* ── Quiz ── */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <div className={`rounded-xl p-4 mb-6 border ${dark ? 'bg-cyan-500/5 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
          <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${dark ? 'text-cyan-400' : 'text-cyan-600'}`}>Knowledge Check</p>
          <h2 className={`text-xl font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>Quiz — Sessions 9.1 &amp; 9.2</h2>
        </div>
        <Quiz
          questions={QUIZ_QUESTIONS}
          onComplete={() => markTopicComplete('rnn')}
        />
      </motion.div>
    </div>
  )
}
