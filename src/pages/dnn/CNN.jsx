import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE_CONV = `import tensorflow as tf
import numpy as np

# Input: shape (batch, height, width, channels) = (1, 5, 5, 1)
x = np.array([[[[1],[2],[3],[0],[1]],
               [[0],[1],[2],[1],[0]],
               [[1],[0],[1],[3],[2]],
               [[2],[1],[0],[1],[1]],
               [[0],[1],[2],[0],[1]]]], dtype=np.float32)

# Edge-detection kernel: shape (kH, kW, C_in, C_out) = (3, 3, 1, 1)
kernel = np.array([[[[-1],[-1],[-1]],
                    [[-1],[ 8],[-1]],
                    [[-1],[-1],[-1]]]], dtype=np.float32).transpose(1,2,0,3)

# Valid padding (no padding), stride=1
out_valid = tf.nn.conv2d(x, kernel, strides=1, padding='VALID')
print("Valid output shape:", out_valid.shape)  # (1,3,3,1)

# Same padding — output matches input spatial size
out_same  = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')
print("Same  output shape:", out_same.shape)   # (1,5,5,1)

# Stride 2
out_stride = tf.nn.conv2d(x, kernel, strides=2, padding='VALID')
print("Stride-2 shape:", out_stride.shape)     # (1,2,2,1)

# Output size formula: floor((H + 2P - K) / S) + 1
H, P, K, S = 5, 0, 3, 1
print(f"Formula: ({H}+2*{P}-{K})/{S}+1 = {(H+2*P-K)//S+1}")`

const PYTHON_CODE_LENET = `import tensorflow as tf

# LeNet-5 adapted for 28×28 MNIST (channels-last: H,W,C)
model = tf.keras.Sequential([
    # Block 1: conv 5×5, 6 filters, same padding → 28×28×6
    tf.keras.layers.Conv2D(6,  kernel_size=5, padding='same', activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),   # → 14×14×6

    # Block 2: conv 5×5, 16 filters, valid → 10×10×16
    tf.keras.layers.Conv2D(16, kernel_size=5, padding='valid', activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),   # → 5×5×16

    # Classifier
    tf.keras.layers.Flatten(),          # 400
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84,  activation='relu'),
    tf.keras.layers.Dense(10,  activation='softmax'),  # 10 digit classes
])

model.summary()
print(f"Total params: {model.count_params():,}")

# Load MNIST and train
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None].astype('float32') / 255.0   # (60k,28,28,1)
x_test  = x_test[...,  None].astype('float32') / 255.0

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=3, batch_size=64,
                    validation_data=(x_test, y_test))
for ep, (l, a) in enumerate(zip(history.history['loss'],
                                 history.history['val_accuracy'])):
    print(f"Epoch {ep+1}  Loss: {l:.4f}  Val Acc: {a:.2%}")`

const QUIZ_QUESTIONS = [
  { question: 'A colour image of size 32×32 pixels is stored as a tensor of shape:', options: ['(32, 32)', '(3, 32, 32)', '(32, 32, 1)', '(3, 1024)'], correct: 1, explanation: 'Colour images have 3 channels (RGB). PyTorch stores them as (C, H, W) = (3, 32, 32). A batch of 64 would be (64, 3, 32, 32).' },
  { question: 'Shared weights in a CNN mean:', options: ['All layers share the same weights', 'The same kernel slides over all spatial positions', 'Weights are initialised to the same value', 'All filters are identical'], correct: 1, explanation: 'A single filter (kernel) is convolved across every position of the input — the same d×d weights are reused everywhere. This is weight sharing and dramatically reduces parameters.' },
  { question: 'Why do fully-connected layers struggle with images?', options: ['They cannot handle colour', 'Parameter count grows quadratically with image size; no spatial structure exploited', 'They require normalisation', 'They cannot backpropagate'], correct: 1, explanation: 'A 224×224 RGB image has 150,528 pixels. One FC layer with 1000 hidden units needs 150M parameters — just for the first layer. CNNs exploit locality and weight sharing to use far fewer.' },
  { question: 'The output size of a convolution with H=28, K=3, P=0, S=1 is:', options: ['28', '26', '27', '30'], correct: 1, explanation: 'Formula: floor((H + 2P − K)/S) + 1 = (28+0−3)/1+1 = 26.' },
  { question: '"Same" padding ensures:', options: ['Output is smaller than input', 'Output has the same spatial size as input', 'No zeros are added', 'Stride is always 1'], correct: 1, explanation: 'Same padding adds P = floor(K/2) zeros around the input so that with stride=1 the output H×W matches the input H×W.' },
  { question: 'Stride 2 in a convolution:', options: ['Doubles output size', 'Halves output size (approximately)', 'Has no effect on size', 'Only works with square kernels'], correct: 1, explanation: 'Stride S moves the kernel S pixels per step. Stride 2 roughly halves each spatial dimension: floor((H−K)/2)+1 ≈ H/2.' },
  { question: 'A 1×1 convolution on a (B, C, H, W) tensor:', options: ['Removes spatial dimensions', 'Mixes channel information without changing H×W', 'Is equivalent to max pooling', 'Only works with C=1'], correct: 1, explanation: 'A 1×1 conv applies a learned linear combination across channels at every spatial position. It changes the number of channels (C_out) while keeping H and W the same — used in bottleneck layers.' },
  { question: 'Max pooling with kernel 2×2 and stride 2:', options: ['Doubles the feature map', 'Halves H and W, keeps C unchanged', 'Changes the number of channels', 'Applies a learnable filter'], correct: 1, explanation: 'Max pooling takes the maximum value in each 2×2 window, reducing H and W by 2× while keeping the number of channels C unchanged. It has no learnable parameters.' },
  { question: 'The receptive field of a neuron refers to:', options: ['The kernel size', 'The region of the original input that affects that neuron\'s output', 'The number of output channels', 'The stride value'], correct: 1, explanation: 'The receptive field is the patch of the original input image that influenced a particular neuron. It grows with depth — deep neurons "see" large regions of the input.' },
  { question: 'LeNet\'s key architectural contribution was:', options: ['Using ReLU activations', 'Alternating convolution + pooling layers to extract hierarchical features, followed by FC layers', 'Introducing batch normalisation', 'Using skip connections'], correct: 1, explanation: 'LeNet (1998) pioneered the conv→pool→conv→pool→FC architecture that became the CNN blueprint. It showed that convolutions could automatically learn spatial features from raw pixels.' },
]

// ── Pixel tensor visualiser ────────────────────────────────────────────────────
function PixelTensorViz({ theme }) {
  const [channel, setChannel] = useState(0)
  const channelNames = ['Red', 'Green', 'Blue']
  const channelColors = ['#EF4444', '#10B981', '#3B82F6']

  const pixels = [
    [[220, 40, 60], [255, 80, 20], [200, 160, 30], [180, 210, 80]],
    [[150, 60, 90], [210, 130, 50], [160, 200, 100], [120, 240, 130]],
    [[100, 90, 120], [170, 150, 80], [130, 180, 160], [90, 200, 190]],
    [[80, 110, 150], [120, 160, 120], [100, 140, 200], [70, 160, 220]],
  ]

  const card = `rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        {/* RGB image */}
        <div className={card}>
          <p className={`text-xs font-semibold mb-3 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>RGB Image (4×4 px)</p>
          <div className="grid grid-cols-4 gap-0.5 w-fit mx-auto rounded overflow-hidden border border-white/10">
            {pixels.flat().map(([r, g, b], i) => (
              <div key={i} className="w-10 h-10" style={{ backgroundColor: `rgb(${r},${g},${b})` }} />
            ))}
          </div>
          <p className={`text-xs mt-2 text-center font-mono ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>shape: (3, 4, 4)</p>
        </div>
        {/* Single channel */}
        <div className={card}>
          <div className="flex gap-1 mb-3">
            {channelNames.map((n, i) => (
              <button key={n} onClick={() => setChannel(i)}
                className={`flex-1 py-1 rounded text-xs font-semibold border transition-all ${channel === i
                  ? 'text-white border-transparent' : theme === 'dark' ? 'bg-slate-700 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}
                style={channel === i ? { backgroundColor: channelColors[i] } : {}}>
                {n}
              </button>
            ))}
          </div>
          <div className="grid grid-cols-4 gap-0.5 w-fit mx-auto rounded overflow-hidden border border-white/10">
            {pixels.flat().map((px, i) => {
              const v = px[channel]
              const bg = channel === 0 ? `rgb(${v},0,0)` : channel === 1 ? `rgb(0,${v},0)` : `rgb(0,0,${v})`
              return (
                <div key={i} className="w-10 h-10 flex items-center justify-center"
                  style={{ backgroundColor: bg }}>
                  <span className="text-[9px] font-mono font-bold text-white/80">{v}</span>
                </div>
              )
            })}
          </div>
          <p className={`text-xs mt-2 text-center font-mono ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>shape: (1, 4, 4) — {channelNames[channel]} only</p>
        </div>
      </div>
      <Callout type="formula" mono="Tensor shape: (Batch, Channels, Height, Width)">
        PyTorch convention: (B, C, H, W). A batch of 32 RGB images at 224×224 = (32, 3, 224, 224). Total values = 32 × 3 × 224 × 224 = 4,816,896 floats ≈ 18 MB (float32).
      </Callout>
    </div>
  )
}

// ── Convolution interactive demo ───────────────────────────────────────────────
function ConvDemo({ theme }) {
  const [kernelType, setKernelType] = useState('edge')
  const [padding, setPadding] = useState(0)
  const [stride, setStride] = useState(1)
  const [pos, setPos] = useState({ r: 0, c: 0 })
  const canvasRef = useRef(null)

  const kernels = {
    edge:   [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
    blur:   [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
    sharpen:[[0,-1,0],[-1,5,-1],[0,-1,0]],
    sobelx: [[-1,0,1],[-2,0,2],[-1,0,1]],
  }

  const input5 = [
    [1,2,3,0,1],
    [0,1,2,1,0],
    [1,0,1,3,2],
    [2,1,0,1,1],
    [0,1,2,0,1],
  ]

  const K = kernels[kernelType]
  const H = 5, kSize = 3
  const outH = Math.floor((H + 2 * padding - kSize) / stride) + 1

  // Pad input
  const padded = Array.from({ length: H + 2 * padding }, (_, r) =>
    Array.from({ length: H + 2 * padding }, (_, c) => {
      const ir = r - padding, ic = c - padding
      return (ir >= 0 && ir < H && ic >= 0 && ic < H) ? input5[ir][ic] : 0
    })
  )

  // Compute full output
  const output = Array.from({ length: outH }, (_, or) =>
    Array.from({ length: outH }, (_, oc) => {
      let s = 0
      for (let kr = 0; kr < kSize; kr++)
        for (let kc = 0; kc < kSize; kc++)
          s += padded[or * stride + kr][oc * stride + kc] * K[kr][kc]
      return parseFloat(s.toFixed(2))
    })
  )

  const curVal = output[pos.r]?.[pos.c] ?? 0
  const safeR = Math.min(pos.r, outH - 1)
  const safeC = Math.min(pos.c, outH - 1)
  const paddedR = safeR * stride, paddedC = safeC * stride

  const cellSz = 36
  const inputSz = (H + 2 * padding) * cellSz
  const W = Math.max(inputSz + 20, 200), H_CANVAS = inputSz + 20

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H_CANVAS)
    const off = 10
    const pSize = H + 2 * padding

    // Draw padded input grid
    for (let r = 0; r < pSize; r++) {
      for (let c = 0; c < pSize; c++) {
        const v = padded[r][c]
        const isOrig = r >= padding && r < H + padding && c >= padding && c < H + padding
        const inReceptive = r >= paddedR && r < paddedR + kSize && c >= paddedC && c < paddedC + kSize
        let bg
        if (inReceptive) bg = theme === 'dark' ? '#312E81' : '#EEF2FF'
        else if (!isOrig) bg = theme === 'dark' ? '#0F172A' : '#F9FAFB'
        else bg = theme === 'dark' ? '#1E293B' : '#F8FAFC'

        ctx.fillStyle = bg
        ctx.fillRect(off + c * cellSz, off + r * cellSz, cellSz - 1, cellSz - 1)
        ctx.strokeStyle = inReceptive ? '#6366F1' : (theme === 'dark' ? '#334155' : '#E2E8F0')
        ctx.lineWidth = inReceptive ? 2 : 1
        ctx.strokeRect(off + c * cellSz, off + r * cellSz, cellSz - 1, cellSz - 1)
        ctx.fillStyle = theme === 'dark' ? (isOrig ? '#E2E8F0' : '#4B5563') : (isOrig ? '#1E293B' : '#9CA3AF')
        ctx.font = 'bold 11px monospace'; ctx.textAlign = 'center'
        ctx.fillText(v, off + c * cellSz + cellSz / 2, off + r * cellSz + cellSz / 2 + 4)
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [padding, stride, pos, theme, kernelType])

  const maxV = Math.max(...output.flat().map(Math.abs), 0.001)

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.keys(kernels).map(k => (
          <button key={k} onClick={() => setKernelType(k)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold border capitalize transition-all ${kernelType === k
              ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
              : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{k}</button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Padding P = <span className="text-indigo-400 font-bold">{padding}</span>
          </label>
          <input type="range" min="0" max="2" step="1" value={padding}
            onChange={e => { setPadding(+e.target.value); setPos({ r: 0, c: 0 }) }}
            className="w-full accent-indigo-500" />
          <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>{padding === 0 ? 'Valid — output shrinks' : padding === 1 ? 'Same — output = input size' : 'Full padding'}</p>
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Stride S = <span className="text-purple-400 font-bold">{stride}</span>
          </label>
          <input type="range" min="1" max="2" step="1" value={stride}
            onChange={e => { setStride(+e.target.value); setPos({ r: 0, c: 0 }) }}
            className="w-full accent-purple-500" />
          <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>{stride === 1 ? 'Step 1 pixel' : 'Step 2 — output halved'}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
        {/* Input + receptive field */}
        <div className={`rounded-xl border p-3 col-span-1 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
          <p className={`text-xs font-semibold mb-2 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Input (padded)</p>
          <canvas ref={canvasRef} width={W} height={H_CANVAS} className="w-full rounded" style={{ maxWidth: W, height: H_CANVAS }} />
        </div>

        {/* Kernel */}
        <div className={`rounded-xl border p-3 col-span-1 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
          <p className={`text-xs font-semibold mb-2 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Kernel ({kernelType})</p>
          <div className="grid grid-cols-3 gap-0.5">
            {K.flat().map((v, i) => (
              <div key={i} className={`h-9 flex items-center justify-center rounded text-xs font-mono font-bold border ${
                v > 0 ? 'text-indigo-400 border-indigo-500/30 bg-indigo-500/10' :
                v < 0 ? 'text-red-400 border-red-500/30 bg-red-500/10' :
                (theme === 'dark' ? 'text-gray-500 border-white/10 bg-slate-800' : 'text-gray-400 border-gray-200 bg-gray-100')}`}>
                {typeof v === 'number' && Math.abs(v) < 0.2 ? v.toFixed(2) : v}
              </div>
            ))}
          </div>
          <p className={`text-xs mt-2 font-mono text-center ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>3×3 kernel</p>
        </div>

        {/* Output feature map */}
        <div className={`rounded-xl border p-3 col-span-1 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
          <p className={`text-xs font-semibold mb-2 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Feature map ({outH}×{outH})</p>
          <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${outH}, 1fr)` }}>
            {output.map((row, r) => row.map((v, c) => {
              const intensity = Math.abs(v) / maxV
              const active = r === safeR && c === safeC
              return (
                <button key={`${r}-${c}`}
                  onClick={() => setPos({ r, c })}
                  className={`h-8 flex items-center justify-center rounded text-[10px] font-mono font-bold border transition-all ${active
                    ? 'border-yellow-400 ring-1 ring-yellow-400' : theme === 'dark' ? 'border-white/10' : 'border-gray-200'}`}
                  style={{
                    backgroundColor: v > 0
                      ? `rgba(99,102,241,${0.1 + intensity * 0.6})`
                      : v < 0 ? `rgba(239,68,68,${0.1 + intensity * 0.6})`
                      : (theme === 'dark' ? '#1E293B' : '#F8FAFC'),
                    color: intensity > 0.5 ? 'white' : (theme === 'dark' ? '#9CA3AF' : '#6B7280'),
                  }}>
                  {v > 99 ? v.toFixed(0) : v.toFixed(1)}
                </button>
              )
            }))}
          </div>
          <p className={`text-xs mt-2 font-mono text-center ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
            ⊙ active: {curVal}
          </p>
        </div>
      </div>

      <div className={`rounded-xl p-3 border font-mono text-xs ${theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-300' : 'bg-gray-100 border-gray-200 text-gray-700'}`}>
        Output size = ⌊(5 + 2×{padding} − 3) / {stride}⌋ + 1 = <span className="text-indigo-400 font-bold">{outH}</span>
        &nbsp;×&nbsp;<span className="text-indigo-400 font-bold">{outH}</span>
        &nbsp;·&nbsp; Click any output cell to highlight its receptive field
      </div>
    </div>
  )
}

// ── Pooling demo ───────────────────────────────────────────────────────────────
function PoolingDemo({ theme }) {
  const [poolType, setPoolType] = useState('max')

  const input = [
    [1,3,2,4],
    [5,6,1,2],
    [3,2,4,1],
    [1,0,2,3],
  ]

  const pool = (r, c) => {
    const patch = [input[r][c], input[r][c+1], input[r+1][c], input[r+1][c+1]]
    return poolType === 'max' ? Math.max(...patch) : parseFloat((patch.reduce((a,b)=>a+b,0)/4).toFixed(2))
  }

  const output = [[pool(0,0),pool(0,2)],[pool(2,0),pool(2,2)]]
  const [hover, setHover] = useState(null)

  const cellBg = (r, c) => {
    if (!hover) return ''
    const [or, oc] = hover
    const inPool = r >= or*2 && r < or*2+2 && c >= oc*2 && c < oc*2+2
    return inPool ? (theme === 'dark' ? 'ring-2 ring-indigo-400 bg-indigo-500/20' : 'ring-2 ring-indigo-400 bg-indigo-50') : ''
  }

  const card = `rounded-xl border p-3 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`

  return (
    <div>
      <div className="flex gap-2 mb-4">
        {['max', 'average'].map(t => (
          <button key={t} onClick={() => setPoolType(t)}
            className={`flex-1 py-2 rounded-lg text-xs font-semibold border capitalize transition-all ${poolType === t
              ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
              : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{t} pooling</button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-6">
        <div className={card}>
          <p className={`text-xs font-semibold mb-2 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Input (4×4)</p>
          <div className="grid grid-cols-4 gap-0.5">
            {input.map((row, r) => row.map((v, c) => (
              <div key={`${r}-${c}`}
                className={`h-10 flex items-center justify-center text-sm font-mono font-bold rounded border transition-all ${cellBg(r, c)} ${theme === 'dark' ? 'border-white/10 bg-slate-800 text-white' : 'border-gray-200 bg-white text-gray-900'}`}>{v}</div>
            )))}
          </div>
        </div>
        <div className={card}>
          <p className={`text-xs font-semibold mb-2 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Output (2×2) — hover cells</p>
          <div className="grid grid-cols-2 gap-1">
            {output.map((row, r) => row.map((v, c) => (
              <button key={`${r}-${c}`}
                onMouseEnter={() => setHover([r, c])} onMouseLeave={() => setHover(null)}
                className={`h-14 flex items-center justify-center text-base font-mono font-bold rounded border transition-all ${theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/30 text-indigo-300 hover:bg-indigo-500/30' : 'bg-indigo-50 border-indigo-200 text-indigo-700 hover:bg-indigo-100'}`}>
                {v}
              </button>
            )))}
          </div>
          <p className={`text-xs mt-2 text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
            {poolType === 'max' ? 'Each cell = max of 2×2 patch' : 'Each cell = mean of 2×2 patch'}
          </p>
        </div>
      </div>
    </div>
  )
}

// ── LeNet architecture diagram ─────────────────────────────────────────────────
function LeNetDiagram({ theme }) {
  const layers = [
    { label: 'Input', shape: '1×28×28', color: '#6B7280', w: 28, h: 28 },
    { label: 'Conv1', shape: '6×28×28', color: '#6366F1', w: 28, h: 28, detail: '5×5, p=2' },
    { label: 'Pool1', shape: '6×14×14', color: '#8B5CF6', w: 14, h: 14, detail: '2×2 avg' },
    { label: 'Conv2', shape: '16×10×10', color: '#6366F1', w: 10, h: 10, detail: '5×5' },
    { label: 'Pool2', shape: '16×5×5', color: '#8B5CF6', w: 5, h: 5, detail: '2×2 avg' },
    { label: 'Flatten', shape: '400', color: '#06B6D4', w: 4, h: 4 },
    { label: 'FC1', shape: '120', color: '#10B981', w: 3, h: 3, detail: 'ReLU' },
    { label: 'FC2', shape: '84', color: '#10B981', w: 2.5, h: 2.5, detail: 'ReLU' },
    { label: 'Output', shape: '10', color: '#F59E0B', w: 2, h: 2, detail: 'softmax' },
  ]

  const [active, setActive] = useState(null)

  const descriptions = {
    'Input': 'Raw greyscale image. 28×28 = 784 pixel values, each in [0,1].',
    'Conv1': '6 filters of size 5×5 with padding=2 → keeps spatial size 28×28. Detects low-level features: edges, corners.',
    'Pool1': 'Average pooling 2×2, stride 2 → halves to 14×14. Reduces computation, adds spatial invariance.',
    'Conv2': '16 filters of size 5×5, no padding → 10×10. Detects higher-level patterns from the 6 feature maps.',
    'Pool2': 'Average pooling 2×2 → 5×5. Final spatial downsampling before flattening.',
    'Flatten': 'Reshapes 16×5×5 = 400 values into a 1D vector for the dense layers.',
    'FC1': 'Fully-connected: 400 → 120 neurons, ReLU. Learns combinations of spatial features.',
    'FC2': 'Fully-connected: 120 → 84 neurons, ReLU. Further abstraction.',
    'Output': '84 → 10 neurons, one per digit class. Softmax gives class probabilities.',
  }

  return (
    <div>
      <div className="overflow-x-auto pb-2">
        <div className="flex items-center gap-1 min-w-max mx-auto w-fit">
          {layers.map((layer, i) => (
            <div key={layer.label} className="flex items-center">
              <button
                onClick={() => setActive(active === layer.label ? null : layer.label)}
                className="flex flex-col items-center gap-1 group"
              >
                <div
                  className={`rounded-lg border-2 transition-all flex items-center justify-center text-white font-bold text-xs ${active === layer.label ? 'scale-110 shadow-lg' : 'opacity-80 hover:opacity-100'}`}
                  style={{
                    width: Math.max(layer.w * 2.2, 28),
                    height: Math.max(layer.h * 2.2, 28),
                    backgroundColor: layer.color + '33',
                    borderColor: layer.color,
                    minWidth: 28, minHeight: 28,
                  }}
                >
                  <span style={{ color: layer.color, fontSize: 9 }}>{layer.w === layer.h ? `${layer.w}` : ''}</span>
                </div>
                <span className={`text-[9px] font-semibold ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{layer.label}</span>
                {layer.detail && <span className={`text-[8px] ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>{layer.detail}</span>}
              </button>
              {i < layers.length - 1 && (
                <div className={`w-4 h-px mx-0.5 ${theme === 'dark' ? 'bg-white/20' : 'bg-gray-300'}`} />
              )}
            </div>
          ))}
        </div>
      </div>
      <div className={`mt-4 rounded-xl border p-3 transition-all ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
        {active ? (
          <div>
            <p className={`text-xs font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{active}
              <span className={`ml-2 font-mono ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>{layers.find(l => l.label === active)?.shape}</span>
            </p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{descriptions[active]}</p>
          </div>
        ) : (
          <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>Click any layer block to see details</p>
        )}
      </div>
      <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
        {[
          { label: 'Conv layers', val: '2', color: 'text-indigo-400' },
          { label: 'Pool layers', val: '2', color: 'text-purple-400' },
          { label: 'FC layers', val: '3', color: 'text-emerald-400' },
        ].map(m => (
          <div key={m.label} className={`rounded-xl p-2 border text-center ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`font-mono font-bold text-lg ${m.color}`}>{m.val}</p>
            <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.label}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Receptive field growth visualiser ─────────────────────────────────────────
function ReceptiveFieldViz({ theme }) {
  const [depth, setDepth] = useState(1)
  // RF after d layers of 3×3 conv, stride 1: RF = 2d+1
  const rf = depth * 2 + 1
  const gridSize = 9
  const center = Math.floor(gridSize / 2)
  const half = Math.floor(rf / 2)

  return (
    <div>
      <div className="mb-4">
        <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
          Depth (# conv layers) = <span className="text-indigo-400 font-bold">{depth}</span>
          &nbsp;→ Receptive field = <span className="text-emerald-400 font-bold">{rf}×{rf}</span>
        </label>
        <input type="range" min="1" max="4" step="1" value={depth} onChange={e => setDepth(+e.target.value)} className="w-full accent-indigo-500" />
      </div>
      <div className={`rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
        <div className="grid gap-0.5 w-fit mx-auto" style={{ gridTemplateColumns: `repeat(${gridSize}, 2rem)` }}>
          {Array.from({ length: gridSize }, (_, r) =>
            Array.from({ length: gridSize }, (_, c) => {
              const inRF = Math.abs(r - center) <= half && Math.abs(c - center) <= half
              const isCenter = r === center && c === center
              return (
                <motion.div key={`${r}-${c}`}
                  animate={{ backgroundColor: isCenter ? '#6366F1' : inRF ? (theme === 'dark' ? '#312E81' : '#EEF2FF') : (theme === 'dark' ? '#1E293B' : '#F8FAFC') }}
                  transition={{ duration: 0.2 }}
                  className={`h-8 rounded border flex items-center justify-center text-[10px] font-mono ${
                    isCenter ? 'text-white font-bold border-indigo-400' : inRF ? (theme === 'dark' ? 'text-indigo-300 border-indigo-500/40' : 'text-indigo-600 border-indigo-200') : (theme === 'dark' ? 'text-gray-700 border-white/5' : 'text-gray-300 border-gray-200')
                  }`}>
                  {isCenter ? '★' : inRF ? '·' : ''}
                </motion.div>
              )
            })
          )}
        </div>
        <p className={`text-xs mt-3 text-center ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
          After {depth} layer{depth > 1 ? 's' : ''} of 3×3 conv: a single output neuron (★) sees a {rf}×{rf} patch of the input
        </p>
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function CNN() {
  const { theme } = useApp()
  const S = `rounded-2xl border p-6 mb-6 ${theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'}`
  const LBL = `text-xs font-semibold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`
  const H2 = `text-xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const BODY = `text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl">

      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/20 border border-cyan-500/30 mb-4">
          <span className="text-xs text-cyan-400 font-medium">Deep Neural Networks • Session 6</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Convolutional <span className="gradient-text">Neural Networks</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          CNNs exploit the <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>spatial structure</strong> of images — using shared kernels, local connections, and pooling to learn hierarchical features with far fewer parameters than fully-connected networks.
        </p>
        <Callout type="analogy" title="Analogy: a sliding magnifying glass">
          A CNN doesn't look at the whole image at once. Instead, it slides a small filter (magnifying glass) across every patch, looking for a specific pattern — an edge, a corner, a texture. Different filters detect different patterns. Stacking layers detects increasingly complex features.
        </Callout>
      </div>

      <TheoryBlock title="Core Concepts" cards={[
        { icon: '🖼️', title: 'Image tensor', body: 'Images are 3D tensors: (C, H, W). Colour = 3 channels (RGB). A batch adds a dimension: (B, C, H, W). Every pixel is a float, usually normalised to [0, 1].', mono: 'shape: (B, C, H, W)' },
        { icon: '🔲', title: 'Convolution', body: 'A small kernel slides over the input, computing dot products at each position. Produces a feature map that highlights where the kernel\'s pattern exists in the input.', mono: '(f★g)[i,j] = Σ f[m,n]·g[i+m,j+n]' },
        { icon: '⚖️', title: 'Shared weights', body: 'The same kernel is applied at every spatial position. A 3×3 filter has only 9 weights regardless of image size — vs. millions in a fully-connected layer.', mono: 'params = K×K×C_in×C_out' },
        { icon: '📐', title: 'Output size', body: 'Given input H, kernel K, padding P, stride S: output = ⌊(H+2P−K)/S⌋+1. Padding controls shrinkage; stride controls downsampling.', mono: '⌊(H+2P-K)/S⌋ + 1' },
        { icon: '🏊', title: 'Pooling', body: 'Reduces spatial dimensions by taking the max or average over non-overlapping windows. Provides translation invariance and reduces computation.', mono: 'max-pool 2×2, s=2 → H/2, W/2' },
        { icon: '🏛️', title: 'LeNet', body: 'The first successful CNN (LeCun, 1998). Two conv+pool blocks extract features, three FC layers classify. Template for all modern CNNs.', mono: 'conv→pool→conv→pool→fc→fc→out' },
      ]} />

      {/* ── Section 6.1 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-cyan-500/5 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`}>Section 6.1</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Image Data and Motivation</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>How images are represented as tensors and why fully-connected networks are the wrong tool for vision.</p>
      </div>

      {/* 6.1.1 Image representation */}
      <div className={S}>
        <p className={LBL}>6.1.1 — Image Data: Pixels, Channels, and Tensors</p>
        <h2 className={H2}>Every image is a 3D array of numbers</h2>
        <p className={`${BODY} mb-4`}>
          A digital image is stored as a rectangular grid of pixels. Each pixel is a colour value — in RGB images, three integers (Red, Green, Blue) each in [0, 255]. PyTorch normalises these to [0, 1] and stacks them into a (C, H, W) tensor.
        </p>
        <PixelTensorViz theme={theme} />
        <DeepDive title="Greyscale, depth maps and beyond">
          <TheoryBlock title="" cards={[
            { icon: '⬜', title: 'Greyscale (C=1)', body: 'Single intensity channel. MNIST digits: (1, 28, 28). Medical X-rays, depth maps. Half the data of RGB.', mono: '(1, H, W)' },
            { icon: '🌈', title: 'RGB (C=3)', body: 'Standard colour images. ImageNet: (3, 224, 224). Each channel captures one colour component of each pixel.', mono: '(3, H, W)' },
            { icon: '🛰️', title: 'Hyperspectral (C≫3)', body: 'Satellite and medical imaging can have dozens to hundreds of channels (wavelengths). CNNs handle arbitrary C — same conv, more input channels.', mono: '(C, H, W)  C up to 200+' },
          ]} />
        </DeepDive>
      </div>

      {/* 6.1.2 Invariance, locality, shared weights */}
      <div className={S}>
        <p className={LBL}>6.1.2 — Invariance, Locality, and Shared Weights</p>
        <h2 className={H2}>Three properties CNNs exploit that FC layers ignore</h2>
        <p className={`${BODY} mb-4`}>
          Images have strong statistical structure that fully-connected networks ignore. CNNs are designed to exploit three key properties of natural images.
        </p>
        <TheoryBlock title="Inductive biases of CNNs" cards={[
          { icon: '📍', title: 'Locality', body: 'Pixels close together are more related than distant ones. A cat\'s eye is made of nearby pixels, not pixels scattered across the image. Conv filters look at small patches (3×3, 5×5).', mono: 'kernel spans local K×K patch' },
          { icon: '🔄', title: 'Translation equivariance', body: 'A cat in the top-left should activate the same "cat detector" as one in the bottom-right. Using the same kernel everywhere gives equivariance — the feature map shifts with the object.', mono: 'f(T(x)) = T(f(x))' },
          { icon: '♻️', title: 'Shared weights', body: 'The same edge detector is useful everywhere in the image, so the same kernel weights are reused at every position. A 3×3 kernel has 9 parameters regardless of whether the image is 28×28 or 1024×1024.', mono: 'one kernel for all positions' },
        ]} />
        <Callout type="info" title="Parameter count comparison">
          A single 3×3 conv filter on a 28×28 greyscale input: <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>9 parameters</strong>.<br />
          An FC layer connecting 28×28=784 inputs to 784 outputs: <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>615,025 parameters</strong>. Same spatial transformation — 68,000× fewer parameters.
        </Callout>
      </div>

      {/* 6.1.3 Why FC fails */}
      <div className={S}>
        <p className={LBL}>6.1.3 — Why Fully-Connected Layers Fail for Images</p>
        <h2 className={H2}>The parameter explosion problem</h2>
        <p className={`${BODY} mb-4`}>
          Applying a dense fully-connected layer directly to image pixels treats every pixel as independent — ignoring all spatial relationships. Worse, the parameter count explodes with image resolution.
        </p>
        <TheoryBlock title="FC vs CNN parameter counts" cards={[
          { icon: '💥', title: 'Parameter explosion', body: 'A 224×224 RGB image has 150,528 inputs. One FC hidden layer with 4096 neurons: 150,528 × 4096 = 616M parameters — just one layer. Impossible to train without massive data.', mono: '150k × 4k = 616M params 😱' },
          { icon: '🤔', title: 'No spatial bias', body: 'FC treats pixel (0,0) and pixel (223,223) as equally related. A cat\'s whisker and its tail get the same "distance" treatment. All spatial structure must be re-learned from scratch.', mono: 'pixels treated as unordered set' },
          { icon: '📦', title: 'No weight reuse', body: 'Each pixel position gets its own set of weights. An edge detector at position (10,10) learns independently from the same detector at (100,100). No knowledge transfer.', mono: 'N separate weights per position' },
        ]} />
        <Callout type="success" title="CNNs solve all three problems">
          Local connectivity → small kernels see only nearby pixels. Weight sharing → same kernel reused everywhere. Pooling → gradual spatial downsampling compresses information efficiently. Modern CNNs classify ImageNet (1000 classes, 1.28M images) in under an hour on a single GPU.
        </Callout>
      </div>

      {/* ── Section 6.2 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 6.2</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Convolution Operation</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>The discrete convolution from first principles — kernels, feature maps, padding, stride, and 1×1 convolutions.</p>
      </div>

      {/* 6.2.1 + 6.2.2 + 6.2.3 Convolution core */}
      <div className={S}>
        <p className={LBL}>6.2.1–6.2.3 — Convolution, Kernels and Feature Maps</p>
        <h2 className={H2}>What a convolution actually computes</h2>
        <p className={`${BODY} mb-4`}>
          A 2D convolution slides a K×K kernel over the input. At each position, it computes the dot product between the kernel weights and the overlapping patch — producing one value in the output feature map.
        </p>
        <TheoryBlock title="Convolution mechanics" cards={[
          { icon: '🔲', title: 'Discrete 2D convolution', body: 'For each output position (i,j): sum over all kernel positions (m,n) the product of kernel weight K[m,n] and input value X[i+m, j+n].', mono: 'O[i,j] = Σₘ Σₙ K[m,n]·X[i+m,j+n]' },
          { icon: '📡', title: 'Feature map', body: 'The full output of sliding one kernel over the input. High activations = the kernel\'s pattern is present at that location. Low = absent.', mono: 'shape: (H_out, W_out)' },
          { icon: '🔭', title: 'Receptive field', body: 'The region of the original input that influences a single neuron\'s output. With 3×3 kernels: depth 1 → 3×3, depth 2 → 5×5, depth d → (2d+1)×(2d+1).', mono: 'RF = 2·depth + 1  (for 3×3 convs)' },
        ]} />
        <Callout type="warning" title="Convolution vs cross-correlation">
          Technically, deep learning libraries compute <em>cross-correlation</em> (no kernel flip), not true mathematical convolution. The difference doesn't matter in practice — kernels are learned anyway, so flipping is just a convention.
        </Callout>
        <p className={`${BODY} mb-4`}>Select a kernel type, adjust padding and stride, and click output cells to highlight their receptive field in the input:</p>
        <ConvDemo theme={theme} />
      </div>

      {/* 6.2.3 Receptive field growth */}
      <div className={S}>
        <p className={LBL}>6.2.3 — Receptive Field Growth with Depth</p>
        <h2 className={H2}>Deeper networks see larger regions of the input</h2>
        <p className={`${BODY} mb-4`}>
          Stack more 3×3 conv layers and each output neuron "sees" a progressively larger patch of the original image. Two 3×3 layers have the same receptive field as one 5×5 layer — but fewer parameters and an extra non-linearity.
        </p>
        <ReceptiveFieldViz theme={theme} />
        <Callout type="info" title="Why use small kernels?">
          VGG (2014) showed that stacking 3×3 convolutions outperforms using larger kernels. Three 3×3 layers (receptive field = 7×7) have 3×9 = 27 parameters vs one 7×7 layer's 49. Smaller kernels are also more efficient and allow more non-linearities.
        </Callout>
      </div>

      {/* 6.2.4 + 6.2.5 Padding and stride */}
      <div className={S}>
        <p className={LBL}>6.2.4–6.2.5 — Padding (Valid vs Same) and Stride</p>
        <h2 className={H2}>Controlling output spatial dimensions</h2>
        <p className={`${BODY} mb-4`}>
          Two hyperparameters control output size: <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>padding</strong> (adding zeros around the border) and <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>stride</strong> (step size of the sliding kernel).
        </p>
        <TheoryBlock title="Padding and stride effects" cards={[
          { icon: '0️⃣', title: 'Valid (P=0)', body: 'No padding. Output shrinks by K−1 pixels in each dimension. Corners get seen less — information at borders is underrepresented.', mono: 'H_out = H − K + 1' },
          { icon: '⬜', title: 'Same (P=⌊K/2⌋)', body: 'Pad with ⌊K/2⌋ zeros on each side. With stride=1, output H×W = input H×W. Preserves spatial size through the layer.', mono: 'H_out = H  (s=1)' },
          { icon: '→→', title: 'Stride S', body: 'Move kernel S pixels per step instead of 1. Larger stride = smaller output. Stride 2 ≈ halves each dimension, replacing pooling in some architectures.', mono: 'H_out = ⌊(H+2P-K)/S⌋ + 1' },
        ]} />
        <Callout type="formula" mono="H_out = ⌊(H + 2·P − K) / S⌋ + 1">
          Memorise this. Given H=224, K=3, P=1, S=1: (224+2−3)/1+1 = 224 (same). With S=2: (224+2−3)/2+1 = 112 (halved). Strided convolutions replace max pooling in modern architectures like ResNet.
        </Callout>
      </div>

      {/* 6.2.6 1×1 conv */}
      <div className={S}>
        <p className={LBL}>6.2.6 — 1×1 Convolution and Its Purpose</p>
        <h2 className={H2}>Channel mixing without spatial aggregation</h2>
        <p className={`${BODY} mb-4`}>
          A 1×1 convolution has a kernel of size 1×1 — it looks at a single pixel across all C channels and produces a new channel through a learned linear combination. It doesn't aggregate spatial information but it reshapes the channel dimension.
        </p>
        <TheoryBlock title="1×1 convolution uses" cards={[
          { icon: '📉', title: 'Dimensionality reduction', body: 'Reduce 256 channels to 64 using a 1×1 conv. This "bottleneck" drastically cuts computation in the next layer. Used heavily in Inception and ResNet.', mono: '(B,256,H,W) → (B,64,H,W)' },
          { icon: '🧠', title: 'Non-linear channel mixing', body: 'After the linear 1×1 combination, apply ReLU. This allows learning non-linear relationships between channels — more expressive than just pooling or reshaping.', mono: '1×1 conv + ReLU = channel-wise MLP' },
          { icon: '📐', title: 'Adjusting channel count', body: 'Any time you need a different number of channels (up or down) without changing H×W, use a 1×1 conv. Also used in skip connections when dimensions mismatch.', mono: 'change C without touching H,W' },
        ]} />
        <Callout type="analogy" title="Analogy: portfolio rebalancing">
          Think of channels as asset classes. A 1×1 conv is a portfolio rebalancer — it creates new "assets" (output channels) as weighted combinations of the existing ones. It changes allocation but doesn't aggregate over time (spatial positions).
        </Callout>
      </div>

      {/* ── Section 6.3 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>Section 6.3</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Pooling, Multiple Channels and LeNet</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Max/average pooling, multi-channel convolutions, and the LeNet architecture that started it all.</p>
      </div>

      {/* 6.3.1 Pooling */}
      <div className={S}>
        <p className={LBL}>6.3.1 — Max Pooling and Average Pooling</p>
        <h2 className={H2}>Spatial downsampling with no learned parameters</h2>
        <p className={`${BODY} mb-4`}>
          Pooling layers reduce spatial dimensions by summarising each local patch with a single value. Unlike convolutions, pooling has no learnable weights — it applies a fixed operation (max or mean) over a window.
        </p>
        <TheoryBlock title="Pooling variants" cards={[
          { icon: '🏆', title: 'Max pooling', body: 'Takes the maximum value in each K×K window. Keeps the strongest activation. Provides strong translation invariance — the feature fires regardless of where exactly in the window the pattern is.', mono: 'max(patch) → most prominent feature' },
          { icon: '📊', title: 'Average pooling', body: 'Takes the mean of each K×K window. Smoother downsampling, keeps all signal. Used in older architectures (LeNet) and in global average pooling before the classifier.', mono: 'mean(patch) → smooth feature' },
          { icon: '🌍', title: 'Global average pooling', body: 'Reduces each entire feature map to a single value (the spatial average). Replaces the "flatten+FC" combination, drastically reducing parameters and improving generalisation.', mono: '(B,C,H,W) → (B,C) → classifier' },
        ]} />
        <p className={`${BODY} mb-4`}>Hover the output cells to highlight which 2×2 patch they summarise. Toggle between max and average pooling:</p>
        <PoolingDemo theme={theme} />
        <DeepDive title="Why max pooling over average pooling?">
          <p className={`text-sm ${BODY} mb-2`}>For feature detection tasks, whether an edge exists matters more than its average intensity. Max pooling keeps the strongest signal and discards weak activations — better for detecting whether a pattern is present anywhere in the patch.</p>
          <p className={`text-sm ${BODY}`}>Average pooling is better when you care about the overall texture or density (e.g. how busy a region is). Global average pooling is now standard before the final FC layer — it averages each feature map to a single number, acting as a spatial bag-of-features.</p>
        </DeepDive>
      </div>

      {/* 6.3.2 Multiple channels */}
      <div className={S}>
        <p className={LBL}>6.3.2 — Multiple Input and Output Channels</p>
        <h2 className={H2}>The full 4D convolution tensor</h2>
        <p className={`${BODY} mb-4`}>
          Real conv layers have C_in input channels and C_out output channels. The weight tensor is 4D: (C_out, C_in, K, K). Each of the C_out output feature maps is computed by convolving <em>all</em> C_in input channels with one 3D filter, then summing.
        </p>
        <TheoryBlock title="Multi-channel convolution" cards={[
          { icon: '🎞️', title: 'Input channels (C_in)', body: 'The number of feature maps coming in. First layer: C_in = RGB channels (3). Deeper layers: C_in = number of filters in the previous layer (e.g. 64, 128, 256).', mono: 'input: (B, C_in, H, W)' },
          { icon: '🔧', title: 'Output channels (C_out)', body: 'Number of different filters to learn. Each filter detects a different pattern. More filters = richer feature representation = more parameters.', mono: 'output: (B, C_out, H_out, W_out)' },
          { icon: '📦', title: 'Parameter count', body: 'One conv layer: C_out × C_in × K × K weights + C_out biases. Example: 64 filters, 32 input channels, 3×3 kernel: 64×32×9+64 = 18,496 parameters.', mono: 'C_out × C_in × K² + C_out' },
        ]} />
        <Callout type="formula" mono="output[n, c_out, i, j] = Σ_{c_in} (input[n,c_in] ★ W[c_out, c_in]) + b[c_out]">
          For batch item n, output channel c_out, position (i,j): convolve all C_in input channels with the corresponding 2D slice of the weight tensor, sum the results, add bias. This gives one value per (n, c_out, i, j).
        </Callout>
        <DeepDive title="How channels grow through a CNN">
          <p className={`text-sm ${BODY} mb-2`}>A typical pattern: channels grow as spatial dimensions shrink. Early layers: 3→64 channels at 224×224. Middle layers: 128→256 at 56×56. Deep layers: 512→512 at 14×14. Each layer trades spatial resolution for richer channel representations.</p>
          <Callout type="info">Total feature volume (C×H×W) stays roughly constant through a well-designed CNN. Halving H and W while doubling C keeps the same number of values per image.</Callout>
        </DeepDive>
      </div>

      {/* 6.3.3 LeNet */}
      <div className={S}>
        <p className={LBL}>6.3.3 — LeNet Architecture: Putting It All Together</p>
        <h2 className={H2}>The blueprint for all modern CNNs</h2>
        <p className={`${BODY} mb-4`}>
          LeNet-5 (LeCun et al., 1998) was the first CNN trained end-to-end with backpropagation on real tasks. It introduced the alternating conv+pool structure that every subsequent CNN — VGG, ResNet, EfficientNet — still follows at its core.
        </p>
        <p className={`${BODY} mb-4`}>Click any layer block to inspect its shape and role:</p>
        <LeNetDiagram theme={theme} />
        <TheoryBlock title="LeNet design principles" cards={[
          { icon: '🔺', title: 'Spatial shrinkage', body: 'Feature maps shrink (28→14→5) as you go deeper via pooling. Each pool halves H and W, reducing computation in subsequent layers.', mono: '28 → 14 → 5' },
          { icon: '🔼', title: 'Channel expansion', body: 'While spatial dims shrink, channels grow (1→6→16). More channels = richer feature vocabulary. Classic CNN trade-off: spatial resolution for representational depth.', mono: '1 → 6 → 16' },
          { icon: '🔀', title: 'Global flattening', body: 'After the last pool (16×5×5=400 values), flatten to a vector and pass through FC layers. FC layers combine all spatial features globally to make the final class decision.', mono: '400 → 120 → 84 → 10' },
        ]} />
        <Callout type="success" title="LeNet's lasting legacy">
          LeNet proved that raw pixel values could be fed directly into a network that learns its own features — no handcrafted feature engineering. This insight, plus the conv+pool architecture, is the foundation of all modern computer vision.
        </Callout>
        <DeepDive title="From LeNet to modern CNNs">
          <TheoryBlock title="" cards={[
            { icon: '🏔️', title: 'AlexNet (2012)', body: 'Scaled LeNet to 8 layers, added ReLU, dropout, and GPU training. Won ImageNet by a massive margin. Started the deep learning revolution.', mono: '5 conv + 3 FC, ReLU, dropout' },
            { icon: '🧱', title: 'VGG (2014)', body: 'Very deep CNNs (16–19 layers) using only 3×3 filters. Showed that depth alone improves performance. Still widely used as a feature extractor.', mono: '16–19 layers, all 3×3 kernels' },
            { icon: '🔗', title: 'ResNet (2015)', body: 'Introduced residual (skip) connections to allow training of 100–1000+ layer networks. Solved vanishing gradients in deep nets.', mono: 'x + F(x) — identity shortcuts' },
          ]} />
        </DeepDive>
      </div>

      {/* Code blocks */}
      <div className={S}>
        <p className={LBL}>Python — Convolution operations (TensorFlow)</p>
        <h2 className={H2}>Manual 2D conv with padding, stride, and size formula</h2>
        <p className={`${BODY} mb-3`}>Demonstrates valid and same padding, stride effects, and the output size formula using TensorFlow's <span className={`font-mono text-xs ${theme === 'dark' ? 'text-cyan-300' : 'text-cyan-600'}`}>tf.nn.conv2d</span>.</p>
        <CodeBlock code={PYTHON_CODE_CONV} />
      </div>

      <div className={S}>
        <p className={LBL}>Python — LeNet on MNIST (TensorFlow / Keras)</p>
        <h2 className={H2}>Full LeNet implementation and training loop</h2>
        <p className={`${BODY} mb-3`}>Complete LeNet-5 in Keras: <span className={`font-mono text-xs ${theme === 'dark' ? 'text-cyan-300' : 'text-cyan-600'}`}>Conv2D + AveragePooling2D + Dense</span>, built-in MNIST loader, and Adam training with validation accuracy.</p>
        <CodeBlock code={PYTHON_CODE_LENET} />
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme === 'dark' ? 'bg-cyan-500/10 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>10 questions covering all three sections • +100 XP on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="cnn" />
      </div>
    </motion.div>
  )
}
