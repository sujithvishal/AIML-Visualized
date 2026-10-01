import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ── Python code blocks ─────────────────────────────────────────────────────────

const CODE_VGG = `import tensorflow as tf

def vgg_block(num_convs, num_filters):
    """One VGG block: num_convs × (Conv 3×3 + ReLU) → MaxPool 2×2."""
    block = tf.keras.Sequential()
    for _ in range(num_convs):
        block.add(tf.keras.layers.Conv2D(
            num_filters, kernel_size=3, padding='same', activation='relu'))
    block.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
    return block

# VGG-16 architecture spec: (num_convs, num_filters) per block
vgg16_spec = [(2,64), (2,128), (3,256), (3,512), (3,512)]

def build_vgg(spec, num_classes=10):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    for num_convs, num_filters in spec:
        model.add(vgg_block(num_convs, num_filters))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

vgg16 = build_vgg(vgg16_spec)
vgg16.summary()
print(f"Parameters: {vgg16.count_params():,}")
# VGG-16 ≈ 138M parameters

# Use pre-trained VGG16 from tf.keras.applications
pretrained = tf.keras.applications.VGG16(
    weights='imagenet', include_top=False, input_shape=(224,224,3))
print("Pretrained VGG16 loaded")`

const CODE_NIN = `import tensorflow as tf

def nin_block(num_filters, kernel_size, strides, padding):
    """NiN block: Conv → 1×1 Conv (ReLU) → 1×1 Conv (ReLU)."""
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               activation='relu'),
        # Two 1×1 convolutions act as a per-pixel MLP across channels
        tf.keras.layers.Conv2D(num_filters, kernel_size=1, activation='relu'),
        tf.keras.layers.Conv2D(num_filters, kernel_size=1, activation='relu'),
    ])

def build_nin(num_classes=10):
    return tf.keras.Sequential([
        nin_block(96,  kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5,  strides=1, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3,  strides=1, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        nin_block(num_classes, kernel_size=3, strides=1, padding='same'),
        # Global average pooling replaces FC — reduces params dramatically
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Softmax(),
    ])

nin = build_nin()
nin.build((None, 224, 224, 3))
nin.summary()`

const CODE_INCEPTION = `import tensorflow as tf

def inception_block(f1, f3_r, f3, f5_r, f5, fpool):
    """
    GoogLeNet Inception block with 4 parallel branches:
      - 1×1 conv  (f1 filters)
      - 1×1 → 3×3 conv  (f3_r reduce, f3 out)
      - 1×1 → 5×5 conv  (f5_r reduce, f5 out)
      - 3×3 max-pool → 1×1 conv  (fpool out)
    All branches use same padding so outputs can be concatenated.
    """
    def call(x):
        # Branch 1: 1×1
        b1 = tf.keras.layers.Conv2D(f1,   1, padding='same', activation='relu')(x)

        # Branch 2: 1×1 → 3×3
        b2 = tf.keras.layers.Conv2D(f3_r, 1, padding='same', activation='relu')(x)
        b2 = tf.keras.layers.Conv2D(f3,   3, padding='same', activation='relu')(b2)

        # Branch 3: 1×1 → 5×5
        b3 = tf.keras.layers.Conv2D(f5_r, 1, padding='same', activation='relu')(x)
        b3 = tf.keras.layers.Conv2D(f5,   5, padding='same', activation='relu')(b3)

        # Branch 4: MaxPool → 1×1
        b4 = tf.keras.layers.MaxPooling2D(3, strides=1, padding='same')(x)
        b4 = tf.keras.layers.Conv2D(fpool, 1, padding='same', activation='relu')(b4)

        # Concatenate along channel axis
        return tf.keras.layers.Concatenate()([b1, b2, b3, b4])
    return call

# Use one inception block
inp = tf.keras.Input((28, 28, 192))
out = inception_block(64, 96, 128, 16, 32, 32)(inp)
model = tf.keras.Model(inp, out)
print("Output shape:", model.output_shape)
# → (None, 28, 28, 256)  (64+128+32+32)

# Or use the pretrained GoogLeNet/InceptionV3 from Keras
inception_v3 = tf.keras.applications.InceptionV3(
    weights='imagenet', include_top=False, input_shape=(299,299,3))`

const CODE_RESNET = `import tensorflow as tf

def residual_block(filters, downsample=False):
    """
    ResNet basic residual block (used in ResNet-18 / ResNet-34).
    Two 3×3 convolutions + identity (or projection) skip connection.
    """
    stride = 2 if downsample else 1

    def call(x):
        shortcut = x

        # Main path
        out = tf.keras.layers.Conv2D(filters, 3, strides=stride,
                                     padding='same', use_bias=False)(x)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Conv2D(filters, 3, strides=1,
                                     padding='same', use_bias=False)(out)
        out = tf.keras.layers.BatchNormalization()(out)

        # Projection shortcut: match dimensions when downsampling
        if downsample or x.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(
                filters, 1, strides=stride, use_bias=False)(x)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        # Skip connection: add shortcut to output BEFORE activation
        out = tf.keras.layers.Add()([out, shortcut])
        out = tf.keras.layers.ReLU()(out)
        return out

    return call

# Build a small ResNet-like model
inp = tf.keras.Input((32, 32, 3))
x   = tf.keras.layers.Conv2D(64, 3, padding='same')(inp)
x   = residual_block(64)(x)
x   = residual_block(128, downsample=True)(x)
x   = residual_block(256, downsample=True)(x)
x   = tf.keras.layers.GlobalAveragePooling2D()(x)
out = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inp, out)
model.summary()

# Or load pretrained ResNet50
resnet50 = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False, input_shape=(224,224,3))`

const QUIZ_QUESTIONS = [
  { question: 'VGG networks are characterised by:', options: ['Large 11×11 and 7×7 kernels', 'Exclusively 3×3 convolutions stacked in blocks', 'Inception modules', 'Skip connections'], correct: 1, explanation: 'VGG (2014) showed that stacking small 3×3 conv filters works better than large kernels. A stack of two 3×3 convs has the same receptive field as one 5×5, with fewer parameters and an extra non-linearity.' },
  { question: 'How many convolutional layers does VGG-16 have?', options: ['8', '11', '13', '16'], correct: 2, explanation: 'VGG-16 has 13 convolutional layers (5 blocks: 2+2+3+3+3) plus 3 fully-connected layers = 16 weight layers total.' },
  { question: 'NiN (Network in Network) replaces fully-connected layers with:', options: ['Max pooling', 'Global Average Pooling after 1×1 convolutions', 'Dropout only', 'Batch normalisation'], correct: 1, explanation: 'NiN uses 1×1 convolutions to implement a per-pixel MLP across channels, then Global Average Pooling instead of dense layers — eliminating millions of FC parameters.' },
  { question: 'A 1×1 convolution in a NiN block acts as:', options: ['Spatial pooling', 'A fully-connected layer applied independently at each pixel', 'Batch normalisation', 'A residual shortcut'], correct: 1, explanation: '1×1 conv applies a learned linear combination (+ activation) across all C channels at every spatial position independently — exactly a per-pixel FC layer.' },
  { question: 'An Inception block uses:', options: ['Sequential convolutions only', 'Multiple filter sizes in parallel, outputs concatenated', 'Skip connections', 'Global pooling only'], correct: 1, explanation: 'GoogLeNet\'s Inception module applies 1×1, 3×3, and 5×5 convolutions (plus pooling) in parallel on the same input and concatenates the results along the channel dimension.' },
  { question: 'The 1×1 "bottleneck" convolutions in Inception blocks serve to:', options: ['Increase spatial resolution', 'Reduce the number of input channels before expensive 3×3 and 5×5 convolutions', 'Add skip connections', 'Replace pooling'], correct: 1, explanation: 'A 1×1 conv before a 3×3 or 5×5 conv reduces channel depth (e.g. 192→32), drastically cutting the FLOPs of the subsequent expensive convolution.' },
  { question: 'A residual block computes:', options: ['F(x) only', 'x + F(x) — adding the input back to the transformed output', 'F(x) × x', 'max(x, F(x))'], correct: 1, explanation: 'The residual block output is H(x) = F(x) + x, where x is the skip connection. This means F(x) only needs to learn the "residual" — the difference from the identity.' },
  { question: 'Residual connections solve the vanishing gradient problem because:', options: ['They add more layers', 'Gradients can flow directly back through the skip path without passing through weight layers', 'They use larger learning rates', 'They apply batch normalisation'], correct: 1, explanation: '∂L/∂x = ∂L/∂H · (∂F/∂x + 1). The +1 ensures the gradient through the skip path is always at least as large as the upstream gradient — preventing exponential decay.' },
  { question: 'ResNet-50 uses bottleneck blocks with filter sizes:', options: ['3×3 only', '1×1 → 3×3 → 1×1 (expand, convolve, reduce)', '5×5 → 3×3', '1×1 only'], correct: 1, explanation: 'ResNet-50+ uses "bottleneck" blocks: 1×1 (reduce channels) → 3×3 (spatial conv) → 1×1 (restore channels). This is more efficient than two 3×3 convs for deeper networks.' },
  { question: 'Which is NOT true about ResNet variants?', options: ['ResNet-18 uses basic (3×3, 3×3) blocks', 'ResNet-101 is deeper than ResNet-50', 'All ResNets have the same accuracy regardless of depth', 'ResNet-50 uses bottleneck blocks'], correct: 2, explanation: 'Deeper ResNets generally achieve higher accuracy on ImageNet (e.g. ResNet-101 > ResNet-50 > ResNet-18), though with diminishing returns and greater compute cost.' },
]

// ── VGG block diagram ──────────────────────────────────────────────────────────
function VGGDiagram({ theme }) {
  const [variant, setVariant] = useState('vgg16')

  const configs = {
    vgg16: {
      label: 'VGG-16',
      blocks: [
        { convs: 2, filters: 64,  label: 'Block 1', size: '224→112' },
        { convs: 2, filters: 128, label: 'Block 2', size: '112→56' },
        { convs: 3, filters: 256, label: 'Block 3', size: '56→28' },
        { convs: 3, filters: 512, label: 'Block 4', size: '28→14' },
        { convs: 3, filters: 512, label: 'Block 5', size: '14→7' },
      ],
      fc: [4096, 4096, 1000],
      params: '138M',
    },
    vgg19: {
      label: 'VGG-19',
      blocks: [
        { convs: 2, filters: 64,  label: 'Block 1', size: '224→112' },
        { convs: 2, filters: 128, label: 'Block 2', size: '112→56' },
        { convs: 4, filters: 256, label: 'Block 3', size: '56→28' },
        { convs: 4, filters: 512, label: 'Block 4', size: '28→14' },
        { convs: 4, filters: 512, label: 'Block 5', size: '14→7' },
      ],
      fc: [4096, 4096, 1000],
      params: '144M',
    },
  }

  const cfg = configs[variant]
  const blockColors = ['#6366F1', '#8B5CF6', '#06B6D4', '#10B981', '#F59E0B']
  const [active, setActive] = useState(null)

  return (
    <div>
      <div className="flex gap-2 mb-4">
        {Object.entries(configs).map(([k, v]) => (
          <button key={k} onClick={() => setActive(null) || setVariant(k)}
            className={`flex-1 py-2 rounded-lg text-xs font-bold border transition-all ${variant === k
              ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
              : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{v.label}</button>
        ))}
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="flex items-center gap-1 min-w-max mx-auto w-fit">
          {/* Input */}
          <div className="flex flex-col items-center gap-1">
            <div className={`w-12 h-12 rounded-lg border-2 flex items-center justify-center text-xs font-bold ${theme === 'dark' ? 'bg-slate-800 border-white/20 text-gray-400' : 'bg-gray-100 border-gray-300 text-gray-500'}`}>
              3
            </div>
            <span className={`text-[9px] ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>Input</span>
            <span className={`text-[8px] font-mono ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>224²</span>
          </div>

          {cfg.blocks.map((blk, i) => (
            <div key={i} className="flex items-center">
              <div className={`w-3 h-px ${theme === 'dark' ? 'bg-white/20' : 'bg-gray-300'}`} />
              <button onClick={() => setActive(active === i ? null : i)} className="flex flex-col items-center gap-1">
                <div className="flex gap-0.5">
                  {Array.from({ length: blk.convs }).map((_, j) => (
                    <div key={j} className="w-5 rounded border flex items-center justify-center"
                      style={{
                        height: Math.min(16 + blk.filters / 16, 52),
                        backgroundColor: blockColors[i] + (active === i ? 'CC' : '33'),
                        borderColor: blockColors[i],
                      }}>
                    </div>
                  ))}
                </div>
                <div className={`w-full h-3 rounded border-2 mt-0.5`}
                  style={{ backgroundColor: blockColors[i] + '44', borderColor: blockColors[i] }} />
                <span className={`text-[9px] font-semibold`} style={{ color: blockColors[i] }}>{blk.label}</span>
                <span className={`text-[8px] font-mono ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>{blk.filters}ch</span>
              </button>
            </div>
          ))}

          {/* FC layers */}
          <div className={`w-3 h-px ${theme === 'dark' ? 'bg-white/20' : 'bg-gray-300'}`} />
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-0.5 items-end">
              {cfg.fc.map((n, i) => (
                <div key={i} className={`w-5 rounded border flex items-center justify-center`}
                  style={{
                    height: Math.min(10 + n / 200, 40),
                    backgroundColor: '#EF4444' + '33',
                    borderColor: '#EF4444',
                  }} />
              ))}
            </div>
            <span className={`text-[9px] font-semibold text-red-400`}>FC</span>
            <span className={`text-[8px] font-mono ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>4k→4k→1k</span>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {active !== null && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
            className={`mt-3 rounded-xl border p-3 overflow-hidden ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`text-xs font-bold mb-1`} style={{ color: blockColors[active] }}>
              {cfg.blocks[active].label} — {cfg.blocks[active].convs}× Conv2D(3×3, {cfg.blocks[active].filters}) + MaxPool 2×2
            </p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              Spatial: {cfg.blocks[active].size} &nbsp;·&nbsp;
              Params per conv: 3×3×{active === 0 ? 3 : cfg.blocks[active-1]?.filters}×{cfg.blocks[active].filters} ≈&nbsp;
              {(9 * (active === 0 ? 3 : cfg.blocks[active-1]?.filters) * cfg.blocks[active].filters).toLocaleString()}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
        {[
          { label: 'Conv layers', val: cfg.blocks.reduce((s, b) => s + b.convs, 0), color: 'text-indigo-400' },
          { label: 'FC layers', val: 3, color: 'text-red-400' },
          { label: 'Parameters', val: cfg.params, color: 'text-amber-400' },
        ].map(m => (
          <div key={m.label} className={`rounded-xl p-2 border text-center ${theme === 'dark' ? 'bg-slate-800 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`font-bold text-base ${m.color}`}>{m.val}</p>
            <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.label}</p>
          </div>
        ))}
      </div>
      <p className={`text-xs mt-2 text-center ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
        Click any block to inspect it · Vertical bar height = filter count · MaxPool shown as coloured band
      </p>
    </div>
  )
}

// ── NiN block diagram ──────────────────────────────────────────────────────────
function NiNDiagram({ theme }) {
  const steps = [
    { label: 'Conv K×K', sub: 'Spatial feature', color: '#6366F1', icon: '🔲' },
    { label: '1×1 Conv + ReLU', sub: 'Channel mix #1', color: '#8B5CF6', icon: '⚡' },
    { label: '1×1 Conv + ReLU', sub: 'Channel mix #2', color: '#06B6D4', icon: '⚡' },
  ]
  return (
    <div>
      <div className={`rounded-xl border p-4 mb-4 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
        <p className={`text-xs font-semibold mb-3 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>One NiN Block</p>
        <div className="flex items-center gap-2 flex-wrap">
          {steps.map((s, i) => (
            <div key={i} className="flex items-center gap-2">
              <div className="flex flex-col items-center">
                <div className={`px-3 py-2 rounded-xl border text-center`}
                  style={{ borderColor: s.color, backgroundColor: s.color + '22' }}>
                  <span className="text-base">{s.icon}</span>
                  <p className="text-xs font-semibold mt-0.5" style={{ color: s.color }}>{s.label}</p>
                  <p className={`text-[10px] ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{s.sub}</p>
                </div>
              </div>
              {i < steps.length - 1 && <span className={`text-lg ${theme === 'dark' ? 'text-gray-600' : 'text-gray-300'}`}>→</span>}
            </div>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {[
          { title: 'Standard block', desc: 'Conv(K×K) → ReLU → MaxPool. Limited to one linear combination per spatial position.', color: '#6B7280' },
          { title: 'NiN block', desc: 'Conv(K×K) → 1×1→ReLU → 1×1→ReLU. Per-pixel MLP across channels — richer non-linear representation.', color: '#6366F1' },
        ].map(c => (
          <div key={c.title} className={`rounded-xl border p-3 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className="text-xs font-semibold mb-1" style={{ color: c.color }}>{c.title}</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{c.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Inception block visualiser ─────────────────────────────────────────────────
function InceptionViz({ theme }) {
  const [filterSet, setFilterSet] = useState(0)

  const filterConfigs = [
    { label: 'Inception 3a', f1: 64, f3r: 96, f3: 128, f5r: 16, f5: 32, fp: 32 },
    { label: 'Inception 3b', f1: 128, f3r: 128, f3: 192, f5r: 32, f5: 96, fp: 64 },
    { label: 'Inception 4a', f1: 192, f3r: 96, f3: 208, f5r: 16, f5: 48, fp: 64 },
  ]

  const cfg = filterConfigs[filterSet]
  const total = cfg.f1 + cfg.f3 + cfg.f5 + cfg.fp

  const branches = [
    { label: '1×1 Conv', filters: cfg.f1, color: '#6366F1', desc: 'Captures local 1-pixel patterns, reduces channels cheaply.' },
    { label: '1×1 → 3×3', filters: cfg.f3, reduce: cfg.f3r, color: '#8B5CF6', desc: `1×1 reduces ${192}→${cfg.f3r}, then 3×3 captures local spatial patterns.` },
    { label: '1×1 → 5×5', filters: cfg.f5, reduce: cfg.f5r, color: '#06B6D4', desc: `1×1 reduces to ${cfg.f5r}, then 5×5 captures larger spatial context.` },
    { label: '3×3 Pool → 1×1', filters: cfg.fp, color: '#10B981', desc: 'MaxPool preserves spatial info; 1×1 reduces output channels.' },
  ]

  return (
    <div>
      <div className="flex gap-2 mb-4">
        {filterConfigs.map((fc, i) => (
          <button key={i} onClick={() => setFilterSet(i)}
            className={`flex-1 py-1.5 rounded-lg text-xs font-semibold border transition-all ${filterSet === i
              ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
              : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{fc.label}</button>
        ))}
      </div>

      {/* Branch visualisation */}
      <div className={`rounded-xl border p-3 mb-3 ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
        {/* Input bar */}
        <div className="flex items-center gap-2 mb-3">
          <span className={`text-xs font-mono w-16 text-right ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>input</span>
          <div className={`flex-1 h-6 rounded-lg border-2 ${theme === 'dark' ? 'bg-slate-700 border-white/20' : 'bg-gray-200 border-gray-300'}`} />
        </div>

        {/* Four branches */}
        {branches.map((br, i) => (
          <div key={i} className="flex items-center gap-2 mb-2">
            <span className={`text-[10px] font-mono w-16 text-right ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{br.label}</span>
            <motion.div className="h-5 rounded-lg border"
              animate={{ width: `${(br.filters / total) * 100}%` }}
              transition={{ duration: 0.4 }}
              style={{ backgroundColor: br.color + '55', borderColor: br.color, minWidth: 24 }}>
            </motion.div>
            <span className="text-[10px] font-mono" style={{ color: br.color }}>{br.filters}</span>
          </div>
        ))}

        {/* Concat bar */}
        <div className="flex items-center gap-2 mt-3">
          <span className={`text-xs font-mono w-16 text-right text-emerald-400`}>concat</span>
          <div className="flex-1 h-6 rounded-lg flex overflow-hidden border border-emerald-500/30">
            {branches.map((br, i) => (
              <motion.div key={i} className="h-full"
                animate={{ width: `${(br.filters / total) * 100}%` }}
                transition={{ duration: 0.4 }}
                style={{ backgroundColor: br.color + 'CC' }} />
            ))}
          </div>
          <span className="text-[10px] font-mono text-emerald-400">{total}</span>
        </div>
      </div>

      {/* Branch detail */}
      <div className="grid grid-cols-2 gap-2">
        {branches.map((br, i) => (
          <div key={i} className={`rounded-xl border p-2.5 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className="text-xs font-semibold mb-0.5" style={{ color: br.color }}>{br.label} → {br.filters}</p>
            <p className={`text-[10px] leading-relaxed ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{br.desc}</p>
          </div>
        ))}
      </div>
      <p className={`text-xs mt-2 text-center font-mono ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
        Output channels: {cfg.f1}+{cfg.f3}+{cfg.f5}+{cfg.fp} = <span className="text-emerald-400 font-bold">{total}</span>
      </p>
    </div>
  )
}

// ── Residual block visualiser ──────────────────────────────────────────────────
function ResidualViz({ theme }) {
  const [blockType, setBlockType] = useState('basic')
  const [showGrad, setShowGrad] = useState(false)

  const basic = [
    { label: 'Conv 3×3', color: '#6366F1' },
    { label: 'BatchNorm', color: '#8B5CF6' },
    { label: 'ReLU', color: '#06B6D4' },
    { label: 'Conv 3×3', color: '#6366F1' },
    { label: 'BatchNorm', color: '#8B5CF6' },
  ]

  const bottleneck = [
    { label: 'Conv 1×1 (reduce)', color: '#6366F1' },
    { label: 'BatchNorm + ReLU', color: '#8B5CF6' },
    { label: 'Conv 3×3', color: '#06B6D4' },
    { label: 'BatchNorm + ReLU', color: '#8B5CF6' },
    { label: 'Conv 1×1 (restore)', color: '#6366F1' },
    { label: 'BatchNorm', color: '#8B5CF6' },
  ]

  const layers = blockType === 'basic' ? basic : bottleneck
  const skipColor = '#10B981'

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="flex gap-2">
          {['basic', 'bottleneck'].map(t => (
            <button key={t} onClick={() => setBlockType(t)}
              className={`flex-1 py-1.5 rounded-lg text-xs font-semibold border capitalize transition-all ${blockType === t
                ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
                : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{t}</button>
          ))}
        </div>
        <button onClick={() => setShowGrad(g => !g)}
          className={`py-1.5 rounded-lg text-xs font-semibold border transition-all ${showGrad
            ? theme === 'dark' ? 'bg-amber-500/20 border-amber-500/40 text-amber-300' : 'bg-amber-100 border-amber-300 text-amber-700'
            : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>
          {showGrad ? '▲ Gradient flow ON' : '▲ Show gradient flow'}
        </button>
      </div>

      <div className="flex gap-6 justify-center">
        {/* Main path */}
        <div className="flex flex-col items-center gap-0">
          {/* Input */}
          <div className={`px-4 py-2 rounded-xl border-2 text-xs font-bold ${theme === 'dark' ? 'bg-slate-700 border-white/20 text-white' : 'bg-gray-200 border-gray-400 text-gray-900'}`}>
            x (input)
          </div>

          {/* Skip connection line */}
          <div className="relative w-full flex justify-center">
            <div className={`absolute left-1/2 top-0 bottom-0 w-0.5 ${theme === 'dark' ? 'bg-white/10' : 'bg-gray-200'}`} style={{ transform: 'translateX(-50%)' }} />

            {/* Main path layers */}
            <div className="flex flex-col items-center gap-1.5 py-2 relative z-10">
              {layers.map((l, i) => (
                <div key={i} className="flex flex-col items-center gap-1">
                  <div className={`w-2 h-2 rounded-full`} style={{ backgroundColor: l.color + '80' }} />
                  <div className={`px-3 py-1.5 rounded-lg border text-center min-w-[140px]`}
                    style={{ backgroundColor: l.color + '22', borderColor: l.color }}>
                    <span className="text-xs font-medium" style={{ color: l.color }}>{l.label}</span>
                  </div>
                </div>
              ))}
              <div className={`w-2 h-2 rounded-full`} style={{ backgroundColor: skipColor + '80' }} />
            </div>
          </div>

          {/* Add node */}
          <div className={`w-10 h-10 rounded-full border-2 flex items-center justify-center text-base font-bold`}
            style={{ borderColor: skipColor, backgroundColor: skipColor + '22', color: skipColor }}>
            +
          </div>
          <div className={`w-2 h-3 ${theme === 'dark' ? 'bg-white/20' : 'bg-gray-300'}`} />

          {/* ReLU after add */}
          <div className={`px-3 py-1.5 rounded-lg border text-xs font-medium`}
            style={{ borderColor: '#06B6D4', backgroundColor: '#06B6D422', color: '#06B6D4' }}>
            ReLU
          </div>
          <div className={`w-2 h-2 ${theme === 'dark' ? 'bg-white/20' : 'bg-gray-300'}`} />

          {/* Output */}
          <div className={`px-4 py-2 rounded-xl border-2 text-xs font-bold`}
            style={{ borderColor: skipColor, backgroundColor: skipColor + '22', color: skipColor }}>
            x + F(x)
          </div>
        </div>

        {/* Skip path label */}
        <div className="flex flex-col justify-center gap-2 text-xs">
          <div className={`rounded-xl border p-3 max-w-[140px] ${theme === 'dark' ? 'bg-slate-800/60 border-emerald-500/30' : 'bg-emerald-50 border-emerald-200'}`}>
            <p className="text-emerald-400 font-semibold mb-1">Skip path</p>
            <p className={`text-[10px] ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Identity (or 1×1 projection if dims differ)</p>
          </div>
          {showGrad && (
            <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }}
              className={`rounded-xl border p-3 max-w-[140px] ${theme === 'dark' ? 'bg-amber-500/10 border-amber-500/30' : 'bg-amber-50 border-amber-200'}`}>
              <p className="text-amber-400 font-semibold mb-1">Gradient</p>
              <p className={`text-[10px] font-mono ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>∂L/∂x = ∂L/∂H·(∂F/∂x + <span className="text-amber-400 font-bold">1</span>)</p>
              <p className={`text-[10px] mt-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-500'}`}>+1 prevents vanishing</p>
            </motion.div>
          )}
        </div>
      </div>

      <p className={`text-xs mt-3 text-center ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
        {blockType === 'basic' ? 'Basic block — ResNet-18 / ResNet-34 (two 3×3 convs)' : 'Bottleneck block — ResNet-50 / ResNet-101 / ResNet-152 (1×1 → 3×3 → 1×1)'}
      </p>
    </div>
  )
}

// ── ResNet variants comparison ─────────────────────────────────────────────────
function ResNetVariants({ theme }) {
  const variants = [
    { name: 'ResNet-18',  layers: 18,  block: 'Basic',      params: '11.7M', top5: '89.1%', color: '#6366F1' },
    { name: 'ResNet-34',  layers: 34,  block: 'Basic',      params: '21.8M', top5: '91.3%', color: '#8B5CF6' },
    { name: 'ResNet-50',  layers: 50,  block: 'Bottleneck', params: '25.6M', top5: '92.9%', color: '#06B6D4' },
    { name: 'ResNet-101', layers: 101, block: 'Bottleneck', params: '44.5M', top5: '93.6%', color: '#10B981' },
    { name: 'ResNet-152', layers: 152, block: 'Bottleneck', params: '60.2M', top5: '93.8%', color: '#F59E0B' },
  ]

  const maxParams = 60.2
  const [hovered, setHovered] = useState(null)

  return (
    <div className="space-y-3">
      {variants.map((v, i) => (
        <div key={v.name}
          onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}
          className={`rounded-xl border p-3 transition-all cursor-default ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'} ${hovered === i ? 'scale-[1.01]' : ''}`}>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-sm font-bold" style={{ color: v.color }}>{v.name}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full border ${theme === 'dark' ? 'bg-slate-700 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{v.block}</span>
            <span className={`ml-auto text-xs font-mono ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{v.params}</span>
            <span className={`text-xs font-mono font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{v.top5}</span>
          </div>
          <div className={`h-3 rounded-full overflow-hidden ${theme === 'dark' ? 'bg-slate-700' : 'bg-gray-200'}`}>
            <motion.div className="h-full rounded-full"
              style={{ backgroundColor: v.color }}
              initial={{ width: 0 }}
              animate={{ width: `${(parseFloat(v.params) / maxParams) * 100}%` }}
              transition={{ duration: 0.5, delay: i * 0.08 }} />
          </div>
          {hovered === i && (
            <p className={`text-[10px] mt-1.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
              {v.layers} layers · {v.block} residual blocks · ImageNet Top-5: {v.top5}
            </p>
          )}
        </div>
      ))}
      <p className={`text-xs text-center mt-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>Bar width = parameter count · Top-5 accuracy on ImageNet (approximate)</p>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function DeepCNN() {
  const { theme } = useApp()
  const S = `rounded-2xl border p-6 mb-6 ${theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'}`
  const LBL = `text-xs font-semibold uppercase tracking-wider mb-3 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`
  const H2 = `text-xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const BODY = `text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl">

      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/20 border border-indigo-500/30 mb-4">
          <span className="text-xs text-indigo-400 font-medium">Deep Neural Networks • Session 7</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Deep CNN <span className="gradient-text">Architectures</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          From LeNet's 5 layers to ResNet's 152 — how VGG, NiN, GoogLeNet, and ResNet pushed the limits of depth, efficiency, and accuracy on ImageNet.
        </p>
        <Callout type="analogy" title="The ImageNet race (2012–2016)">
          AlexNet (2012) started the deep learning revolution with an 8-layer CNN. VGG showed depth matters. NiN introduced 1×1 convolutions. GoogLeNet added parallel branches. ResNet proved 152 layers can be trained reliably. Each breakthrough reduced the top-5 error from ~26% to ~3.5% — near human level.
        </Callout>
      </div>

      <TheoryBlock title="Architecture evolution" cards={[
        { icon: '🧱', title: 'VGG (2014)', body: 'Very deep networks (16–19 layers) using only 3×3 conv blocks. Showed uniformity beats complexity — same kernel size everywhere, just stack more.', mono: '2–4 × Conv(3×3) + MaxPool' },
        { icon: '🔷', title: 'NiN (2014)', body: 'Replaced FC layers with Global Average Pooling. Introduced 1×1 convolutions as per-pixel MLPs inside conv blocks — the ancestor of all bottleneck designs.', mono: 'Conv(K×K) + 1×1 + 1×1 + GAP' },
        { icon: '🌐', title: 'GoogLeNet (2014)', body: 'Inception blocks apply multiple filter sizes in parallel (1×1, 3×3, 5×5) then concatenate results. More efficient than VGG with 12× fewer parameters.', mono: 'parallel branches → concat' },
        { icon: '🔗', title: 'ResNet (2015)', body: 'Skip connections let gradients bypass weight layers, enabling 100–1000+ layer networks. Solved vanishing gradients. Won ImageNet 2015 with 152 layers.', mono: 'H(x) = F(x) + x' },
        { icon: '📦', title: 'Bottleneck block', body: '1×1 conv reduces channels, 3×3 conv does spatial work, 1×1 conv restores channels. Used in ResNet-50+, Inception, and most modern architectures.', mono: '1×1 → 3×3 → 1×1' },
        { icon: '🏁', title: 'Transfer learning', body: 'All these architectures are available pretrained on ImageNet. Fine-tune on your task in minutes rather than training from scratch for weeks.', mono: 'tf.keras.applications.*' },
      ]} />

      {/* ── Section 7.1 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 7.1</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>VGG and NiN</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Very deep homogeneous networks and the introduction of 1×1 convolutions.</p>
      </div>

      {/* 7.1.1 VGG */}
      <div className={S}>
        <p className={LBL}>7.1.1 — VGG: Very Deep Networks Using Blocks</p>
        <h2 className={H2}>Depth through uniform 3×3 blocks</h2>
        <p className={`${BODY} mb-4`}>
          VGGNet (Simonyan & Zisserman, 2014) made one observation: <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>depth matters more than kernel size</strong>. By using exclusively 3×3 convolutions stacked into blocks, they showed that 16–19 layer networks significantly outperform shallower ones with larger kernels.
        </p>
        <TheoryBlock title="VGG design principles" cards={[
          { icon: '🔲', title: '3×3 only', body: 'Every convolutional layer uses a 3×3 kernel with padding=1 (same padding). Two stacked 3×3 convs cover the same 5×5 receptive field as one 5×5 — with fewer params and an extra non-linearity.', mono: '2×(3×3) params: 2×9=18 vs 5×5=25' },
          { icon: '🧱', title: 'VGG blocks', body: 'The architecture is a sequence of VGG blocks, each containing 2–4 conv layers followed by one MaxPool 2×2. Channels double as spatial size halves — trading resolution for representational depth.', mono: 'channels: 64→128→256→512→512' },
          { icon: '📦', title: 'Classifier head', body: 'After the 5 conv blocks, three fully-connected layers (4096→4096→1000) produce class scores. These FC layers contain ~123M of the total ~138M parameters in VGG-16.', mono: 'FC: 7×7×512 → 4096 → 4096 → 1000' },
        ]} />
        <Callout type="info" title="VGG-16 vs VGG-19">
          VGG-16 (13 conv + 3 FC) vs VGG-19 (16 conv + 3 FC). The extra 3 convs in blocks 3–5 improve accuracy slightly (+0.2% top-5) at a cost of ~6M extra parameters. In practice VGG-16 is more commonly used as a feature extractor.
        </Callout>
        <p className={`${BODY} mb-4`}>Click any block to inspect its layer count and parameter estimate. Toggle between VGG-16 and VGG-19:</p>
        <VGGDiagram theme={theme} />
        <DeepDive title="Why VGG is still widely used today">
          <p className={`text-sm ${BODY} mb-2`}>Despite being surpassed in accuracy by ResNets, VGG remains popular as a feature extractor for style transfer, perceptual loss functions, and image quality assessment. Its simple, uniform structure makes intermediate feature maps easy to interpret.</p>
          <Callout type="warning">VGG's main weakness is memory: 138M parameters need ~550MB just to store, and the FC layers dominate. Modern architectures (ResNet, EfficientNet) achieve better accuracy with 10× fewer parameters by using global average pooling instead of FC heads.</Callout>
        </DeepDive>
        <CodeBlock code={CODE_VGG} />
      </div>

      {/* 7.1.2 NiN */}
      <div className={S}>
        <p className={LBL}>7.1.2 — Network in Network (NiN)</p>
        <h2 className={H2}>Per-pixel MLPs and Global Average Pooling</h2>
        <p className={`${BODY} mb-4`}>
          NiN (Lin et al., 2014) challenged the "linear conv → FC" paradigm. Two innovations: (1) replace each conv with a mini-MLP using 1×1 convolutions, and (2) replace the FC classifier with Global Average Pooling — cutting parameters dramatically.
        </p>
        <NiNDiagram theme={theme} />
        <TheoryBlock title="NiN innovations" cards={[
          { icon: '⚡', title: '1×1 convolution = per-pixel MLP', body: 'A 1×1 conv applies a learned linear combination across all channels at every spatial position independently. Stack two with ReLU and you have a per-pixel 2-layer MLP — much more expressive than a single linear filter.', mono: 'per-pixel: 1×1 + ReLU + 1×1 + ReLU' },
          { icon: '🌍', title: 'Global Average Pooling (GAP)', body: 'Instead of flattening to a huge FC layer, average each feature map into a single number. For C feature maps → C-dimensional vector. Then softmax directly. Adds almost zero parameters to the classifier.', mono: '(B,C,H,W) → GAP → (B,C) → softmax' },
          { icon: '🎁', title: 'Interpretability', body: 'With GAP, each feature map corresponds directly to a class score. This enables Class Activation Mapping (CAM) — visualising which regions of the image influenced each class prediction.', mono: 'Class Activation Mapping (CAM)' },
        ]} />
        <Callout type="success" title="NiN's lasting legacy">
          NiN's 1×1 conv is now ubiquitous — it appears in GoogLeNet's bottleneck, ResNet's projection shortcuts, and virtually every modern efficient architecture. GAP replacing FC heads is standard in ResNet, EfficientNet, and MobileNet.
        </Callout>
        <CodeBlock code={CODE_NIN} />
      </div>

      {/* ── Section 7.2 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-purple-500/5 border-purple-500/20' : 'bg-purple-50 border-purple-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-purple-400' : 'text-purple-600'}`}>Section 7.2</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>GoogLeNet and ResNet</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Parallel branches, skip connections, and the architecture that made 150+ layer networks trainable.</p>
      </div>

      {/* 7.2.1 GoogLeNet */}
      <div className={S}>
        <p className={LBL}>7.2.1 — GoogLeNet: Inception Blocks</p>
        <h2 className={H2}>Multiple filter sizes in parallel</h2>
        <p className={`${BODY} mb-4`}>
          GoogLeNet (Szegedy et al., 2014) won ImageNet with a radical question: <em>"What's the right filter size?"</em> — answered with <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>all of them at once</strong>. The Inception block applies 1×1, 3×3, and 5×5 convolutions in parallel, letting the network learn which scale of features to rely on.
        </p>
        <TheoryBlock title="Inception block design" cards={[
          { icon: '🌐', title: 'Parallel branches', body: 'Four branches process the same input simultaneously: 1×1 conv, 1×1→3×3 conv, 1×1→5×5 conv, and 3×3 max-pool→1×1 conv. All use same padding so spatial dimensions match.', mono: '4 branches → concat along channels' },
          { icon: '📉', title: 'Bottleneck 1×1', body: 'Before the expensive 3×3 and 5×5 convolutions, a 1×1 conv reduces the channel count (e.g. 192→96). This cuts FLOPs by 8× for the 3×3 and 25× for the 5×5.', mono: '1×1 reduce → 3×3 or 5×5' },
          { icon: '⚖️', title: 'Efficient design', body: 'GoogLeNet has 22 layers but only 6.8M parameters — 20× fewer than VGG-16 (138M). Achieved similar ImageNet accuracy with far less computation.', mono: '6.8M params vs VGG-16\'s 138M' },
        ]} />
        <Callout type="formula" mono="output_channels = f1 + f3 + f5 + f_pool">
          All four branches produce feature maps of the same H×W (same padding). They are concatenated along the channel axis. The total output channels is the sum of all branch filter counts — the network learns to weight each branch's contribution.
        </Callout>
        <p className={`${BODY} mb-4`}>Select an Inception block variant to see how filter counts change across branches and the concatenated output:</p>
        <InceptionViz theme={theme} />
        <DeepDive title="Inception V2, V3, V4 improvements">
          <TheoryBlock title="" cards={[
            { icon: '2️⃣', title: 'Inception V2', body: 'Factorises 5×5 into two 3×3 convs, and n×n into 1×n + n×1 (asymmetric factorisation). More non-linearities, fewer parameters.', mono: '5×5 → 3×3 + 3×3' },
            { icon: '3️⃣', title: 'Inception V3', body: 'Adds batch normalisation throughout. Factorises large filters more aggressively. Uses auxiliary classifiers to inject gradient into earlier layers.', mono: 'BatchNorm + deeper factorisation' },
            { icon: '4️⃣', title: 'Inception-ResNet', body: 'Combines Inception branches with ResNet skip connections. Training is faster and accuracy higher than either architecture alone.', mono: 'Inception branches + skip connections' },
          ]} />
        </DeepDive>
        <CodeBlock code={CODE_INCEPTION} />
      </div>

      {/* 7.2.2 + 7.2.3 ResNet */}
      <div className={S}>
        <p className={LBL}>7.2.2–7.2.3 — ResNet: Residual Blocks and Vanishing Gradients</p>
        <h2 className={H2}>The skip connection that changed everything</h2>
        <p className={`${BODY} mb-4`}>
          Residual Networks (He et al., 2015) solved a fundamental problem: why do very deep networks train <em>worse</em> than shallow ones, even on training data? The answer is the vanishing gradient — and the fix is adding the input back to the output of each block.
        </p>
        <TheoryBlock title="Residual learning" cards={[
          { icon: '🔗', title: 'Residual formulation', body: 'Instead of learning H(x) directly, the block learns the residual F(x) = H(x) − x. The output is H(x) = F(x) + x. If the optimal H(x) ≈ x, the block just learns F(x) ≈ 0 — much easier.', mono: 'H(x) = F(x) + x' },
          { icon: '∂', title: 'Gradient through skip path', body: '∂L/∂x = ∂L/∂H · (∂F/∂x + 1). The +1 from the skip connection ensures a gradient of at least the upstream value reaches x — preventing it from vanishing over 100+ layers.', mono: '∂L/∂x = ∂L/∂H · (∂F/∂x + 1)' },
          { icon: '📐', title: 'Projection shortcut', body: 'When the block changes channel count or spatial size (stride 2), the identity shortcut is replaced by a 1×1 conv (projection) to match dimensions before the addition.', mono: 'if dims differ: shortcut = Conv(1×1)' },
        ]} />
        <Callout type="formula" mono="H(x) = F(x) + x     gradient: ∂L/∂x includes +1 term">
          This +1 means even if ∂F/∂x → 0 (vanishing), the gradient still flows back as ∂L/∂H. In a 150-layer network, this difference is the reason training converges at all — without skip connections, gradients in early layers are effectively zero.
        </Callout>
        <p className={`${BODY} mb-4`}>Toggle between basic and bottleneck blocks, and enable gradient flow mode to see the +1 term:</p>
        <ResidualViz theme={theme} />
        <DeepDive title="Why not just use identity everywhere — the degradation problem">
          <p className={`text-sm ${BODY} mb-2`}>Without skip connections, very deep networks suffer <em>degradation</em>: training accuracy decreases with depth even when the deeper model should theoretically be at least as good as the shallower one (it could just learn identity for extra layers).</p>
          <p className={`text-sm ${BODY} mb-2`}>The problem is optimisation — it's hard for a stack of conv+BN+ReLU layers to learn the identity function. Residual formulation makes this trivial: just drive all weights to zero. This is why residual networks don't suffer degradation.</p>
          <Callout type="success">ResNet won 1st place in all 5 ImageNet competition tracks (classification, detection, localisation, COCO detection, COCO segmentation) in 2015 — a sweep that had never happened before.</Callout>
        </DeepDive>
      </div>

      {/* 7.2.4 ResNet variants */}
      <div className={S}>
        <p className={LBL}>7.2.4 — ResNet Variants: ResNet-18, 50, 101</p>
        <h2 className={H2}>Trading depth for accuracy — the full family</h2>
        <p className={`${BODY} mb-4`}>
          ResNet comes in several variants differing in depth and block type. Shallow variants (18, 34) use basic blocks (two 3×3 convs). Deeper ones (50, 101, 152) switch to bottleneck blocks (1×1→3×3→1×1) for efficiency.
        </p>
        <ResNetVariants theme={theme} />
        <TheoryBlock title="Block types by variant" cards={[
          { icon: '🔹', title: 'ResNet-18 / 34', body: 'Basic blocks: Conv(3×3) → BN → ReLU → Conv(3×3) → BN → Add → ReLU. Simple, fast, small. Good for edge deployment or small datasets.', mono: 'basic: [3×3, 3×3]' },
          { icon: '🔷', title: 'ResNet-50 / 101 / 152', body: 'Bottleneck blocks: Conv(1×1) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1) → BN → Add → ReLU. More efficient for deeper networks.', mono: 'bottleneck: [1×1, 3×3, 1×1]' },
          { icon: '🏎️', title: 'ResNeXt / Wide ResNet', body: 'ResNeXt adds grouped convolutions (multiple parallel paths in bottleneck). Wide ResNet increases channel width instead of depth. Both outperform standard ResNets at same parameter count.', mono: 'grouped conv / wider channels' },
        ]} />
        <Callout type="info" title="Which ResNet to use?">
          ResNet-50 is the standard starting point — excellent trade-off between accuracy and speed. ResNet-18 for fast inference or small datasets. ResNet-101 when accuracy matters more than speed. For production, consider EfficientNet or MobileNet which achieve ResNet-50 accuracy at 10× fewer parameters.
        </Callout>
        <CodeBlock code={CODE_RESNET} />
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme === 'dark' ? 'bg-indigo-500/10 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>10 questions covering VGG, NiN, GoogLeNet and ResNet • +100 XP on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="deep-cnn" />
      </div>
    </motion.div>
  )
}
