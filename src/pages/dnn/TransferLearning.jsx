import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

// ── Code blocks ────────────────────────────────────────────────────────────────

const CODE_FEATURE_EXTRACT = `import tensorflow as tf
import numpy as np

# ── Feature extraction: freeze the base, train only the new head ──

base = tf.keras.applications.ResNet50(
    weights='imagenet',       # pretrained ImageNet weights
    include_top=False,         # remove the 1000-class FC head
    input_shape=(224, 224, 3)
)
base.trainable = False         # freeze ALL base layers

# Add a custom classification head for your task (e.g. 5 flower classes)
model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation='softmax'),  # 5 new classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Only head layers are trained — fast!
trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
total     = sum(np.prod(v.shape) for v in model.weights)
print(f"Trainable params : {trainable:,}")
print(f"Frozen params    : {total - trainable:,}")
print(f"Trainable %      : {100*trainable/total:.1f}%")`

const CODE_FINE_TUNE = `import tensorflow as tf

# Step 1 — Feature extraction (warm up the head)
base = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=5, validation_data=val_ds)

# Step 2 — Fine-tune: unfreeze top layers of the base
base.trainable = True

# Freeze everything below layer 140; unfreeze the last ~10 layers
for layer in base.layers[:140]:
    layer.trainable = False

print(f"Trainable layers: {sum(1 for l in base.layers if l.trainable)}")

# Re-compile with a MUCH smaller learning rate (10-100× smaller)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),   # ← key: tiny lr
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, epochs=10, validation_data=val_ds)`

const CODE_LR_SCHEDULE = `import tensorflow as tf
import numpy as np

# ── Learning rate warmup + cosine decay for fine-tuning ──

def warmup_cosine_decay(warmup_steps, total_steps, peak_lr, min_lr=1e-7):
    """Linear warmup → cosine decay schedule."""
    def schedule(step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(warmup_steps, tf.float32)
        total  = tf.cast(total_steps,  tf.float32)
        # Warmup phase
        warmup_lr = peak_lr * (step / warmup)
        # Cosine decay phase
        progress  = (step - warmup) / (total - warmup)
        cosine_lr = min_lr + 0.5 * (peak_lr - min_lr) * (
            1 + tf.cos(np.pi * progress))
        return tf.where(step < warmup, warmup_lr, cosine_lr)
    return schedule

steps_per_epoch = 100   # batches per epoch
total_epochs    = 15

lr_fn = warmup_cosine_decay(
    warmup_steps = 2 * steps_per_epoch,   # 2 epoch warmup
    total_steps  = total_epochs * steps_per_epoch,
    peak_lr      = 1e-4,
    min_lr       = 1e-7,
)

optimiser = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.LambdaDecay(
        initial_learning_rate=1e-4,
        decay_steps=total_epochs * steps_per_epoch,
        decay_fn=lambda step: lr_fn(step) / 1e-4  # normalised
    )
)

# Simpler: ReduceLROnPlateau for automatic lr reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3,
    min_lr=1e-7, verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)`

const QUIZ_QUESTIONS = [
  { question: 'Transfer learning works because:', options: ['All tasks share the same data', 'Low-level features (edges, textures) learned on large datasets are useful for related tasks', 'ImageNet contains every possible image', 'Pretrained weights are always optimal'], correct: 1, explanation: 'Early CNN layers learn generic features (edges, colours, textures) that are useful across many visual tasks. Only high-level task-specific layers need relearning.' },
  { question: 'Feature extraction vs fine-tuning — the key difference is:', options: ['Feature extraction uses more data', 'Feature extraction freezes the pretrained base; fine-tuning updates some or all base weights', 'Fine-tuning removes the pretrained weights', 'They are identical'], correct: 1, explanation: 'Feature extraction: base.trainable=False, only the new head is trained. Fine-tuning: some or all base layers are unfrozen and trained at a very small learning rate.' },
  { question: 'When should you use feature extraction over fine-tuning?', options: ['Always', 'When your dataset is very small and similar to the source domain', 'When you have millions of labelled examples', 'When the source and target domains are very different'], correct: 1, explanation: 'With little data, fine-tuning risks overfitting. Feature extraction keeps the powerful pretrained representations frozen and only trains the small new head.' },
  { question: 'Why must the learning rate be much smaller during fine-tuning?', options: ['To save computation', 'To avoid overwriting useful pretrained features with large gradient updates', 'Larger lr causes faster convergence', 'Pretrained weights require larger gradients'], correct: 1, explanation: 'The pretrained weights encode valuable representations from millions of images. Large updates would destroy them. A small lr (1e-5 vs 1e-3) makes only gentle adjustments.' },
  { question: 'Which layers should you unfreeze first when fine-tuning?', options: ['The earliest (first) layers', 'The latest (top) layers of the base, closest to the head', 'All layers simultaneously', 'Only the batch normalisation layers'], correct: 1, explanation: 'Top layers of the base contain the most task-specific, high-level features — most likely to differ between source and target domains. Early layers (edges, textures) are more universal and should stay frozen longer.' },
  { question: 'Domain shift refers to:', options: ['Changing the model architecture', 'A difference in distribution between the source (pretrained) domain and the target domain', 'Changing the loss function', 'Using a different optimiser'], correct: 1, explanation: 'Domain shift is the statistical difference between training and target distributions. Large shift (e.g. ImageNet → medical X-rays) means top-layer features may not transfer and deeper fine-tuning is needed.' },
  { question: 'For a small dataset with large domain shift, the best strategy is:', options: ['Full fine-tuning with large lr', 'Train only the head (feature extraction), possibly fine-tune last few layers', 'Discard pretrained weights and train from scratch', 'Freeze everything including the head'], correct: 1, explanation: 'Small data + large shift: pretrained features may not help much, but fine-tuning everything risks overfitting. The safest bet is to start with feature extraction then gently fine-tune the last few layers.' },
  { question: 'include_top=False in tf.keras.applications removes:', options: ['The first convolutional layer', 'The final FC classification head (designed for ImageNet 1000 classes)', 'All pooling layers', 'The batch normalisation layers'], correct: 1, explanation: 'include_top=False removes the GlobalAveragePooling + Dense(1000) head that produces ImageNet class scores, leaving the convolutional feature extractor for you to attach your own head.' },
  { question: 'Progressive unfreezing trains layers:', options: ['All at once with the same lr', 'From top to bottom in stages, each stage with lower lr for earlier layers', 'Randomly', 'Only the head, never the base'], correct: 1, explanation: 'Progressive unfreezing (from ULMFiT) gradually unfreezes from the top layer downward in stages. Earlier layers are fine-tuned at smaller lr because they encode more general features.' },
  { question: 'A warmup phase in a learning rate schedule:', options: ['Starts high and stays high', 'Starts from near zero and gradually increases to the peak lr before decaying', 'Is only used in feature extraction', 'Removes the need for weight decay'], correct: 1, explanation: 'Warmup prevents large gradient updates at the start of fine-tuning when the new head is random. Starting from near-zero lr gives the head time to stabilise before the full lr kicks in.' },
]

// ── Transfer learning strategy selector ───────────────────────────────────────
function StrategySelector({ theme }) {
  const [dataset, setDataset] = useState('small')
  const [similarity, setSimilarity] = useState('high')

  const strategies = {
    'small-high':  { title: 'Feature Extraction', icon: '🔒', color: '#6366F1', lr: '1e-3', unfreeze: 'None', epochs: '10–20', desc: 'Base fully frozen. Train only the new head. Safe from overfitting. ImageNet features transfer directly — ideal scenario.' },
    'small-low':   { title: 'Light Fine-tuning', icon: '🔓', color: '#F59E0B', lr: '1e-5', unfreeze: 'Last 10–20%', epochs: '15–30', desc: 'Unfreeze top few layers. Features differ enough that top-layer adaptation helps, but small dataset still limits how deep you can go.' },
    'large-high':  { title: 'Full Fine-tuning', icon: '🔥', color: '#10B981', lr: '1e-4', unfreeze: 'All layers', epochs: '20–50', desc: 'Enough data to fine-tune the whole network. Start with feature extraction for 5 epochs, then unfreeze all. Use a small lr with warmup.' },
    'large-low':   { title: 'Deep Fine-tuning', icon: '🚀', color: '#EF4444', lr: '1e-4→1e-5', unfreeze: 'All (progressive)', epochs: '30–60', desc: 'Large domain shift requires significant adaptation. Progressive unfreezing + discriminative lr: lower for early layers, higher for top layers.' },
  }

  const key = `${dataset}-${similarity}`
  const s = strategies[key]

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-5">
        <div>
          <p className={`text-xs font-semibold mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>Your dataset size</p>
          <div className="flex gap-2">
            {[['small', 'Small (< 5k)'], ['large', 'Large (> 10k)']].map(([v, l]) => (
              <button key={v} onClick={() => setDataset(v)}
                className={`flex-1 py-2 rounded-lg text-xs font-semibold border transition-all ${dataset === v
                  ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
                  : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{l}</button>
            ))}
          </div>
        </div>
        <div>
          <p className={`text-xs font-semibold mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>Similarity to ImageNet</p>
          <div className="flex gap-2">
            {[['high', 'Similar'], ['low', 'Very different']].map(([v, l]) => (
              <button key={v} onClick={() => setSimilarity(v)}
                className={`flex-1 py-2 rounded-lg text-xs font-semibold border transition-all ${similarity === v
                  ? theme === 'dark' ? 'bg-purple-500/20 border-purple-500/40 text-purple-300' : 'bg-purple-100 border-purple-300 text-purple-700'
                  : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{l}</button>
            ))}
          </div>
        </div>
      </div>
      <motion.div key={key} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
        className={`rounded-2xl border p-4 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">{s.icon}</span>
          <div>
            <p className="text-base font-bold" style={{ color: s.color }}>{s.title}</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>Recommended strategy</p>
          </div>
        </div>
        <p className={`text-sm mb-3 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>{s.desc}</p>
        <div className="grid grid-cols-3 gap-2 text-xs">
          {[
            { label: 'Learning rate', val: s.lr },
            { label: 'Unfreeze', val: s.unfreeze },
            { label: 'Epochs', val: s.epochs },
          ].map(m => (
            <div key={m.label} className={`rounded-xl p-2 border text-center ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-white border-gray-200'}`}>
              <p className={`font-mono font-bold text-xs`} style={{ color: s.color }}>{m.val}</p>
              <p className={`text-[10px] mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{m.label}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}

// ── Layer freeze visualiser ────────────────────────────────────────────────────
function FreezeViz({ theme }) {
  const [unfreezeFrom, setUnfreezeFrom] = useState(140)

  const layerGroups = [
    { label: 'Block 1–2', range: [0, 36],   desc: 'Edges, colours, basic textures', icon: '🔲' },
    { label: 'Block 3',   range: [36, 80],  desc: 'Corners, shapes, simple patterns', icon: '⬡' },
    { label: 'Block 4',   range: [80, 120], desc: 'Complex textures, object parts', icon: '🌀' },
    { label: 'Block 5',   range: [120, 155], desc: 'High-level features, semantic parts', icon: '🧩' },
    { label: 'Head',      range: [155, 160], desc: 'Task-specific classification', icon: '🎯' },
  ]

  const totalLayers = 160
  const trainable = totalLayers - unfreezeFrom
  const trainPct = ((trainable / totalLayers) * 100).toFixed(0)

  return (
    <div>
      <div className="mb-4">
        <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
          Freeze layers 0–{unfreezeFrom - 1} &nbsp;|&nbsp; Unfreeze layers {unfreezeFrom}–{totalLayers - 1}
          &nbsp;(<span className="text-emerald-400 font-bold">{trainPct}%</span> trainable)
        </label>
        <input type="range" min="0" max={totalLayers} step="5" value={unfreezeFrom}
          onChange={e => setUnfreezeFrom(+e.target.value)}
          className="w-full accent-indigo-500" />
      </div>

      <div className="space-y-2">
        {layerGroups.map((grp) => {
          const start = grp.range[0], end = grp.range[1]
          const frozenEnd = Math.min(end, unfreezeFrom)
          const thawedStart = Math.max(start, unfreezeFrom)
          const frozenPct = Math.max(0, ((frozenEnd - start) / (end - start)) * 100)
          const thawedPct = Math.max(0, ((end - thawedStart) / (end - start)) * 100)
          const fullyFrozen = unfreezeFrom >= end
          const fullyThawed = unfreezeFrom <= start

          return (
            <div key={grp.label} className={`rounded-xl border p-3 transition-all ${
              fullyFrozen ? (theme === 'dark' ? 'bg-slate-800/40 border-white/5 opacity-60' : 'bg-gray-50 border-gray-100 opacity-70')
              : fullyThawed ? (theme === 'dark' ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-emerald-50 border-emerald-200')
              : (theme === 'dark' ? 'bg-indigo-500/10 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200')
            }`}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-base">{grp.icon}</span>
                <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{grp.label}</span>
                <span className={`ml-auto text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                  fullyFrozen ? (theme === 'dark' ? 'bg-slate-700 text-gray-500' : 'bg-gray-200 text-gray-400') :
                  fullyThawed ? 'bg-emerald-500/20 text-emerald-400' :
                  'bg-indigo-500/20 text-indigo-400'}`}>
                  {fullyFrozen ? '🔒 Frozen' : fullyThawed ? '🔥 Trainable' : '⚡ Partial'}
                </span>
              </div>
              <p className={`text-[10px] mb-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{grp.desc}</p>
              <div className={`h-3 rounded-full overflow-hidden flex ${theme === 'dark' ? 'bg-slate-700' : 'bg-gray-200'}`}>
                <motion.div className="h-full bg-slate-500/60 rounded-l-full"
                  animate={{ width: `${frozenPct}%` }} transition={{ duration: 0.25 }} />
                <motion.div className="h-full bg-emerald-500 rounded-r-full"
                  animate={{ width: `${thawedPct}%` }} transition={{ duration: 0.25 }} />
              </div>
            </div>
          )
        })}
      </div>
      <p className={`text-xs mt-3 text-center ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
        Grey = frozen &nbsp;·&nbsp; Green = trainable &nbsp;·&nbsp; Drag the slider to unfreeze more layers
      </p>
    </div>
  )
}

// ── Learning rate schedule visualiser ─────────────────────────────────────────
function LRScheduleViz({ theme }) {
  const [schedule, setSchedule] = useState('warmup_cosine')
  const [peakLR, setPeakLR] = useState(1e-4)
  const canvasRef = useRef(null)
  const W = 380, H = 200

  const totalSteps = 150
  const warmupSteps = 20

  const getLR = (step, type) => {
    switch (type) {
      case 'constant':
        return peakLR
      case 'warmup_cosine': {
        if (step < warmupSteps) return peakLR * (step / warmupSteps)
        const prog = (step - warmupSteps) / (totalSteps - warmupSteps)
        return 1e-7 + 0.5 * (peakLR - 1e-7) * (1 + Math.cos(Math.PI * prog))
      }
      case 'step_decay':
        if (step < 50)  return peakLR
        if (step < 100) return peakLR * 0.2
        return peakLR * 0.04
      case 'exponential':
        return peakLR * Math.pow(0.95, step)
      case 'warmup_linear': {
        if (step < warmupSteps) return peakLR * (step / warmupSteps)
        return peakLR * (1 - (step - warmupSteps) / (totalSteps - warmupSteps))
      }
      default: return peakLR
    }
  }

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = { l: 50, r: 15, t: 15, b: 28 }
    const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b

    const maxLR = peakLR * 1.05
    const toX = step => pad.l + (step / totalSteps) * pW
    const toY = lr => (H - pad.b) - (lr / maxLR) * pH

    // Grid
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
    ctx.lineWidth = 1; ctx.setLineDash([3, 3])
    for (let v = 0; v <= 4; v++) {
      ctx.beginPath(); ctx.moveTo(pad.l, (H - pad.b) - (v / 4) * pH); ctx.lineTo(W - pad.r, (H - pad.b) - (v / 4) * pH); ctx.stroke()
    }
    ctx.setLineDash([])

    // Warmup boundary
    ctx.beginPath(); ctx.moveTo(toX(warmupSteps), pad.t); ctx.lineTo(toX(warmupSteps), H - pad.b)
    ctx.strokeStyle = 'rgba(251,191,36,0.3)'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([])
    ctx.fillStyle = 'rgba(251,191,36,0.5)'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center'
    ctx.fillText('warmup', toX(warmupSteps), pad.t + 9)

    // Axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke()

    // LR curve
    const grad = ctx.createLinearGradient(pad.l, 0, W - pad.r, 0)
    grad.addColorStop(0, '#6366F1'); grad.addColorStop(0.5, '#8B5CF6'); grad.addColorStop(1, '#06B6D4')
    ctx.beginPath()
    for (let step = 0; step <= totalSteps; step++) {
      const lr = getLR(step, schedule)
      step === 0 ? ctx.moveTo(toX(step), toY(lr)) : ctx.lineTo(toX(step), toY(lr))
    }
    ctx.strokeStyle = grad; ctx.lineWidth = 2.5; ctx.stroke()

    // Axis labels
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '9px monospace'; ctx.textAlign = 'right'
    const exponent = Math.round(Math.log10(peakLR))
    ctx.fillText(`1e${exponent}`, pad.l - 4, toY(peakLR) + 3)
    ctx.fillText('0', pad.l - 4, H - pad.b + 3)
    ctx.textAlign = 'center'
    for (const step of [0, 50, 100, 150]) ctx.fillText(step, toX(step), H - pad.b + 12)
    ctx.fillText('Step', W / 2, H - 2)
    ctx.save(); ctx.translate(11, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('Learning Rate', 0, 0); ctx.restore()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schedule, peakLR, theme])

  const scheduleNames = {
    warmup_cosine:  'Warmup + Cosine Decay',
    warmup_linear:  'Warmup + Linear Decay',
    step_decay:     'Step Decay',
    exponential:    'Exponential Decay',
    constant:       'Constant LR',
  }

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>Schedule</label>
          <div className="space-y-1">
            {Object.entries(scheduleNames).map(([k, v]) => (
              <button key={k} onClick={() => setSchedule(k)}
                className={`w-full py-1.5 px-2 rounded-lg text-xs text-left border transition-all ${schedule === k
                  ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
                  : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'}`}>{v}</button>
            ))}
          </div>
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Peak LR = <span className="text-indigo-400 font-bold font-mono">{peakLR.toExponential(0)}</span>
          </label>
          <input type="range" min="-6" max="-3" step="1" value={Math.round(Math.log10(peakLR))}
            onChange={e => setPeakLR(Math.pow(10, +e.target.value))}
            className="w-full accent-indigo-500 mb-3" />
          <div className={`rounded-xl border p-3 text-xs ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className={`font-semibold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{scheduleNames[schedule]}</p>
            <p className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
              {schedule === 'warmup_cosine' && 'Best for fine-tuning. Gradual warmup avoids large early updates; cosine decay reduces lr smoothly.'}
              {schedule === 'warmup_linear' && 'Simpler than cosine. Warmup + linear decay to zero. Used in BERT and many NLP models.'}
              {schedule === 'step_decay' && 'Drop lr by 0.2× at fixed intervals. Classic but abrupt. May cause loss spikes at drop points.'}
              {schedule === 'exponential' && 'Smooth continuous decay. Easy to tune. Can decay too fast and stop learning early.'}
              {schedule === 'constant' && 'No decay. Simple but may oscillate near minimum. Only good for short fine-tuning runs.'}
            </p>
          </div>
        </div>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
    </div>
  )
}

// ── Domain shift quadrant ──────────────────────────────────────────────────────
function DomainShiftQuadrant({ theme }) {
  const examples = [
    { x: 80,  y: 20,  label: 'Cats vs Dogs', src: 'ImageNet', strategy: 'Feature extraction', color: '#6366F1' },
    { x: 70,  y: 35,  label: 'Flowers (102 classes)', src: 'ImageNet', strategy: 'Feature extraction + light fine-tune', color: '#8B5CF6' },
    { x: 30,  y: 55,  label: 'Satellite imagery', src: 'ImageNet', strategy: 'Fine-tune top 50%', color: '#F59E0B' },
    { x: 15,  y: 80,  label: 'Medical X-rays', src: 'ImageNet', strategy: 'Deep fine-tuning or train from scratch', color: '#EF4444' },
    { x: 60,  y: 70,  label: 'Cartoon faces', src: 'ImageNet', strategy: 'Progressive fine-tuning', color: '#06B6D4' },
    { x: 85,  y: 85,  label: 'Street scenes', src: 'ImageNet', strategy: 'Full fine-tuning', color: '#10B981' },
  ]
  const [active, setActive] = useState(null)

  return (
    <div>
      <div className={`relative rounded-xl border overflow-hidden`}
        style={{ height: 260, background: theme === 'dark' ? '#0F172A' : '#F8FAFC', borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#E5E7EB' }}>

        {/* Axis labels */}
        <div className={`absolute bottom-2 left-1/2 -translate-x-1/2 text-xs font-semibold ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
          ← Less similar to ImageNet &nbsp;&nbsp; More similar →
        </div>
        <div className="absolute left-2 top-1/2 -translate-y-1/2 text-xs font-semibold" style={{ writingMode: 'vertical-rl', color: theme === 'dark' ? '#6B7280' : '#9CA3AF' }}>
          More data ↑
        </div>

        {/* Quadrant backgrounds */}
        <div className="absolute inset-0 grid grid-cols-2 grid-rows-2 pointer-events-none" style={{ left: 24, bottom: 20, right: 0, top: 0 }}>
          {[
            { bg: 'rgba(16,185,129,0.06)', label: 'Fine-tune all', pos: 'bottom-right' },
            { bg: 'rgba(99,102,241,0.06)', label: 'Feature extract', pos: 'bottom-left' },
            { bg: 'rgba(245,158,11,0.06)', label: 'Progressive FT', pos: 'top-right' },
            { bg: 'rgba(239,68,68,0.06)',  label: 'Deep FT / scratch', pos: 'top-left' },
          ].map((q, i) => (
            <div key={i} className="relative flex items-center justify-center" style={{ backgroundColor: q.bg }}>
            </div>
          ))}
        </div>

        {/* Data points */}
        {examples.map((ex, i) => (
          <button key={i}
            onClick={() => setActive(active === i ? null : i)}
            className="absolute w-4 h-4 rounded-full border-2 border-white transition-transform hover:scale-150 focus:scale-150"
            style={{ left: `${ex.x}%`, top: `${100 - ex.y}%`, backgroundColor: ex.color, transform: 'translate(-50%, -50%)' }}
            title={ex.label}
          />
        ))}
      </div>

      <AnimatePresence>
        {active !== null && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
            className={`mt-3 rounded-xl border p-3 overflow-hidden ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
            <p className="text-xs font-bold mb-0.5" style={{ color: examples[active].color }}>{examples[active].label}</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
              <span className={`font-semibold ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>Strategy: </span>
              {examples[active].strategy}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
      <p className={`text-xs mt-2 text-center ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>Click any dot to see the recommended strategy</p>
    </div>
  )
}

// ── Feature layer hierarchy ────────────────────────────────────────────────────
function LayerHierarchy({ theme }) {
  const layers = [
    { depth: 1, label: 'Conv Block 1', features: 'Edges, colours, gradients', transferable: 99, icon: '〰️' },
    { depth: 2, label: 'Conv Block 2', features: 'Corners, simple textures', transferable: 95, icon: '⬡' },
    { depth: 3, label: 'Conv Block 3', features: 'Complex textures, patterns', transferable: 80, icon: '🌀' },
    { depth: 4, label: 'Conv Block 4', features: 'Object parts, shapes', transferable: 55, icon: '🧩' },
    { depth: 5, label: 'Conv Block 5', features: 'High-level semantics', transferable: 25, icon: '🐱' },
    { depth: 6, label: 'Head (FC / GAP)', features: 'Task-specific logits', transferable: 0, icon: '🎯' },
  ]

  return (
    <div className="space-y-2">
      {layers.map((l, i) => (
        <div key={i} className={`rounded-xl border p-3 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`}>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-base">{l.icon}</span>
            <span className={`text-xs font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{l.label}</span>
            <span className={`ml-auto text-xs font-mono font-bold ${l.transferable > 70 ? 'text-emerald-400' : l.transferable > 30 ? 'text-amber-400' : 'text-red-400'}`}>
              {l.transferable}% transferable
            </span>
          </div>
          <p className={`text-xs mb-1.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{l.features}</p>
          <div className={`h-2 rounded-full overflow-hidden ${theme === 'dark' ? 'bg-slate-700' : 'bg-gray-200'}`}>
            <motion.div className={`h-full rounded-full ${l.transferable > 70 ? 'bg-emerald-500' : l.transferable > 30 ? 'bg-amber-500' : 'bg-red-500'}`}
              initial={{ width: 0 }} animate={{ width: `${l.transferable}%` }}
              transition={{ duration: 0.4, delay: i * 0.07 }} />
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function TransferLearning() {
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
          <span className="text-xs text-cyan-400 font-medium">Deep Neural Networks • Session 8</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Transfer Learning <span className="gradient-text">&amp; Fine-Tuning</span>
        </h1>
        <p className={`text-lg mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
          Don't train from scratch — <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>borrow representations</strong> from models trained on millions of images and adapt them to your task in hours.
        </p>
        <Callout type="analogy" title="Analogy: learning to drive after riding a bike">
          A cyclist already knows balance, traffic rules, and spatial reasoning. Learning to drive doesn't start from zero — they transfer those skills and only learn what's new (steering wheel, gears, mirrors). Transfer learning works the same way: the model already "knows" about edges, shapes, and textures. You only teach it what's specific to your task.
        </Callout>
      </div>

      <TheoryBlock title="Core Concepts" cards={[
        { icon: '♻️', title: 'Transfer learning', body: 'Reuse weights pretrained on a large source dataset (ImageNet, 1.28M images, 1000 classes) as the starting point for a different, smaller target task.', mono: 'source domain → target domain' },
        { icon: '🔒', title: 'Feature extraction', body: 'Freeze all pretrained layers. Train only a new classification head on top. Fast, low data requirement, low overfitting risk.', mono: 'base.trainable = False' },
        { icon: '🔥', title: 'Fine-tuning', body: 'Unfreeze some or all pretrained layers and continue training on the target data with a very small learning rate. Adapts high-level features to the new task.', mono: 'base.trainable = True  lr ≪ 1e-4' },
        { icon: '📐', title: 'Domain shift', body: 'Statistical difference between source and target distributions. High shift = more fine-tuning needed. Low shift = feature extraction may suffice.', mono: 'P(x_source) ≠ P(x_target)' },
        { icon: '⏱️', title: 'Warmup schedule', body: 'Linearly increase lr from ~0 to peak over first few epochs. Prevents the new head\'s random weights from destroying pretrained features with large early gradients.', mono: 'lr: 0 → peak → decay' },
        { icon: '📦', title: 'Progressive unfreezing', body: 'Unfreeze from top to bottom in stages. Top layers adapt first; early layers (most general) are touched last at the smallest lr.', mono: 'top → middle → bottom' },
      ]} />

      {/* ── Section 8.1 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-cyan-500/5 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`}>Section 8.1</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Transfer Learning</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>What transfer learning is, why it works, and when feature extraction vs fine-tuning is the right choice.</p>
      </div>

      {/* 8.1.1 What is TL */}
      <div className={S}>
        <p className={LBL}>8.1.1 — What is Transfer Learning? Motivation and Use Cases</p>
        <h2 className={H2}>Borrowing knowledge from a related task</h2>
        <p className={`${BODY} mb-4`}>
          Training a CNN from scratch on ImageNet takes days on multiple GPUs and requires ~1.28 million labelled images. Most real tasks have far less data. Transfer learning lets you start from a model that already understands visual concepts and adapt it quickly.
        </p>
        <TheoryBlock title="Why transfer learning works" cards={[
          { icon: '🔺', title: 'Hierarchical features', body: 'Early layers learn task-agnostic features (edges, gradients) that are useful everywhere. Middle layers learn textures and shapes. Only top layers are highly task-specific.', mono: 'generic → specific, bottom → top' },
          { icon: '📊', title: 'Data efficiency', body: 'A pretrained model needs 10–100× less labelled data to achieve good performance. The features are already learned — you\'re just mapping them to new labels.', mono: '100 examples → ~80% accuracy' },
          { icon: '⚡', title: 'Training speed', body: 'Feature extraction converges in 10–20 epochs. Fine-tuning in 15–50. Training from scratch may need 100+ epochs on the same dataset for comparable results.', mono: 'hours vs days of compute' },
        ]} />
        <Callout type="success" title="Real-world use cases">
          Medical imaging (few labelled scans → ResNet fine-tuned on X-rays), defect detection in manufacturing (100 labelled defect images → 95%+ accuracy), custom object detection, satellite image classification, art style recognition, and virtually any computer vision task with limited data.
        </Callout>
      </div>

      {/* 8.1.2 Pretrained models */}
      <div className={S}>
        <p className={LBL}>8.1.2 — Pretrained Models and ImageNet Weights</p>
        <h2 className={H2}>What the model already knows</h2>
        <p className={`${BODY} mb-4`}>
          A model pretrained on ImageNet has seen 1.28 million images across 1000 categories. Its weights encode a rich hierarchy of visual knowledge. The lower the layer, the more universal the knowledge — and the less you need to change it.
        </p>
        <p className={`${BODY} mb-4`}>Each layer group learns progressively more abstract and task-specific features — the deeper you go, the less transferable to a new domain:</p>
        <LayerHierarchy theme={theme} />
        <Callout type="info" title="Available pretrained models in tf.keras.applications">
          <span className="font-mono">VGG16/19, ResNet50/101/152, InceptionV3, EfficientNetB0–B7, MobileNetV2/V3, DenseNet121/169/201, NASNet</span> — all pretrained on ImageNet, directly usable with <span className="font-mono">weights='imagenet'</span>.
        </Callout>
        <DeepDive title="Beyond ImageNet: other pretrained sources">
          <TheoryBlock title="" cards={[
            { icon: '💬', title: 'CLIP (OpenAI)', body: 'Pretrained on 400M image-text pairs from the web. Learns visual + language representations. Transfers to any task you can describe in text.', mono: 'image encoder + text encoder' },
            { icon: '🧬', title: 'BiT (Big Transfer)', body: 'Google\'s models pretrained on JFT-300M (300M images). Even better transfer than ImageNet pretrained models, especially for small target datasets.', mono: 'JFT-300M → fine-tune' },
            { icon: '🤗', title: 'Hugging Face', body: 'Hub of thousands of pretrained vision models (ViT, Swin, ConvNeXt). pip install transformers gives access to all.', mono: 'from transformers import ViTModel' },
          ]} />
        </DeepDive>
      </div>

      {/* 8.1.3 Feature extraction vs fine-tuning */}
      <div className={S}>
        <p className={LBL}>8.1.3 — Feature Extraction vs Fine-Tuning</p>
        <h2 className={H2}>Choosing the right strategy for your situation</h2>
        <p className={`${BODY} mb-4`}>
          The right approach depends on two axes: <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>how much data</strong> you have, and <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>how similar</strong> your task is to ImageNet. Use the selector to get a tailored recommendation:
        </p>
        <StrategySelector theme={theme} />
        <TheoryBlock title="Strategy comparison" cards={[
          { icon: '🔒', title: 'Feature extraction', body: 'base.trainable=False. Train head only. Fast (few epochs), minimal overfitting risk, works with very little data. Best when domain is similar to ImageNet.', mono: 'trainable params ≈ 0.5% of total' },
          { icon: '🔥', title: 'Fine-tuning', body: 'Unfreeze top N layers of base. Re-compile with lr ≪ original. Two-phase: warm up head first, then fine-tune. Needs more data but achieves better accuracy.', mono: 'lr: 1e-3 (head) → 1e-5 (base)' },
        ]} />
        <CodeBlock code={CODE_FEATURE_EXTRACT} />
      </div>

      {/* ── Section 8.2 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 8.2</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Fine-Tuning</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Freeze/unfreeze strategy, learning rate schedules, and handling domain shift.</p>
      </div>

      {/* 8.2.1 Freezing and unfreezing */}
      <div className={S}>
        <p className={LBL}>8.2.1 — Fine-Tuning Strategy: Freezing and Unfreezing Layers</p>
        <h2 className={H2}>A staged approach to updating pretrained weights</h2>
        <p className={`${BODY} mb-4`}>
          Fine-tuning is a two-phase process. Phase 1: freeze the entire base, train only the new head (feature extraction) for a few epochs. Phase 2: unfreeze the top N layers of the base and continue training at a much smaller learning rate.
        </p>
        <TheoryBlock title="Unfreezing strategy" cards={[
          { icon: '1️⃣', title: 'Phase 1 — Warm up head', body: 'base.trainable=False. Train new head for 5–10 epochs at lr=1e-3. Head weights converge from random to sensible values. This prevents large random gradients from destroying the base.', mono: 'base.trainable = False  lr=1e-3' },
          { icon: '2️⃣', title: 'Phase 2 — Fine-tune base', body: 'Unfreeze top layers. Re-compile with lr=1e-5. Continue training. Start from the top (most task-specific layers) and unfreeze progressively.', mono: 'base.trainable = True  lr=1e-5' },
          { icon: '📐', title: 'How many layers to unfreeze?', body: 'Start with 10–20% (top layers). If validation accuracy plateaus, unfreeze more. If it overfits, freeze more. Rule of thumb: more data → unfreeze more.', mono: 'unfreeze% ∝ dataset_size' },
        ]} />
        <Callout type="warning" title="Re-compile after changing trainable status">
          In TensorFlow/Keras, changing <span className="font-mono">layer.trainable</span> has no effect until you call <span className="font-mono">model.compile()</span> again. Forgetting this is the most common fine-tuning bug — the model keeps training only the head even though you set <span className="font-mono">base.trainable=True</span>.
        </Callout>
        <p className={`${BODY} mb-4`}>Drag the slider to control which layers are frozen (grey) and which are trainable (green):</p>
        <FreezeViz theme={theme} />
        <CodeBlock code={CODE_FINE_TUNE} />
        <DeepDive title="Batch Normalisation during fine-tuning">
          <p className={`text-sm ${BODY} mb-2`}>BatchNorm layers have two types of parameters: learnable (γ, β — scaled with backprop) and non-learnable running statistics (mean, variance — updated during forward passes). When you set <span className="font-mono">layer.trainable=False</span>, BatchNorm freezes in inference mode — running stats are NOT updated. This is usually what you want during feature extraction.</p>
          <Callout type="warning">During fine-tuning, if you unfreeze BatchNorm layers, their running statistics will be updated from your (small, potentially biased) dataset. This can hurt performance. Common practice: keep BatchNorm layers frozen even when unfreezing surrounding conv layers. Set <span className="font-mono">training=False</span> in the base model call.</Callout>
        </DeepDive>
      </div>

      {/* 8.2.2 LR schedule */}
      <div className={S}>
        <p className={LBL}>8.2.2 — Learning Rate Schedule for Fine-Tuning</p>
        <h2 className={H2}>The right learning rate at the right time</h2>
        <p className={`${BODY} mb-4`}>
          Learning rate is the most critical hyperparameter in fine-tuning. Using a schedule — rather than a constant lr — prevents early instability, ensures steady convergence, and avoids overshooting the optimum at the end.
        </p>
        <Callout type="formula" mono="fine-tuning lr ≈ 1/10 to 1/100 × original training lr">
          If the model was originally trained at lr=1e-3, use 1e-4 to 1e-5 for fine-tuning. This ensures gradient updates are small enough not to overwrite the useful pretrained representations.
        </Callout>
        <p className={`${BODY} mb-4`}>Select a schedule and see how the learning rate evolves over 150 training steps. Warmup + cosine decay is the recommended default for fine-tuning:</p>
        <LRScheduleViz theme={theme} />
        <TheoryBlock title="Schedule design choices" cards={[
          { icon: '📈', title: 'Warmup (first 2–5 epochs)', body: 'Start lr near 0, increase linearly to peak. Prevents large gradient updates from the randomly initialised head from corrupting the frozen base before it\'s stable.', mono: 'lr = peak × (step / warmup_steps)' },
          { icon: '📉', title: 'Cosine decay (main phase)', body: 'Smooth lr reduction following a cosine curve. Reaches near-zero at the end. Empirically outperforms step decay and linear decay for fine-tuning in most benchmarks.', mono: 'lr = min + 0.5(peak−min)(1+cos(π·t))' },
          { icon: '🔄', title: 'ReduceLROnPlateau', body: 'Automatically halve lr when validation loss hasn\'t improved for N epochs. Adaptive — no manual schedule design. Works well when you\'re unsure how many epochs to train.', mono: 'tf.keras.callbacks.ReduceLROnPlateau' },
        ]} />
        <CodeBlock code={CODE_LR_SCHEDULE} />
      </div>

      {/* 8.2.3 Domain shift */}
      <div className={S}>
        <p className={LBL}>8.2.3 — Domain Shift and Its Impact on Fine-Tuning</p>
        <h2 className={H2}>When the source and target distributions differ</h2>
        <p className={`${BODY} mb-4`}>
          Domain shift is the degree of statistical difference between the dataset the model was pretrained on (ImageNet: natural photographs, diverse subjects) and your target dataset. High shift means fewer features transfer usefully, requiring more extensive fine-tuning.
        </p>
        <TheoryBlock title="Types of domain shift" cards={[
          { icon: '🎨', title: 'Visual domain shift', body: 'Source: natural photographs. Target: paintings, X-rays, satellite images, microscopy. The pixel statistics and texture distributions are fundamentally different.', mono: 'P(x_source) ≠ P(x_target)' },
          { icon: '📋', title: 'Label space shift', body: 'Source: 1000 generic ImageNet classes. Target: 2 medical classes (benign/malignant). The semantic concepts are different — more fine-tuning needed for the top layers.', mono: 'class set completely replaced' },
          { icon: '📏', title: 'Data imbalance shift', body: 'Target dataset may be highly imbalanced (e.g. 95% normal, 5% defect). ImageNet is roughly balanced. Use class weights or oversampling alongside fine-tuning.', mono: 'class_weight={0:1, 1:20}' },
        ]} />
        <p className={`${BODY} mb-4`}>Click any point to see the recommended strategy for that use case — domain similarity (x-axis) vs dataset size (y-axis):</p>
        <DomainShiftQuadrant theme={theme} />
        <Callout type="analogy" title="Analogy: specialist retraining">
          A radiologist (pretrained on all medicine) switching to dermatology (target domain) needs minimal retraining — domains are similar. A mechanical engineer switching to surgery needs much more — the fundamental skills don't transfer as directly. The more different the domains, the deeper the retraining required.
        </Callout>
        <DeepDive title="Adapting to very different domains">
          <TheoryBlock title="" cards={[
            { icon: '🔄', title: 'Domain adaptation', body: 'Techniques like DANN (Domain Adversarial Neural Networks) explicitly minimise domain shift by training a domain classifier adversarially alongside the task classifier.', mono: 'adversarial domain loss' },
            { icon: '🎨', title: 'Data augmentation for shift', body: 'Artificially bridge the domain gap with aggressive augmentation: random crops, colour jitter, grayscale, Gaussian blur, CutMix. Especially useful for medical imaging.', mono: 'Albumentations / tf.image' },
            { icon: '📦', title: 'Self-supervised pretraining', body: 'If you have unlabelled target domain images, pretrain on them with contrastive learning (SimCLR, MoCo) or masked autoencoders (MAE) before supervised fine-tuning.', mono: 'SimCLR → label fine-tune' },
          ]} />
        </DeepDive>
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme === 'dark' ? 'bg-cyan-500/10 border-cyan-500/20' : 'bg-cyan-50 border-cyan-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>10 questions covering Sections 8.1 and 8.2 • +100 XP on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="transfer-learning" />
      </div>
    </motion.div>
  )
}
