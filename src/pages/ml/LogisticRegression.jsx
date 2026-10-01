import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE = `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Study hours → pass/fail
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Probability predictions
probs = model.predict_proba(X)[:, 1]  # P(pass)
preds = model.predict(X)

print(f"Accuracy     : {accuracy_score(y, preds):.0%}")
print(f"Coefficient  : {model.coef_[0][0]:.4f}")
print(f"Intercept    : {model.intercept_[0]:.4f}")

# Decision boundary: where P = 0.5
boundary = -model.intercept_[0] / model.coef_[0][0]
print(f"Boundary     : {boundary:.2f} hours")

# Predict a new student (5.5 hours)
p = model.predict_proba([[5.5]])[0][1]
print(f"P(pass|5.5h) : {p:.2%}")`

const SOFTMAX_CODE = `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Multi-class: Iris dataset (3 classes)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# One-vs-Rest strategy
ovr = LogisticRegression(multi_class='ovr', max_iter=200)
ovr.fit(X_train, y_train)
print(f"OvR  Accuracy: {ovr.score(X_test, y_test):.2%}")

# Softmax (multinomial)
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
softmax.fit(X_train, y_train)
print(f"Softmax Accuracy: {softmax.score(X_test, y_test):.2%}")

# Probability distribution from softmax
sample = X_test[0:1]
probs = softmax.predict_proba(sample)[0]
for cls, p in enumerate(probs):
    print(f"  Class {cls}: {p:.2%}")`

const QUIZ_QUESTIONS = [
  { question: 'What does the Sigmoid function output?', options: ['Any real number','A value between 0 and 1','Only 0 or 1','A negative number'], correct: 1, explanation: 'Sigmoid squashes any real z to (0,1), making it ideal for probability outputs.' },
  { question: 'Logistic Regression is primarily used for:', options: ['Predicting continuous values','Classification problems','Clustering','Dimensionality reduction'], correct: 1, explanation: 'Logistic Regression predicts the probability of class membership — it\'s a classification algorithm.' },
  { question: 'The decision boundary is where:', options: ['Loss = 0','P(y=1) = 0.5','Weight = 0','Bias = 1'], correct: 1, explanation: 'At the boundary σ(z)=0.5, meaning z=0. Points with z>0 → class 1; z<0 → class 0.' },
  { question: 'What loss function does Logistic Regression minimise?', options: ['MSE','Binary Cross-Entropy (Log Loss)','Hinge Loss','MAE'], correct: 1, explanation: 'Binary Cross-Entropy penalises confident wrong predictions heavily: −[y·log(p)+(1−y)·log(1−p)].' },
  { question: 'If σ(z) = 0.85, the model:', options: ['Is uncertain','Predicts class 1 with 85% confidence','Predicts class 0','Has an error'], correct: 1, explanation: 'σ(z)=0.85 > 0.5 → predict class 1. The model is 85% confident the sample belongs to class 1.' },
  { question: 'What is the gradient of log-loss with respect to weights?', options: ['w − α·X','Xᵀ(p − y) / n','2·w·X','y·log(p)'], correct: 1, explanation: 'The gradient of binary cross-entropy simplifies to Xᵀ(p−y)/n, making gradient descent updates clean.' },
  { question: 'What does L2 regularisation do to weights?', options: ['Zeros them out','Shrinks them toward zero','Makes them larger','Has no effect'], correct: 1, explanation: 'L2 (Ridge) adds λ·Σw² to the loss, penalising large weights and shrinking them uniformly.' },
  { question: 'Softmax outputs:', options: ['A single probability','Raw scores','A probability distribution summing to 1','Binary labels'], correct: 2, explanation: 'Softmax converts K raw scores into K probabilities that sum to exactly 1.' },
  { question: 'One-vs-Rest (OvR) trains:', options: ['One model for all classes','K binary classifiers, one per class','One model with K outputs','No classifier'], correct: 1, explanation: 'OvR trains K separate binary classifiers. Each treats one class as positive and all others as negative.' },
  { question: 'Cross-entropy loss for multi-class is:', options: ['-Σ yᵢ·log(pᵢ)','Σ(y−p)²','|y−p|','Σ pᵢ²'], correct: 0, explanation: 'Multi-class cross-entropy is −Σ yᵢ·log(pᵢ) where y is one-hot encoded and p is the softmax output.' },
]

const STUDENTS = [
  {hours:1,pass:false},{hours:2,pass:false},{hours:2.5,pass:false},
  {hours:3,pass:false},{hours:4,pass:false},{hours:5,pass:true},
  {hours:5.5,pass:true},{hours:6,pass:true},{hours:7,pass:true},{hours:8,pass:true},
]

function SigmoidCanvas({ theme }) {
  const [weight, setWeight] = useState(1)
  const [bias, setBias] = useState(0)
  const canvasRef = useRef(null)
  const W=380, H=240

  useEffect(()=>{
    const canvas=canvasRef.current; if(!canvas) return
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    const sigmoid=z=>1/(1+Math.exp(-z))
    const toY=p=>H-p*H*0.8-H*0.1
    ctx.strokeStyle=theme==='dark'?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.05)'; ctx.lineWidth=1
    for(let x=0;x<W;x+=W/6){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}
    for(const p of[0,0.25,0.5,0.75,1]){ctx.beginPath();ctx.moveTo(0,toY(p));ctx.lineTo(W,toY(p));ctx.stroke()}
    ctx.beginPath();ctx.moveTo(0,toY(0.5));ctx.lineTo(W,toY(0.5))
    ctx.strokeStyle='rgba(239,68,68,0.35)';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);ctx.stroke();ctx.setLineDash([])
    const grad=ctx.createLinearGradient(0,0,W,0); grad.addColorStop(0,'#6366F1'); grad.addColorStop(0.5,'#8B5CF6'); grad.addColorStop(1,'#06B6D4')
    ctx.beginPath()
    for(let px=0;px<=W;px++){const x=(px/W)*12-6;const p=sigmoid(weight*x+bias);if(px===0)ctx.moveTo(px,toY(p));else ctx.lineTo(px,toY(p))}
    ctx.strokeStyle=grad; ctx.lineWidth=3; ctx.stroke()
    ctx.fillStyle=theme==='dark'?'#9CA3AF':'#6B7280'; ctx.font='10px monospace'; ctx.textAlign='center'
    for(const p of[{v:1,l:'P=1.0'},{v:0.5,l:'P=0.5'},{v:0,l:'P=0.0'}])ctx.fillText(p.l,28,toY(p.v)+4)
  },[weight,bias,theme])

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>Weight (w) = <span className="text-indigo-400 font-bold">{weight.toFixed(1)}</span></label>
          <input type="range" min="-3" max="3" step="0.1" value={weight} onChange={e=>setWeight(parseFloat(e.target.value))} className="w-full accent-indigo-500"/>
          <p className={`text-xs mt-1 ${theme==='dark'?'text-gray-600':'text-gray-400'}`}>{Math.abs(weight)>1.5?'Steep S-curve (sharp decision)':'Gentle S-curve (gradual transition)'}</p>
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>Bias (b) = <span className="text-purple-400 font-bold">{bias.toFixed(1)}</span></label>
          <input type="range" min="-4" max="4" step="0.1" value={bias} onChange={e=>setBias(parseFloat(e.target.value))} className="w-full accent-purple-500"/>
          <p className={`text-xs mt-1 ${theme==='dark'?'text-gray-600':'text-gray-400'}`}>{bias>0?'Boundary shifted left (easier to predict 1)':bias<0?'Boundary shifted right (harder to predict 1)':'Boundary at x=0'}</p>
        </div>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
        style={{maxWidth:W,height:H}} />
      <p className={`text-xs mt-2 text-center font-mono ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>
        σ(z) = 1/(1+e⁻ᶻ) where z = {weight.toFixed(1)}x + {bias.toFixed(1)} &nbsp;•&nbsp; Red = decision boundary (P=0.5)
      </p>
    </div>
  )
}

function ClassificationDemo({ theme }) {
  const canvasRef = useRef(null)
  const [boundary, setBoundary] = useState(4.5)
  const W=380, H=200
  const students=STUDENTS
  const correct=students.filter(s=>(s.hours>=boundary)===s.pass).length
  const accuracy=Math.round((correct/students.length)*100)

  useEffect(()=>{
    const canvas=canvasRef.current; if(!canvas) return
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    const toX=h=>(h/10)*W; const yFail=H*0.3,yPass=H*0.7
    const bx=toX(boundary)
    ctx.fillStyle='rgba(239,68,68,0.07)'; ctx.fillRect(0,0,bx,H)
    ctx.fillStyle='rgba(16,185,129,0.07)'; ctx.fillRect(bx,0,W-bx,H)
    ctx.beginPath();ctx.moveTo(bx,0);ctx.lineTo(bx,H);ctx.strokeStyle='#6366F1';ctx.lineWidth=2.5;ctx.setLineDash([6,3]);ctx.stroke();ctx.setLineDash([])
    ctx.fillStyle='rgba(239,68,68,0.6)'; ctx.font='12px sans-serif'; ctx.fillText('Fail zone',8,20)
    ctx.fillStyle='rgba(16,185,129,0.6)'; ctx.fillText('Pass zone',bx+8,20)
    students.forEach(s=>{
      const x=toX(s.hours),y=s.pass?yPass:yFail
      const pred=s.hours>=boundary
      ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2)
      ctx.fillStyle=s.pass?'#10B981':'#EF4444'; ctx.fill()
      if(pred!==s.pass){ctx.strokeStyle='#F59E0B';ctx.lineWidth=3;ctx.stroke()}
      else{ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke()}
    })
    ctx.beginPath();ctx.moveTo(0,H-20);ctx.lineTo(W,H-20)
    ctx.strokeStyle=theme==='dark'?'rgba(255,255,255,0.1)':'rgba(0,0,0,0.1)';ctx.lineWidth=1;ctx.stroke()
    ctx.fillStyle=theme==='dark'?'#6B7280':'#9CA3AF'; ctx.font='10px sans-serif'
    for(let h=1;h<=9;h+=2)ctx.fillText(h+'h',toX(h)-6,H-6)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[boundary,theme])

  return (
    <div>
      <div className="mb-4">
        <label className={`text-xs font-medium block mb-1.5 ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>
          Decision Boundary = <span className="text-indigo-400 font-bold">{boundary.toFixed(1)} study hours</span>
        </label>
        <input type="range" min="1" max="9" step="0.5" value={boundary} onChange={e=>setBoundary(parseFloat(e.target.value))} className="w-full accent-indigo-500"/>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
        style={{maxWidth:W,height:H}} />
      <div className="flex flex-wrap gap-3 mt-3 text-xs">
        <div className={`px-3 py-1.5 rounded-lg ${theme==='dark'?'bg-slate-800':'bg-gray-100'}`}>
          <span className={theme==='dark'?'text-gray-500':'text-gray-400'}>Accuracy: </span>
          <span className={`font-bold ${accuracy>=80?'text-emerald-400':accuracy>=60?'text-yellow-400':'text-red-400'}`}>{accuracy}%</span>
        </div>
        <div className={`px-3 py-1.5 rounded-lg ${theme==='dark'?'bg-slate-800':'bg-gray-100'} ${theme==='dark'?'text-gray-400':'text-gray-500'}`}>
          🟢 Pass &nbsp; 🔴 Fail &nbsp; 🟡 Misclassified
        </div>
      </div>
    </div>
  )
}

function LogisticFlow({ theme }) {
  const steps = [
    {label:'Linear Output',icon:'📏',formula:'z = wx + b',color:'#6366F1',desc:'Compute weighted sum of inputs — same as linear regression.'},
    {label:'Sigmoid',icon:'σ',formula:'σ(z) = 1/(1+e⁻ᶻ)',color:'#8B5CF6',desc:'Squash z into a probability between 0 and 1.'},
    {label:'Probability',icon:'📊',formula:'P(y=1|x) ∈ [0,1]',color:'#06B6D4',desc:'Interpret output as the probability of belonging to class 1.'},
    {label:'Decision',icon:'✅',formula:'ŷ = 1 if P ≥ 0.5',color:'#10B981',desc:'Apply threshold to convert probability to a class label.'},
  ]
  return (
    <div className="flex flex-col items-center gap-2">
      {steps.map((step,i)=>(
        <div key={i} className="flex flex-col items-center w-full max-w-sm">
          <motion.div initial={{opacity:0,x:-20}} animate={{opacity:1,x:0}} transition={{delay:i*0.1}}
            className={`w-full px-5 py-3 rounded-xl border text-center ${theme==='dark'?'bg-slate-800/60 border-white/10':'bg-gray-50 border-gray-200'}`}>
            <div className="flex items-center justify-center gap-2 mb-1">
              <span className="text-xl">{step.icon}</span>
              <span className={`font-semibold text-sm ${theme==='dark'?'text-white':'text-gray-900'}`}>{step.label}</span>
            </div>
            <p className="font-mono text-xs mb-1" style={{color:step.color}}>{step.formula}</p>
            <p className={`text-xs ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>{step.desc}</p>
          </motion.div>
          {i<steps.length-1&&<div className="text-indigo-500 text-xl leading-none py-1">↓</div>}
        </div>
      ))}
    </div>
  )
}

// ── 5.1 Loss Visualiser ─────────────────────────────────────────────────────
function LossCanvas({ theme }) {
  const canvasRef = useRef(null)
  const [pred, setPred] = useState(0.7)
  const W = 380, H = 220

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = 40
    const plotW = W - pad - 10, plotH = H - pad - 10

    // axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.1)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad, 10); ctx.lineTo(pad, H - pad); ctx.lineTo(W - 10, H - pad); ctx.stroke()

    // grid lines
    ctx.setLineDash([3, 3])
    for (let v = 1; v <= 4; v++) {
      const gy = (H - pad) - (v / 4) * plotH
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W - 10, gy)
      ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
      ctx.stroke()
    }
    ctx.setLineDash([])

    // curves
    const drawCurve = (yTrue, color) => {
      ctx.beginPath()
      for (let px = 0; px <= plotW; px++) {
        const p = Math.max(0.001, Math.min(0.999, px / plotW))
        const loss = yTrue === 1 ? -Math.log(p) : -Math.log(1 - p)
        const clipped = Math.min(loss, 4)
        const cx = pad + px
        const cy = (H - pad) - (clipped / 4) * plotH
        px === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy)
      }
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke()
    }
    drawCurve(1, '#6366F1')
    drawCurve(0, '#EF4444')

    // vertical marker for current pred
    const mx = pad + pred * plotW
    ctx.beginPath(); ctx.moveTo(mx, 10); ctx.lineTo(mx, H - pad)
    ctx.strokeStyle = '#F59E0B'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([])

    // dots on curves
    const lossY1 = Math.min(-Math.log(pred), 4)
    const lossY0 = Math.min(-Math.log(1 - pred), 4)
    const dotY1 = (H - pad) - (lossY1 / 4) * plotH
    const dotY0 = (H - pad) - (lossY0 / 4) * plotH
    ctx.beginPath(); ctx.arc(mx, dotY1, 5, 0, Math.PI * 2); ctx.fillStyle = '#6366F1'; ctx.fill()
    ctx.beginPath(); ctx.arc(mx, dotY0, 5, 0, Math.PI * 2); ctx.fillStyle = '#EF4444'; ctx.fill()

    // axis labels
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '10px monospace'; ctx.textAlign = 'right'
    for (let v = 0; v <= 4; v++) {
      const gy = (H - pad) - (v / 4) * plotH
      ctx.fillText(v, pad - 4, gy + 3)
    }
    ctx.textAlign = 'center'
    for (const t of [0.25, 0.5, 0.75, 1.0]) {
      ctx.fillText(t.toFixed(2), pad + t * plotW, H - pad + 12)
    }
  }, [pred, theme])

  const loss1 = Math.min(-Math.log(Math.max(0.001, pred)), 10).toFixed(3)
  const loss0 = Math.min(-Math.log(Math.max(0.001, 1 - pred)), 10).toFixed(3)

  return (
    <div>
      <div className="mb-4">
        <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
          Predicted probability p = <span className="text-yellow-400 font-bold">{pred.toFixed(2)}</span>
        </label>
        <input type="range" min="0.01" max="0.99" step="0.01" value={pred} onChange={e => setPred(parseFloat(e.target.value))} className="w-full accent-yellow-500" />
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <div className="flex gap-4 mt-3 text-xs flex-wrap">
        <div className={`flex-1 px-3 py-2 rounded-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-indigo-50'}`}>
          <span className="text-indigo-400 font-semibold">y=1 loss: </span>
          <span className={`font-mono font-bold ${theme === 'dark' ? 'text-white' : 'text-indigo-700'}`}>{loss1}</span>
          <span className={`block mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>−log(p) → huge if p≈0</span>
        </div>
        <div className={`flex-1 px-3 py-2 rounded-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-red-50'}`}>
          <span className="text-red-400 font-semibold">y=0 loss: </span>
          <span className={`font-mono font-bold ${theme === 'dark' ? 'text-white' : 'text-red-700'}`}>{loss0}</span>
          <span className={`block mt-0.5 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>−log(1−p) → huge if p≈1</span>
        </div>
      </div>
    </div>
  )
}

// ── 5.1 Gradient Descent Visualiser ─────────────────────────────────────────
function GradientDescentDemo({ theme }) {
  const [lr, setLr] = useState(0.3)
  const [step, setStep] = useState(0)
  const maxSteps = 20

  // Simple 1D loss: L(w) = (w-2)^2 + 0.5  — clear bowl
  const loss = w => (w - 2) ** 2 + 0.5
  const grad = w => 2 * (w - 2)

  const history = (() => {
    const pts = [{ w: -2, l: loss(-2) }]
    let w = -2
    for (let i = 0; i < maxSteps; i++) {
      w = w - lr * grad(w)
      pts.push({ w, l: loss(w) })
    }
    return pts
  })()

  const cur = history[Math.min(step, history.length - 1)]

  const canvasRef = useRef(null)
  const W = 380, H = 200

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, W, H)
    const pad = 35
    const plotW = W - pad - 10, plotH = H - pad - 10
    const wMin = -3, wMax = 5
    const lMin = 0, lMax = 26
    const toX = w => pad + ((w - wMin) / (wMax - wMin)) * plotW
    const toY = l => (H - pad) - ((Math.min(l, lMax) - lMin) / (lMax - lMin)) * plotH

    // grid
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
    ctx.lineWidth = 1; ctx.setLineDash([3, 3])
    for (let v = 0; v <= 4; v++) {
      const gy = (H - pad) - (v / 4) * plotH
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W - 10, gy); ctx.stroke()
    }
    ctx.setLineDash([])

    // axes
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad, 10); ctx.lineTo(pad, H - pad); ctx.lineTo(W - 10, H - pad); ctx.stroke()

    // loss curve
    ctx.beginPath()
    for (let px = 0; px <= plotW; px++) {
      const w = wMin + (px / plotW) * (wMax - wMin)
      const l = loss(w)
      const x = pad + px, y = toY(l)
      px === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.strokeStyle = theme === 'dark' ? 'rgba(99,102,241,0.6)' : 'rgba(99,102,241,0.8)'
    ctx.lineWidth = 2.5; ctx.stroke()

    // path taken
    const visible = history.slice(0, step + 1)
    if (visible.length > 1) {
      ctx.beginPath()
      visible.forEach((pt, i) => {
        const x = toX(pt.w), y = toY(pt.l)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.strokeStyle = '#F59E0B'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 2]); ctx.stroke(); ctx.setLineDash([])
    }

    // current position dot
    ctx.beginPath()
    ctx.arc(toX(cur.w), toY(cur.l), 7, 0, Math.PI * 2)
    ctx.fillStyle = '#F59E0B'; ctx.fill()
    ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke()

    // minimum marker
    ctx.beginPath(); ctx.arc(toX(2), toY(loss(2)), 5, 0, Math.PI * 2)
    ctx.fillStyle = '#10B981'; ctx.fill()

    // labels
    ctx.fillStyle = theme === 'dark' ? '#6B7280' : '#9CA3AF'
    ctx.font = '10px monospace'; ctx.textAlign = 'right'
    ctx.fillText('0', pad - 4, H - pad + 3)
    ctx.fillText('w*', toX(2) + 4, toY(loss(2)) - 8)
    ctx.fillStyle = '#10B981'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center'
    ctx.fillText('minimum', toX(2), toY(loss(2)) - 8)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, lr, theme, cur])

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Learning rate α = <span className="text-yellow-400 font-bold">{lr.toFixed(2)}</span>
          </label>
          <input type="range" min="0.05" max="0.95" step="0.05" value={lr}
            onChange={e => { setLr(parseFloat(e.target.value)); setStep(0) }}
            className="w-full accent-yellow-500" />
          <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
            {lr > 0.7 ? '⚠️ Too large — may overshoot' : lr < 0.2 ? 'Very slow convergence' : 'Good learning rate'}
          </p>
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            Step = <span className="text-indigo-400 font-bold">{step}</span>
          </label>
          <input type="range" min="0" max={maxSteps} step="1" value={step}
            onChange={e => setStep(parseInt(e.target.value))}
            className="w-full accent-indigo-500" />
        </div>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-gray-50 border-gray-200'}`}
        style={{ maxWidth: W, height: H }} />
      <div className="flex gap-3 mt-3 text-xs flex-wrap">
        {[
          { label: 'w', val: cur.w.toFixed(3), color: 'text-yellow-400' },
          { label: 'loss', val: cur.l.toFixed(4), color: 'text-red-400' },
          { label: 'gradient', val: grad(cur.w).toFixed(3), color: 'text-indigo-400' },
        ].map(item => (
          <div key={item.label} className={`px-3 py-1.5 rounded-lg ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-100'}`}>
            <span className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}>{item.label}: </span>
            <span className={`font-mono font-bold ${item.color}`}>{item.val}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── 5.1 Regularisation Visualiser ────────────────────────────────────────────
function RegularisationDemo({ theme }) {
  const [lambda, setLambda] = useState(0.5)
  const [regType, setRegType] = useState('l2')

  // Illustrative: weights shrinkage as lambda increases for 4 features
  const baseWeights = [3.2, -2.1, 1.8, -0.9]
  const shrunken = baseWeights.map(w => {
    if (regType === 'l2') return w / (1 + lambda)
    // L1: soft-thresholding
    const sign = w >= 0 ? 1 : -1
    return sign * Math.max(0, Math.abs(w) - lambda * 0.5)
  })

  const features = ['Hours studied', 'Sleep quality', 'Practice tests', 'Stress level']

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
            λ (penalty) = <span className="text-cyan-400 font-bold">{lambda.toFixed(1)}</span>
          </label>
          <input type="range" min="0" max="5" step="0.1" value={lambda}
            onChange={e => setLambda(parseFloat(e.target.value))}
            className="w-full accent-cyan-500" />
        </div>
        <div className="flex gap-2 items-end pb-1">
          {['l1', 'l2'].map(t => (
            <button key={t} onClick={() => setRegType(t)}
              className={`flex-1 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                regType === t
                  ? theme === 'dark' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-indigo-100 border-indigo-300 text-indigo-700'
                  : theme === 'dark' ? 'bg-slate-800 border-white/10 text-gray-400' : 'bg-gray-100 border-gray-200 text-gray-500'
              }`}>{t.toUpperCase()}</button>
          ))}
        </div>
      </div>
      <div className="space-y-3">
        {features.map((feat, i) => {
          const orig = baseWeights[i]
          const shrunk = shrunken[i]
          const maxAbs = 3.5
          const pctOrig = (Math.abs(orig) / maxAbs) * 100
          const pctShrunk = (Math.abs(shrunk) / maxAbs) * 100
          const zeroed = Math.abs(shrunk) < 0.01
          return (
            <div key={feat}>
              <div className="flex justify-between mb-1">
                <span className={`text-xs ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>{feat}</span>
                <span className="text-xs font-mono">
                  <span className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}>{orig.toFixed(2)} → </span>
                  <span className={zeroed ? 'text-red-400 font-bold' : 'text-emerald-400 font-bold'}>{zeroed ? '0.00 (zeroed!)' : shrunk.toFixed(2)}</span>
                </span>
              </div>
              <div className={`h-4 rounded-full relative overflow-hidden ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-200'}`}>
                <div className="absolute h-full rounded-full bg-gray-400/30 transition-all duration-300" style={{ width: `${pctOrig}%` }} />
                <motion.div className={`absolute h-full rounded-full ${zeroed ? 'bg-red-500' : 'bg-emerald-500'}`}
                  animate={{ width: `${pctShrunk}%` }} transition={{ duration: 0.3 }} />
              </div>
            </div>
          )
        })}
      </div>
      <p className={`text-xs mt-3 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
        {regType === 'l1'
          ? '📌 L1 (Lasso): sparse solution — some weights go exactly to zero (feature selection)'
          : '📌 L2 (Ridge): weights shrink proportionally — none go to exactly zero'}
      </p>
    </div>
  )
}

// ── 5.2 Softmax Visualiser ────────────────────────────────────────────────────
function SoftmaxDemo({ theme }) {
  const classes = ['Cat', 'Dog', 'Bird']
  const [scores, setScores] = useState([2.0, 1.0, 0.1])

  const softmax = zs => {
    const max = Math.max(...zs)
    const exps = zs.map(z => Math.exp(z - max))
    const sum = exps.reduce((a, b) => a + b, 0)
    return exps.map(e => e / sum)
  }

  const probs = softmax(scores)
  const colors = ['#6366F1', '#10B981', '#F59E0B']

  return (
    <div>
      <p className={`text-xs mb-3 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
        Drag the raw scores (logits) — watch probabilities update in real time.
      </p>
      <div className="space-y-4">
        {classes.map((cls, i) => (
          <div key={cls}>
            <div className="flex items-center justify-between mb-1">
              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{cls}</span>
              <div className="flex items-center gap-3 text-xs font-mono">
                <span className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}>z = <span className="font-bold" style={{ color: colors[i] }}>{scores[i].toFixed(1)}</span></span>
                <span className={`font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{(probs[i] * 100).toFixed(1)}%</span>
              </div>
            </div>
            <input type="range" min="-3" max="5" step="0.1" value={scores[i]}
              onChange={e => setScores(prev => { const n = [...prev]; n[i] = parseFloat(e.target.value); return n })}
              className="w-full mb-1.5" style={{ accentColor: colors[i] }} />
            <div className={`h-3 rounded-full overflow-hidden ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-200'}`}>
              <motion.div className="h-full rounded-full" style={{ backgroundColor: colors[i] }}
                animate={{ width: `${probs[i] * 100}%` }} transition={{ duration: 0.2 }} />
            </div>
          </div>
        ))}
      </div>
      <div className={`mt-4 p-3 rounded-xl font-mono text-xs ${theme === 'dark' ? 'bg-slate-800 text-gray-300' : 'bg-gray-100 text-gray-700'}`}>
        softmax([{scores.map(s => s.toFixed(1)).join(', ')}]) = [{probs.map(p => p.toFixed(3)).join(', ')}]
        <br />sum = {probs.reduce((a, b) => a + b, 0).toFixed(6)} (always 1.0)
      </div>
    </div>
  )
}

// ── 5.2 OvR vs Softmax comparison ────────────────────────────────────────────
function OvRExplainer({ theme }) {
  const card = `rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-800/60 border-white/10' : 'bg-gray-50 border-gray-200'}`
  const h = `font-semibold text-sm mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`
  const b = `text-xs leading-relaxed ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div className={card}>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-lg">⚔️</span>
          <p className={h}>One-vs-Rest (OvR)</p>
        </div>
        <p className={b}>Trains <strong className={theme === 'dark' ? 'text-white' : 'text-gray-800'}>K separate binary classifiers</strong>. Each classifier asks: "Is this sample class k or not?" The class with the highest probability wins.</p>
        <p className={`font-mono text-xs mt-2 text-indigo-400`}>K models → predict argmax(p₁,p₂,...,pₖ)</p>
        <ul className={`text-xs mt-2 space-y-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
          <li>✅ Simple and parallelisable</li>
          <li>✅ Works with any binary classifier</li>
          <li>⚠️ Probabilities may not sum to 1</li>
        </ul>
      </div>
      <div className={card}>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-lg">🎲</span>
          <p className={h}>Softmax (Multinomial)</p>
        </div>
        <p className={b}>Trains <strong className={theme === 'dark' ? 'text-white' : 'text-gray-800'}>one joint model</strong> with K output nodes. Softmax normalises raw scores into a true probability distribution that always sums to 1.</p>
        <p className={`font-mono text-xs mt-2 text-emerald-400`}>1 model → softmax(Wx+b)</p>
        <ul className={`text-xs mt-2 space-y-1 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
          <li>✅ True probability distribution</li>
          <li>✅ Classes compete with each other</li>
          <li>⚠️ Requires multinomial solver</li>
        </ul>
      </div>
    </div>
  )
}

export default function LogisticRegression() {
  const { theme } = useApp()
  const S=`rounded-2xl border p-6 mb-6 ${theme==='dark'?'bg-slate-900/80 border-white/10':'bg-white border-gray-200'}`
  const LBL=`text-xs font-semibold uppercase tracking-wider mb-3 ${theme==='dark'?'text-purple-400':'text-purple-600'}`
  const H2=`text-xl font-bold mb-2 ${theme==='dark'?'text-white':'text-gray-900'}`
  const BODY=`text-sm leading-relaxed ${theme==='dark'?'text-gray-400':'text-gray-600'}`
  return (
    <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} className="max-w-3xl">
      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{opacity:0,scale:0.9}} animate={{opacity:1,scale:1}}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-500/20 border border-purple-500/30 mb-4">
          <span className="text-xs text-purple-400 font-medium">Machine Learning • Classification</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme==='dark'?'text-white':'text-gray-900'}`}>
          Logistic <span className="gradient-text">Regression</span>
        </h1>
        <p className={`text-lg mb-4 ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>
          Despite its name, it's a <strong className={theme==='dark'?'text-white':'text-gray-900'}>classification</strong> algorithm. It uses the Sigmoid function to output a probability between 0 and 1.
        </p>
        <Callout type="analogy" title="Why 'Regression' in the name?">
          Logistic Regression first computes a linear combination of features (like linear regression), then maps the result through the sigmoid function to get a probability. The "regression" refers to that internal linear step.
        </Callout>
      </div>

      <TheoryBlock title="Core Theory" cards={[
        { icon: 'σ', title: 'Sigmoid Function', body: 'Maps any real z to (0,1). Large positive z → close to 1. Large negative z → close to 0. The "S" shape gives smooth, continuous probability outputs.', mono: 'σ(z) = 1 / (1 + e⁻ᶻ)' },
        { icon: '✂️', title: 'Decision Boundary', body: 'The hyperplane where P(y=1) = 0.5 (i.e. z=0). Points on one side are class 1, the other class 0. For linear logistic regression, it\'s always a straight line.', mono: 'boundary: wx + b = 0' },
        { icon: '📋', title: 'Log Loss (Cross-Entropy)', body: 'Penalises confident wrong predictions exponentially. If the model predicts P=0.99 for class 0 (true label), the loss is very high.', mono: 'L = −[y·log(p)+(1−y)·log(1−p)]' },
        { icon: '📏', title: 'Linear Component', body: 'First compute z = wx + b — a weighted sum identical to linear regression. Then sigmoid transforms this to a probability.', mono: 'z = w₁x₁ + w₂x₂ + b' },
        { icon: '📊', title: 'Multi-class Extension', body: 'For more than 2 classes, use Softmax instead of Sigmoid. Softmax outputs a probability distribution summing to 1 across all classes.', mono: 'softmax(zᵢ) = eᶻⁱ / Σeᶻʲ' },
        { icon: '🔍', title: 'Regularisation', body: 'To prevent overfitting, add L1 (Lasso) or L2 (Ridge) penalty to the loss. L2 shrinks weights; L1 can zero them out entirely (feature selection).', mono: 'Loss + λ·Σw²  (L2/Ridge)' },
      ]} />

      {/* Flow */}
      <div className={S}>
        <p className={LBL}>How Logistic Regression Works</p>
        <p className={`${BODY} mb-4`}>Four steps transform raw features into a class prediction:</p>
        <LogisticFlow theme={theme} />
        <Callout type="info" title="Key insight" className="mt-4">
          The sigmoid output is a probability, not a decision. The threshold (0.5 by default) converts probability to class. You can adjust the threshold depending on the problem — e.g. lower it to 0.3 to catch more true positives in medical diagnosis.
        </Callout>
      </div>

      {/* Sigmoid Playground */}
      <div className={S}>
        <p className={LBL}>Sigmoid Function Playground</p>
        <h2 className={H2}>See how weight and bias reshape the curve</h2>
        <p className={`${BODY} mb-3`}>
          <strong className={theme==='dark'?'text-white':'text-gray-900'}>Weight</strong> controls steepness — high weight = sharp, confident decision. <strong className={theme==='dark'?'text-white':'text-gray-900'}>Bias</strong> shifts the curve left or right, moving the boundary.
        </p>
        <Callout type="warning" title="Try this">
          Set weight = 0. The curve flattens to P=0.5 everywhere — the model can't distinguish classes. This is why features need to be informative.
        </Callout>
        <SigmoidCanvas theme={theme} />
        <DeepDive title="Why not just use a step function?">
          <p className={`text-sm ${BODY} mb-2`}>A step function outputs exactly 0 or 1 — it has no gradient, so gradient descent cannot update weights. Sigmoid has a smooth, continuous gradient everywhere, enabling learning.</p>
          <Callout type="formula" mono="σ'(z) = σ(z) · (1 − σ(z))">The sigmoid derivative is always positive and reaches max at z=0. This is used in backpropagation to compute weight updates.</Callout>
        </DeepDive>
      </div>

      {/* Classification Demo */}
      <div className={S}>
        <p className={LBL}>Pass / Fail Classifier</p>
        <h2 className={H2}>Slide the decision boundary — watch predictions change</h2>
        <p className={`${BODY} mb-3`}>Students above the boundary are predicted to pass. Yellow dots are misclassifications. Notice how accuracy peaks at a specific boundary value.</p>
        <Callout type="warning" title="Try this">
          Move the boundary to 1 hour (all predicted pass). Then to 9 hours (all predicted fail). Notice accuracy drops dramatically — demonstrating why the optimal boundary matters.
        </Callout>
        <ClassificationDemo theme={theme} />
        <DeepDive title="Precision, Recall, and F1 Score">
          <p className={`text-sm ${BODY} mb-2`}>Accuracy alone can be misleading on imbalanced datasets. Use these metrics instead:</p>
          <TheoryBlock title="" cards={[
            { icon: '🎯', title: 'Precision', body: 'Of all predicted positives, how many were actually positive? High precision = few false alarms.', mono: 'TP / (TP + FP)' },
            { icon: '🔍', title: 'Recall', body: 'Of all actual positives, how many did we catch? High recall = few misses.', mono: 'TP / (TP + FN)' },
            { icon: '⚖️', title: 'F1 Score', body: 'Harmonic mean of precision and recall. Best single metric when classes are imbalanced.', mono: '2 · P·R / (P + R)' },
          ]} />
        </DeepDive>
      </div>

      {/* ── Section 5.1 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>Section 5.1</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Loss Function &amp; Optimization</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>How logistic regression learns — from log-loss derivation to regularised gradient descent.</p>
      </div>

      {/* 5.1.1 Log-loss derivation */}
      <div className={S}>
        <p className={LBL}>5.1.1 — Log-Loss (Binary Cross-Entropy)</p>
        <h2 className={H2}>Deriving the loss function</h2>
        <p className={`${BODY} mb-4`}>
          We need a loss that is large when the model is confidently wrong. Mean Squared Error doesn't work well for classification — it's non-convex with sigmoid. Log-loss is derived from the likelihood of the data.
        </p>
        <TheoryBlock title="Step-by-step derivation" cards={[
          { icon: '1️⃣', title: 'Probability model', body: 'The model predicts p = σ(z). For a positive label y=1, we want p to be large. For y=0, we want 1−p to be large.', mono: 'P(y|x) = pʸ · (1−p)¹⁻ʸ' },
          { icon: '2️⃣', title: 'Log-likelihood', body: 'Take the log of the joint likelihood over all n samples. Logs turn products into sums and prevent numeric underflow.', mono: 'ℓ = Σ [y·log(p) + (1−y)·log(1−p)]' },
          { icon: '3️⃣', title: 'Negate to minimise', body: 'Maximising log-likelihood = minimising its negation. Dividing by n gives the average loss per sample.', mono: 'L = −(1/n)·Σ [y·log(p)+(1−y)·log(1−p)]' },
        ]} />
        <Callout type="formula" mono="L = −[y·log(p) + (1−y)·log(1−p)]">
          When y=1: loss = −log(p). As p → 0 (wrong and confident), loss → ∞. When y=0: loss = −log(1−p). As p → 1, loss → ∞. This asymmetric heavy penalty is exactly what we want.
        </Callout>
        <p className={`${BODY} mb-4`}>Use the interactive visualiser below to see exactly how the loss explodes when the model makes a confident wrong prediction:</p>
        <LossCanvas theme={theme} />
        <DeepDive title="Why not MSE for classification?">
          <p className={`text-sm ${BODY} mb-2`}>When MSE is composed with the sigmoid, the resulting loss surface is <strong className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>non-convex</strong> — it has multiple local minima making gradient descent unreliable. Binary cross-entropy composed with sigmoid is always convex, guaranteeing a unique global minimum.</p>
          <Callout type="info">Log-loss also has a clean gradient: ∂L/∂w = (1/n)·Xᵀ(p−y). No chain rule complexity — sigmoid's derivative cancels beautifully.</Callout>
        </DeepDive>
      </div>

      {/* 5.1.2 MLE */}
      <div className={S}>
        <p className={LBL}>5.1.2 — Maximum Likelihood Estimation (MLE)</p>
        <h2 className={H2}>Why MLE gives us log-loss</h2>
        <p className={`${BODY} mb-4`}>
          MLE asks: <em>"What parameters θ make the observed data most probable?"</em> For logistic regression, maximising the likelihood is exactly equivalent to minimising binary cross-entropy.
        </p>
        <TheoryBlock title="MLE for logistic regression" cards={[
          { icon: '🎯', title: 'Likelihood function', body: 'Assuming samples are independent, the joint probability of observing all labels y given inputs X is the product of individual probabilities.', mono: 'L(θ) = Π P(yᵢ|xᵢ;θ)' },
          { icon: '📉', title: 'Log-likelihood', body: 'Taking logs converts the product to a sum and avoids numerical underflow with many samples. We then negate it to turn maximisation into minimisation.', mono: 'log L(θ) = Σ [yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ)]' },
          { icon: '🔗', title: 'Connection to cross-entropy', body: 'The negative log-likelihood per sample IS binary cross-entropy. Minimising cross-entropy = maximising likelihood. They are two names for the same thing.', mono: 'min NLL = min Cross-Entropy' },
        ]} />
        <Callout type="analogy" title="Intuition: coin-flip analogy">
          Imagine each prediction is a biased coin. MLE finds the bias (model weights) that would make the sequence of observed heads/tails (labels) most likely. The more confident and correct you are, the higher the likelihood.
        </Callout>
      </div>

      {/* 5.1.3 Gradient Descent */}
      <div className={S}>
        <p className={LBL}>5.1.3 — Gradient Descent for Logistic Regression</p>
        <h2 className={H2}>Walking down the loss surface</h2>
        <p className={`${BODY} mb-4`}>
          Logistic regression has no closed-form solution (unlike linear regression). We use gradient descent to iteratively nudge weights in the direction that reduces the loss.
        </p>
        <TheoryBlock title="Update rule" cards={[
          { icon: '∂', title: 'Gradient of loss', body: 'Differentiate the cross-entropy loss w.r.t. weights. The sigmoid derivative cancels cleanly, giving a simple residual form.', mono: '∂L/∂w = (1/n) Xᵀ(p − y)' },
          { icon: '⬇️', title: 'Weight update', body: 'Subtract a fraction α of the gradient from current weights. Repeat until convergence (loss stops decreasing).', mono: 'w ← w − α · (1/n) Xᵀ(p − y)' },
          { icon: '🔄', title: 'Bias update', body: 'Bias has its own gradient — the mean residual. It shifts the decision boundary without affecting the slope.', mono: 'b ← b − α · (1/n) Σ(p − y)' },
        ]} />
        <Callout type="warning" title="Learning rate matters">
          Too large → overshoots the minimum (loss oscillates or diverges). Too small → takes thousands of steps to converge. Typical starting values: 0.01–0.3.
        </Callout>
        <p className={`${BODY} mb-4`}>Step through gradient descent on a toy loss surface to see how the learning rate affects convergence:</p>
        <GradientDescentDemo theme={theme} />
        <DeepDive title="Variants: SGD, Mini-batch, Adam">
          <TheoryBlock title="" cards={[
            { icon: '🎲', title: 'Stochastic GD (SGD)', body: 'Update weights after each single sample. Very noisy but can escape local minima. Used for large datasets.', mono: 'w ← w − α·∇Lᵢ (one sample)' },
            { icon: '📦', title: 'Mini-batch GD', body: 'Update using a small batch (32–256 samples). Balances noise of SGD with efficiency of full-batch. Most common in practice.', mono: 'w ← w − α·∇L_batch' },
            { icon: '⚡', title: 'Adam optimiser', body: 'Adapts the learning rate per parameter using first and second moment estimates. Converges faster and is more robust to learning rate choice.', mono: 'm̂ₜ / (√v̂ₜ + ε)' },
          ]} />
        </DeepDive>
      </div>

      {/* 5.1.4 Regularisation */}
      <div className={S}>
        <p className={LBL}>5.1.4 — Regularised Logistic Regression</p>
        <h2 className={H2}>L1 and L2 penalties to prevent overfitting</h2>
        <p className={`${BODY} mb-4`}>
          Without regularisation, logistic regression can overfit — especially with many features or correlated inputs. We add a penalty term to the loss that discourages large weights.
        </p>
        <TheoryBlock title="Regularisation types" cards={[
          { icon: '🔵', title: 'L2 Ridge', body: 'Adds sum of squared weights to the loss. Shrinks all weights proportionally toward zero — none become exactly zero. Prefers solutions with many small weights.', mono: 'L_total = L + λ·Σwᵢ²' },
          { icon: '🔶', title: 'L1 Lasso', body: 'Adds sum of absolute weights. Produces sparse solutions — some weights go exactly to zero, effectively performing feature selection.', mono: 'L_total = L + λ·Σ|wᵢ|' },
          { icon: '🔷', title: 'Elastic Net', body: 'Combines L1 and L2. Useful when you want sparsity (L1 benefit) while keeping some stability for correlated features (L2 benefit).', mono: 'L + α·λ·Σ|w| + (1−α)·λ·Σw²' },
        ]} />
        <Callout type="info" title="λ controls the trade-off">
          λ=0 → no regularisation (pure fit). λ→∞ → all weights → 0 (underfit). Choose λ via cross-validation. In scikit-learn, C = 1/λ (default C=1.0).
        </Callout>
        <p className={`${BODY} mb-4`}>Increase λ below to see how L1 zeros out features while L2 shrinks them uniformly:</p>
        <RegularisationDemo theme={theme} />
        <DeepDive title="Regularised gradient update">
          <p className={`text-sm ${BODY} mb-2`}>Regularisation changes the gradient update. For L2, the update becomes:</p>
          <Callout type="formula" mono="w ← w·(1 − α·λ) − α·(1/n)·Xᵀ(p−y)">
            The term (1 − α·λ) is called weight decay — it multiplicatively shrinks weights before each step, preventing them from growing too large.
          </Callout>
          <p className={`text-sm ${BODY} mt-2`}>For L1, the gradient of |w| is the sign function: ∂|w|/∂w = sign(w). The update subtracts a constant in the direction of each weight's sign, which is what drives small weights to exactly zero.</p>
        </DeepDive>
      </div>

      {/* ── Section 5.2 ── */}
      <div className={`rounded-2xl border px-6 pt-5 pb-2 mb-2 ${theme === 'dark' ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-emerald-50 border-emerald-200'}`}>
        <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600'}`}>Section 5.2</p>
        <h2 className={`text-2xl font-bold mb-1 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Multi-Class Classification</h2>
        <p className={`text-sm mb-4 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>Extending logistic regression beyond two classes — OvR, softmax, and their respective loss functions.</p>
      </div>

      {/* 5.2.1 OvR */}
      <div className={S}>
        <p className={LBL}>5.2.1 — One-vs-Rest (OvR) Strategy</p>
        <h2 className={H2}>K binary classifiers, one per class</h2>
        <p className={`${BODY} mb-4`}>
          The simplest way to handle K classes with a binary classifier: train K separate models. Model k treats class k as positive and all other classes as negative. At prediction time, pick the class whose model fires with the highest probability.
        </p>
        <OvRExplainer theme={theme} />
        <Callout type="analogy" title="Analogy: tournament brackets">
          Imagine a round-robin tournament where each team (class) plays against everyone else (all other classes). The team that wins most matches (highest binary probability) is the overall champion.
        </Callout>
        <DeepDive title="When does OvR fail?">
          <p className={`text-sm ${BODY} mb-2`}>OvR can produce ambiguous predictions when two or more classifiers both fire with high confidence, or when none do. Since classifiers are trained independently, their probability outputs are not calibrated to sum to 1 — you might get [0.9, 0.85, 0.7] for three classes, which is inconsistent.</p>
          <Callout type="warning">For well-calibrated probability estimates across classes, use Softmax (multinomial) instead.</Callout>
        </DeepDive>
      </div>

      {/* 5.2.2 Softmax */}
      <div className={S}>
        <p className={LBL}>5.2.2 — Softmax Regression (Multinomial Logistic Regression)</p>
        <h2 className={H2}>One joint model with a probability distribution</h2>
        <p className={`${BODY} mb-4`}>
          Softmax generalises the sigmoid to K classes. Instead of one weight vector, we have K weight vectors (one per class). The softmax function converts K raw scores (logits) into a proper probability distribution.
        </p>
        <TheoryBlock title="Softmax mechanics" cards={[
          { icon: '📐', title: 'Raw scores (logits)', body: 'Compute one linear score per class: zₖ = wₖᵀx + bₖ. These can be any real number — positive or negative.', mono: 'zₖ = wₖᵀx + bₖ  for k=1..K' },
          { icon: 'Σ', title: 'Softmax normalisation', body: 'Exponentiate each score and divide by the total. This ensures outputs are positive and sum to exactly 1. Larger scores → higher probability.', mono: 'p(y=k|x) = e^zₖ / Σⱼ e^zⱼ' },
          { icon: '🏆', title: 'Prediction', body: 'Predict the class with the highest softmax probability. Equivalent to the argmax of the raw logits — softmax doesn\'t change the ordering.', mono: 'ŷ = argmax_k p(y=k|x)' },
        ]} />
        <Callout type="info" title="Numeric stability trick">
          Computing e^z can overflow for large z. In practice, subtract the max logit first: e^(z−max) / Σ e^(zⱼ−max). This is mathematically equivalent but numerically stable.
        </Callout>
        <p className={`${BODY} mb-4`}>Drag the logits below to see how softmax converts raw scores into probabilities:</p>
        <SoftmaxDemo theme={theme} />
      </div>

      {/* 5.2.3 Cross-entropy for multi-class */}
      <div className={S}>
        <p className={LBL}>5.2.3 — Cross-Entropy Loss for Multi-Class Problems</p>
        <h2 className={H2}>Extending log-loss to K classes</h2>
        <p className={`${BODY} mb-4`}>
          Binary cross-entropy has one probability p. Multi-class cross-entropy sums over all K class probabilities, but with a one-hot label vector y that is 1 for the true class and 0 elsewhere — so only one term survives in the sum.
        </p>
        <TheoryBlock title="Multi-class cross-entropy" cards={[
          { icon: '🏷️', title: 'One-hot encoding', body: 'The true label y is a K-dimensional vector with a 1 in the true class position and 0s elsewhere. This enables vectorised loss computation.', mono: 'y = [0, 0, 1, 0]  (class 3 of 4)' },
          { icon: '📉', title: 'Loss formula', body: 'Only the log-probability of the true class contributes to the loss. Wrong class probabilities are implicitly penalised via the softmax denominator.', mono: 'L = −Σₖ yₖ · log(pₖ) = −log(p_true)' },
          { icon: '🔢', title: 'Batch loss', body: 'Average over the n training examples. Each sample only contributes the log-probability it assigned to its true class.', mono: 'L_batch = −(1/n) Σᵢ log(p_i[y_i])' },
        ]} />
        <Callout type="formula" mono="L = −log(p_correct_class)">
          The loss is simply the negative log of the probability the model assigned to the right answer. If the model says 95% → loss = −log(0.95) = 0.051. If it says 5% → loss = −log(0.05) = 3.0. Huge difference — exactly the right incentive.
        </Callout>
        <DeepDive title="Why cross-entropy, not MSE, for classification?">
          <p className={`text-sm ${BODY} mb-2`}>MSE between a one-hot vector and softmax probabilities is non-convex and has vanishing gradients when the model is confidently wrong. Cross-entropy with softmax has clean gradients that stay large even for wrong confident predictions:</p>
          <Callout type="formula" mono="∂L/∂zₖ = pₖ − yₖ">
            The gradient is the prediction error for each class. When the model predicts class k with probability 0.9 but the true label is 0, the gradient is 0.9 — a strong signal to correct. This is mathematically identical to the binary case.
          </Callout>
        </DeepDive>
      </div>

      {/* 5.2.4 Gradient descent for softmax */}
      <div className={S}>
        <p className={LBL}>5.2.4 — Gradient Descent for Softmax Regression</p>
        <h2 className={H2}>Learning K weight vectors simultaneously</h2>
        <p className={`${BODY} mb-4`}>
          Softmax regression trains K weight vectors simultaneously. The gradient for each class k's weights depends on the residual between the predicted probability for class k and the true one-hot label for that class.
        </p>
        <TheoryBlock title="Softmax gradient descent" cards={[
          { icon: '∂', title: 'Gradient per class', body: 'For each class k, the gradient of the loss w.r.t. wₖ is the input features scaled by the prediction error for that class, averaged over the batch.', mono: '∂L/∂wₖ = (1/n) Xᵀ(pₖ − yₖ)' },
          { icon: '⬇️', title: 'Weight update (all classes)', body: 'Update all K weight vectors simultaneously. Classes that are over-predicted get their weights reduced; under-predicted classes get a boost.', mono: 'wₖ ← wₖ − α · (1/n) Xᵀ(pₖ − yₖ)' },
          { icon: '🤝', title: 'Classes compete', body: 'Softmax is a competitive normalisation — boosting one class necessarily reduces others. The gradients are coupled through the shared denominator.', mono: 'Σₖ pₖ = 1  (always)' },
        ]} />
        <Callout type="success" title="Gradient simplicity">
          Despite the complex softmax function, the gradient of cross-entropy loss w.r.t. the logits is simply (p − y) — the vector of prediction errors. This elegant result holds because softmax and cross-entropy are designed as a matched pair.
        </Callout>
        <DeepDive title="Redundancy in softmax parameters">
          <p className={`text-sm ${BODY} mb-2`}>Softmax has a redundancy: adding any constant vector c to all logits doesn't change the output (the constant cancels in numerator and denominator). This means the solution isn't unique — we can always subtract one class's weights (fix wₖ=0 for some k) without loss of expressiveness. This is why some implementations use K−1 weight vectors instead of K.</p>
          <Callout type="info">scikit-learn handles this automatically when using multi_class='multinomial'. The L2 regularisation also breaks the symmetry and pins the solution to the minimum-norm one.</Callout>
        </DeepDive>
      </div>

      {/* Code */}
      <div className={S}>
        <p className={LBL}>Python Example</p>
        <h2 className={H2}>Binary logistic regression — full workflow</h2>
        <p className={`${BODY} mb-3`}>Trains a logistic regression model and computes accuracy, the decision boundary, and probability for a new student.</p>
        <CodeBlock code={PYTHON_CODE} />
      </div>

      <div className={S}>
        <p className={LBL}>Python Example — Multi-Class</p>
        <h2 className={H2}>OvR vs Softmax on the Iris dataset</h2>
        <p className={`${BODY} mb-3`}>Compares One-vs-Rest and multinomial softmax on a 3-class classification problem.</p>
        <CodeBlock code={SOFTMAX_CODE} />
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme==='dark'?'bg-purple-500/10 border-purple-500/20':'bg-purple-50 border-purple-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme==='dark'?'text-white':'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>10 questions covering all sections • +100 XP on completion</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="logistic-regression" />
      </div>
    </motion.div>
  )
}
