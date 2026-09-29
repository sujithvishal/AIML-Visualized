import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE = `from sklearn.linear_model import Perceptron
import numpy as np

# AND gate — linearly separable ✓
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0, 0, 0, 1])

model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_and, y_and)
print("AND gate predictions:", model.predict(X_and))
# → [0 0 0 1] ✓

# XOR gate — NOT linearly separable ✗
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0, 1, 1, 0])

model_xor = Perceptron(max_iter=1000)
model_xor.fit(X_xor, y_xor)
print("XOR predictions:    ", model_xor.predict(X_xor))
# → Wrong! A single Perceptron cannot solve XOR`

const QUIZ_QUESTIONS = [
  { question: 'What is a Perceptron?', options: ['A multi-layer neural network','The simplest artificial neuron — makes binary decisions','A data preprocessing technique','A convolutional layer'], correct: 1, explanation: 'A Perceptron is the simplest artificial neuron — it takes weighted inputs, sums them with bias, and outputs a binary decision.' },
  { question: 'The weighted sum x₁w₁ + x₂w₂ + b is called:', options: ['The output','The pre-activation value z','The learning rate','The loss'], correct: 1, explanation: 'z = Σ(xᵢwᵢ) + b is the pre-activation (linear) value. The activation function then transforms z into an output.' },
  { question: 'A single Perceptron can solve XOR:', options: ['True','False — only linearly separable problems','Only with 3 inputs','Only with large datasets'], correct: 1, explanation: 'XOR is not linearly separable — you cannot draw one straight line to separate the classes. An MLP with hidden layers is required.' },
  { question: 'The step activation function outputs:', options: ['A continuous probability','1 if z ≥ 0, else 0','Always 0.5','The gradient'], correct: 1, explanation: 'The step function is the original Perceptron activation — output 1 if the weighted sum ≥ 0, else 0.' },
  { question: 'Increasing weight w₁ makes:', options: ['No difference','Feature x₁ more influential in the decision','The bias larger','x₁ irrelevant'], correct: 1, explanation: 'A larger weight means that input feature has more influence on the final output decision.' },
]

function SliderRow({ label, val, set, min, max, color }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-mono w-16 shrink-0" style={{color}}>
        {label} = <span className="font-bold">{val.toFixed(1)}</span>
      </span>
      <input type="range" min={min} max={max} step="0.1" value={val}
        onChange={e=>set(parseFloat(e.target.value))}
        className="flex-1" style={{accentColor:color}} />
    </div>
  )
}

function PerceptronViz({ theme }) {
  const [x1,setX1]=useState(0.6),[x2,setX2]=useState(0.7),[w1,setW1]=useState(0.8),[w2,setW2]=useState(0.5),[b,setB]=useState(-0.3)
  const canvasRef=useRef(null)
  const W=380,H=220
  const z=x1*w1+x2*w2+b
  const out=z>=0?1:0
  const sig=1/(1+Math.exp(-z))

  useEffect(()=>{
    const canvas=canvasRef.current; if(!canvas) return
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    const cx=W*0.55,cy=H*0.5,r=35
    const inputs=[{x:W*0.08,y:H*0.3,label:'x₁',val:x1,w:w1,color:'#6366F1'},{x:W*0.08,y:H*0.7,label:'x₂',val:x2,w:w2,color:'#8B5CF6'}]
    inputs.forEach(inp=>{
      const grad=ctx.createLinearGradient(inp.x,inp.y,cx-r,cy); grad.addColorStop(0,inp.color); grad.addColorStop(1,'rgba(99,102,241,0.3)')
      ctx.beginPath(); ctx.moveTo(inp.x+20,inp.y); ctx.lineTo(cx-r,cy)
      ctx.strokeStyle=Math.abs(inp.val*inp.w)>0.1?grad:theme==='dark'?'rgba(255,255,255,0.1)':'rgba(0,0,0,0.1)'
      ctx.lineWidth=Math.abs(inp.val*inp.w)*4+0.5; ctx.stroke()
      ctx.fillStyle=inp.color; ctx.font='bold 11px monospace'; ctx.textAlign='left'
      ctx.fillText(`w=${inp.w.toFixed(1)}`,(inp.x+cx)/2-10,(inp.y+cy)/2)
    })
    ctx.beginPath(); ctx.moveTo(cx,cy-r-30); ctx.lineTo(cx,cy-r-5); ctx.strokeStyle='#06B6D4'; ctx.lineWidth=2; ctx.stroke()
    ctx.fillStyle='#06B6D4'; ctx.font='11px monospace'; ctx.textAlign='center'; ctx.fillText(`b=${b.toFixed(1)}`,cx,cy-r-35)
    const glow=ctx.createRadialGradient(cx,cy,0,cx,cy,r+10); glow.addColorStop(0,out===1?'rgba(99,102,241,0.3)':'rgba(0,0,0,0)'); glow.addColorStop(1,'rgba(0,0,0,0)')
    ctx.beginPath(); ctx.arc(cx,cy,r+10,0,Math.PI*2); ctx.fillStyle=glow; ctx.fill()
    ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.fillStyle=out===1?'#6366F1':theme==='dark'?'#1E293B':'#F1F5F9'; ctx.fill()
    ctx.strokeStyle=out===1?'#818CF8':theme==='dark'?'#334155':'#CBD5E1'; ctx.lineWidth=2.5; ctx.stroke()
    ctx.fillStyle='white'; ctx.font='bold 12px monospace'; ctx.textAlign='center'; ctx.fillText('σ',cx,cy-6); ctx.fillText(sig.toFixed(2),cx,cy+10)
    const sx=cx+r+20,sy=cy-25,sw=60,sh=50
    ctx.fillStyle=theme==='dark'?'#1E293B':'#F8FAFC'; ctx.strokeStyle=theme==='dark'?'#334155':'#CBD5E1'; ctx.lineWidth=1.5
    ctx.beginPath(); ctx.roundRect(sx,sy,sw,sh,8); ctx.fill(); ctx.stroke()
    ctx.fillStyle=theme==='dark'?'#9CA3AF':'#6B7280'; ctx.font='10px monospace'; ctx.textAlign='center'; ctx.fillText('step',sx+sw/2,sy+14)
    ctx.fillStyle=out===1?'#10B981':'#EF4444'; ctx.font='bold 18px monospace'; ctx.fillText(out===1?'1':'0',sx+sw/2,sy+38)
    ctx.beginPath(); ctx.moveTo(sx+sw,cy); ctx.lineTo(W-15,cy); ctx.strokeStyle=out===1?'#10B981':'#EF4444'; ctx.lineWidth=2; ctx.stroke()
    inputs.forEach(inp=>{
      ctx.beginPath(); ctx.arc(inp.x,inp.y,20,0,Math.PI*2); ctx.fillStyle=inp.color; ctx.fill()
      ctx.strokeStyle='white'; ctx.lineWidth=1.5; ctx.stroke()
      ctx.fillStyle='white'; ctx.font='bold 11px monospace'; ctx.textAlign='center'
      ctx.fillText(inp.label,inp.x,inp.y-4); ctx.fillText(inp.val.toFixed(1),inp.x,inp.y+10)
    })
    ctx.textAlign='left'
  },[x1,x2,w1,w2,b,theme,out,sig])

  return (
    <div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border mb-4 ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
        style={{maxWidth:W,height:H}} />
      <div className={`p-4 rounded-xl border space-y-3 ${theme==='dark'?'bg-slate-800/60 border-white/10':'bg-gray-50 border-gray-200'}`}>
        <SliderRow label="x₁" val={x1} set={setX1} min={0} max={1} color="#6366F1" />
        <SliderRow label="x₂" val={x2} set={setX2} min={0} max={1} color="#8B5CF6" />
        <SliderRow label="w₁" val={w1} set={setW1} min={-2} max={2} color="#6366F1" />
        <SliderRow label="w₂" val={w2} set={setW2} min={-2} max={2} color="#8B5CF6" />
        <SliderRow label="b"  val={b}  set={setB}  min={-2} max={2} color="#06B6D4" />
      </div>
      <div className={`mt-3 p-3 rounded-xl border font-mono text-sm ${theme==='dark'?'bg-slate-800/60 border-white/10 text-gray-300':'bg-gray-50 border-gray-200 text-gray-700'}`}>
        z = {x1.toFixed(1)}×{w1.toFixed(1)} + {x2.toFixed(1)}×{w2.toFixed(1)} + ({b.toFixed(1)}) =
        <span className={`font-bold ml-1 ${z>=0?'text-indigo-400':'text-red-400'}`}>{z.toFixed(3)}</span>
        &nbsp;→ output:
        <span className={`font-bold ml-1 ${out===1?'text-emerald-400':'text-red-400'}`}>{out}</span>
      </div>
    </div>
  )
}

function PerceptronBoundary({ theme }) {
  const canvasRef=useRef(null)
  const [points,setPoints]=useState([{x:60,y:60,cls:1},{x:100,y:80,cls:1},{x:80,y:110,cls:1},{x:200,y:200,cls:0},{x:250,y:220,cls:0},{x:220,y:180,cls:0}])
  const [activeClass,setActiveClass]=useState(0)
  const W=340,H=240

  const draw=()=>{
    const canvas=canvasRef.current; if(!canvas) return
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    ctx.strokeStyle=theme==='dark'?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.04)'; ctx.lineWidth=1
    for(let i=0;i<=W;i+=30){ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,H);ctx.stroke()}
    for(let i=0;i<=H;i+=30){ctx.beginPath();ctx.moveTo(0,i);ctx.lineTo(W,i);ctx.stroke()}
    const cls0=points.filter(p=>p.cls===0),cls1=points.filter(p=>p.cls===1)
    if(cls0.length&&cls1.length){
      const m0x=cls0.reduce((s,p)=>s+p.x,0)/cls0.length,m0y=cls0.reduce((s,p)=>s+p.y,0)/cls0.length
      const m1x=cls1.reduce((s,p)=>s+p.x,0)/cls1.length,m1y=cls1.reduce((s,p)=>s+p.y,0)/cls1.length
      const mx=(m0x+m1x)/2,my=(m0y+m1y)/2,dx=m1x-m0x,dy=m1y-m0y,len=Math.hypot(dx,dy)
      if(len>0){const nx=-dy/len*W,ny=dx/len*W; ctx.beginPath();ctx.moveTo(mx-nx,my-ny);ctx.lineTo(mx+nx,my+ny);ctx.strokeStyle='#6366F1';ctx.lineWidth=2.5;ctx.setLineDash([6,4]);ctx.stroke();ctx.setLineDash([])}
    }
    points.forEach(p=>{ctx.beginPath();ctx.arc(p.x,p.y,8,0,Math.PI*2);ctx.fillStyle=p.cls===1?'#6366F1':'#06B6D4';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke()})
  }

  useEffect(draw,[points,theme])

  const handleClick=e=>{const rect=canvasRef.current.getBoundingClientRect();const sx=W/rect.width,sy=H/rect.height;setPoints(p=>[...p,{x:(e.clientX-rect.left)*sx,y:(e.clientY-rect.top)*sy,cls:activeClass}])}

  return (
    <div>
      <div className="flex gap-2 mb-3">
        {[0,1].map(c=>(
          <button key={c} onClick={()=>setActiveClass(c)}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-all ${activeClass===c?c===0?'bg-cyan-500 text-white':'bg-indigo-500 text-white':theme==='dark'?'bg-slate-800 text-gray-400':'bg-gray-100 text-gray-500'}`}>
            ● Class {c}
          </button>
        ))}
        <button onClick={()=>setPoints([])} className={`ml-auto px-3 py-1 rounded-lg text-xs ${theme==='dark'?'bg-slate-800 text-gray-400':'bg-gray-100 text-gray-500'}`}>Clear</button>
      </div>
      <canvas ref={canvasRef} width={W} height={H} onClick={handleClick}
        className={`w-full rounded-xl border cursor-crosshair ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
        style={{maxWidth:W,height:H}} />
      <p className={`text-xs mt-1.5 ${theme==='dark'?'text-gray-600':'text-gray-400'}`}>
        Click to add points • The Perceptron can only draw one straight boundary line
      </p>
    </div>
  )
}

export default function Perceptron() {
  const { theme } = useApp()
  const S=`rounded-2xl border p-6 mb-6 ${theme==='dark'?'bg-slate-900/80 border-white/10':'bg-white border-gray-200'}`
  const LBL=`text-xs font-semibold uppercase tracking-wider mb-3 ${theme==='dark'?'text-cyan-400':'text-cyan-600'}`
  const H2=`text-xl font-bold mb-2 ${theme==='dark'?'text-white':'text-gray-900'}`
  const BODY=`text-sm leading-relaxed ${theme==='dark'?'text-gray-400':'text-gray-600'}`

  return (
    <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} className="max-w-3xl">
      <div className="mb-8">
        <motion.div initial={{opacity:0,scale:0.9}} animate={{opacity:1,scale:1}}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/20 border border-cyan-500/30 mb-4">
          <span className="text-xs text-cyan-400 font-medium">Deep Neural Networks • Building Block</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme==='dark'?'text-white':'text-gray-900'}`}>
          The <span className="gradient-text">Perceptron</span>
        </h1>
        <p className={`text-lg mb-4 ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>
          The simplest artificial neuron — the <strong className={theme==='dark'?'text-white':'text-gray-900'}>fundamental building block</strong> of all neural networks, invented by Frank Rosenblatt in 1957.
        </p>
        <Callout type="analogy" title="Real-world analogy">
          Imagine a judge who weighs evidence (inputs) by importance (weights), adds up the total, and gives a yes/no verdict. The Perceptron does exactly that — weighted evidence → threshold decision.
        </Callout>
      </div>

      <TheoryBlock title="Core Theory" cards={[
        { icon: '⚖️', title: 'Weighted Sum', body: 'Each input xᵢ is multiplied by weight wᵢ. Higher weight = more important feature. Sum all products, add bias b to get z.', mono: 'z = x₁w₁ + x₂w₂ + b' },
        { icon: '🚦', title: 'Step Activation', body: 'If z ≥ 0, the neuron "fires" (output = 1). Otherwise silent (output = 0). This is the simplest activation function — a hard threshold.', mono: 'out = 1 if z ≥ 0, else 0' },
        { icon: '📚', title: 'Learning Rule', body: 'Weights update only when the Perceptron misclassifies a point. Correct prediction → no update. Wrong prediction → adjust weights toward correctness.', mono: 'wᵢ := wᵢ + α·(y−ŷ)·xᵢ' },
        { icon: '⚠️', title: 'Linear Limitation', body: 'A single Perceptron can only learn linearly separable problems — those separable by one straight line (or hyperplane in higher dimensions).', mono: 'cannot solve XOR' },
        { icon: '🔢', title: 'Bias Term', body: 'The bias b shifts the activation threshold. Without bias, the decision boundary must pass through the origin — very restrictive.', mono: 'z = wx + b  (b shifts boundary)' },
        { icon: '🏗️', title: 'Historical Impact', body: 'The Perceptron sparked enormous excitement in the 1950s–60s as the first trainable computing unit. Minsky & Papert\'s 1969 book showed its limits, leading to the first "AI winter".', },
      ]} />

      {/* Interactive Viz */}
      <div className={S}>
        <p className={LBL}>Interactive Perceptron</p>
        <h2 className={H2}>All 5 parameters — live output</h2>
        <p className={`${BODY} mb-3`}>Adjust inputs, weights, and bias. The neuron visualisation shows connection weights as line thickness. The output box shows the step function decision.</p>
        <Callout type="info" title="Try this">
          Set both weights to 0. The output becomes entirely determined by the bias. This is why bias matters — it gives the model a "default" activation level independent of inputs.
        </Callout>
        <PerceptronViz theme={theme} />
        <DeepDive title="Why don't modern networks use the step function?">
          <p className={`text-sm ${BODY} mb-2`}>The step function has a gradient of 0 everywhere (except at z=0 where it's undefined). That means backpropagation cannot compute useful weight updates — the gradient vanishes.</p>
          <Callout type="formula" mono="ReLU(z) = max(0, z)   →   gradient = 1 if z>0, else 0">Modern networks use ReLU which has a non-zero gradient for positive z, enabling backpropagation to work.</Callout>
        </DeepDive>
      </div>

      {/* Decision Boundary */}
      <div className={S}>
        <p className={LBL}>Linear Decision Boundary</p>
        <h2 className={H2}>Place points — see the limitation</h2>
        <p className={`${BODY} mb-3`}>
          The Perceptron draws the best straight line separating two classes. Works perfectly for linearly separable data — fails completely for anything more complex.
        </p>
        <Callout type="warning" title="The XOR problem">
          Try placing: (0,0)=class0, (1,1)=class0 in top-left and bottom-right; (1,0)=class1, (0,1)=class1 in top-right and bottom-left. No single straight line can separate them — this is XOR, and it's why we need multi-layer networks.
        </Callout>
        <PerceptronBoundary theme={theme} />
      </div>

      {/* Logic Gates */}
      <div className={S}>
        <p className={LBL}>Logic Gate Examples</p>
        <p className={`${BODY} mb-3`}>Logic gates are the classic test cases for Perceptrons. AND and OR are linearly separable; XOR is not.</p>
        <div className="grid grid-cols-3 gap-3">
          {[
            {gate:'AND',rows:[[0,0,0],[0,1,0],[1,0,0],[1,1,1]],ok:true},
            {gate:'OR', rows:[[0,0,0],[0,1,1],[1,0,1],[1,1,1]],ok:true},
            {gate:'XOR',rows:[[0,0,0],[0,1,1],[1,0,1],[1,1,0]],ok:false},
          ].map(({gate,rows,ok})=>(
            <div key={gate} className={`p-3 rounded-xl border ${theme==='dark'?'bg-slate-800/60 border-white/10':'bg-gray-50 border-gray-200'}`}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-bold" style={{color:ok?'#10B981':'#EF4444'}}>{gate}</span>
                <span className="text-xs">{ok?'✅':'❌'}</span>
              </div>
              <table className="w-full text-xs">
                <thead><tr>{['x₁','x₂','y'].map(h=><th key={h} className={`text-center pb-1 ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>{h}</th>)}</tr></thead>
                <tbody>
                  {rows.map((r,i)=>(
                    <tr key={i}>
                      {r.map((v,j)=>(
                        <td key={j} className={`text-center py-0.5 font-mono ${j===2?v?'text-emerald-400':'text-red-400':theme==='dark'?'text-gray-300':'text-gray-700'}`}>{v}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
        <p className={`text-xs mt-2 ${theme==='dark'?'text-gray-600':'text-gray-400'}`}>✅ = Perceptron can learn this &nbsp; ❌ = Requires MLP</p>
      </div>

      {/* Code */}
      <div className={S}>
        <p className={LBL}>Python Example</p>
        <h2 className={H2}>AND gate vs XOR — success and failure</h2>
        <p className={`${BODY} mb-3`}>This example demonstrates the Perceptron's limitation: it solves AND perfectly but fails at XOR.</p>
        <CodeBlock code={PYTHON_CODE} />
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme==='dark'?'bg-cyan-500/10 border-cyan-500/20':'bg-cyan-50 border-cyan-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme==='dark'?'text-white':'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>5 questions • +100 XP</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="perceptron" />
      </div>
    </motion.div>
  )
}
