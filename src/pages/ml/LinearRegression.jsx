import { useState, useRef, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE = `from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Dataset: study hours → exam score
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([15, 25, 35, 45, 52, 61, 70, 79, 85, 95])

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Slope (m)      : {model.coef_[0]:.2f}")
print(f"Intercept (b)  : {model.intercept_:.2f}")
print(f"MSE            : {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² score       : {r2_score(y_test, y_pred):.4f}")

# Predict for 7.5 study hours
print(f"Predicted score: {model.predict([[7.5]])[0]:.1f}")`

const QUIZ_QUESTIONS = [
  { question: 'In Y = mX + b, what does "m" represent?', options: ['The intercept','The slope','The error','The bias'], correct: 1, explanation: '"m" is the slope — it determines how much Y changes per unit increase in X.' },
  { question: 'What does MSE measure?', options: ['The mean of all data points','Average squared differences between predicted and actual values','Maximum prediction error','The slope'], correct: 1, explanation: 'MSE averages (yᵢ − ŷᵢ)² — it penalises large errors more heavily by squaring them.' },
  { question: 'When slope m = 0, the regression line is:', options: ['Vertical','Diagonal','Horizontal','Curved'], correct: 2, explanation: 'Slope = 0 means Y does not change with X — the line is horizontal at Y = b.' },
  { question: 'Gradient Descent minimises:', options: ['The number of features','The slope','The cost function (MSE)','The intercept'], correct: 2, explanation: 'Gradient Descent iteratively adjusts m and b to minimise MSE, converging toward optimal parameters.' },
  { question: 'R² score of 1.0 means:', options: ['The model is wrong','The model explains 100% of variance in the data','The slope is 1','The intercept is 0'], correct: 1, explanation: 'R² measures the proportion of variance explained. R² = 1 is a perfect fit; R² = 0 means the model is no better than predicting the mean.' },
]

function LinearRegressionCanvas({ theme }) {
  const canvasRef = useRef(null)
  const [points, setPoints] = useState([[50,200],[80,175],[110,155],[150,130],[190,105],[230,85],[270,65],[310,45]])
  const [dragging, setDragging] = useState(null)
  const W = 380, H = 260

  const calcReg = useCallback((pts) => {
    const n = pts.length
    if (n < 2) return { m: 0, b: H/2 }
    const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1])
    const mx = xs.reduce((a,b)=>a+b,0)/n, my = ys.reduce((a,b)=>a+b,0)/n
    const num = xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0)
    const den = xs.reduce((s,x)=>s+(x-mx)**2,0)
    const m = den!==0?num/den:0
    return { m, b: my-m*mx }
  }, [])

  const draw = useCallback(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    ctx.strokeStyle = theme==='dark'?'rgba(255,255,255,0.04)':'rgba(0,0,0,0.05)'; ctx.lineWidth=1
    for(let x=0;x<W;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}
    for(let y=0;y<H;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
    const {m,b}=calcReg(points)
    const grad=ctx.createLinearGradient(0,0,W,0); grad.addColorStop(0,'#6366F1'); grad.addColorStop(1,'#06B6D4')
    ctx.beginPath(); ctx.moveTo(0,b); ctx.lineTo(W,m*W+b); ctx.strokeStyle=grad; ctx.lineWidth=2.5; ctx.stroke()
    points.forEach(([px,py])=>{
      const pr=m*px+b
      ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px,pr)
      ctx.strokeStyle='rgba(239,68,68,0.4)'; ctx.lineWidth=1.5; ctx.setLineDash([3,3]); ctx.stroke(); ctx.setLineDash([])
    })
    points.forEach(([px,py],i)=>{
      ctx.beginPath(); ctx.arc(px,py,dragging===i?9:7,0,Math.PI*2)
      ctx.fillStyle='#6366F1'; ctx.fill(); ctx.strokeStyle='white'; ctx.lineWidth=2; ctx.stroke()
    })
  }, [points,dragging,theme,calcReg])

  useEffect(()=>{draw()},[draw])

  const getXY=(e,c)=>{const r=c.getBoundingClientRect();const cx=e.touches?e.touches[0].clientX:e.clientX;const cy=e.touches?e.touches[0].clientY:e.clientY;return[(cx-r.left)*W/r.width,(cy-r.top)*H/r.height]}
  const handleMouseDown=(e)=>{const[mx,my]=getXY(e,canvasRef.current);const idx=points.findIndex(([px,py])=>Math.hypot(px-mx,py-my)<12);if(idx>=0)setDragging(idx);else setPoints(p=>[...p,[mx,my]])}
  const handleMouseMove=(e)=>{if(dragging===null)return;const[mx,my]=getXY(e,canvasRef.current);setPoints(p=>p.map((pt,i)=>i===dragging?[mx,my]:pt))}
  const handleMouseUp=()=>setDragging(null)

  const {m,b}=calcReg(points)
  const mse=points.reduce((s,[px,py])=>s+(py-(m*px+b))**2,0)/points.length

  return (
    <div>
      <canvas ref={canvasRef} width={W} height={H}
        onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}
        className={`w-full rounded-xl border cursor-crosshair ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
        style={{maxWidth:W,height:H}} />
      <div className="flex flex-wrap gap-3 mt-3 text-xs">
        {[['Slope (m)', m.toFixed(3),'text-indigo-300'],['Intercept (b)',b.toFixed(1),'text-purple-300'],['MSE',mse.toFixed(1),'text-red-300']].map(([k,v,c])=>(
          <div key={k} className={`px-3 py-1.5 rounded-lg ${theme==='dark'?'bg-slate-800':'bg-gray-100'}`}>
            <span className={theme==='dark'?'text-gray-500':'text-gray-400'}>{k}: </span>
            <span className={`font-mono font-medium ${c}`}>{v}</span>
          </div>
        ))}
      </div>
      <p className={`text-xs mt-1 ${theme==='dark'?'text-gray-600':'text-gray-400'}`}>Drag points • Click empty space to add • Red lines = residuals (errors)</p>
    </div>
  )
}

function SlopeInterceptPlayground({ theme }) {
  const [m, setM] = useState(1)
  const [b, setB] = useState(0)
  const canvasRef = useRef(null)
  const W=380, H=260

  useEffect(()=>{
    const canvas=canvasRef.current; if(!canvas) return
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    ctx.strokeStyle=theme==='dark'?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.06)'; ctx.lineWidth=1
    for(let x=0;x<W;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}
    for(let y=0;y<H;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
    ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.strokeStyle=theme==='dark'?'rgba(255,255,255,0.2)':'rgba(0,0,0,0.2)';ctx.lineWidth=1;ctx.stroke()
    ctx.beginPath();ctx.moveTo(W/2,0);ctx.lineTo(W/2,H);ctx.stroke()
    const toX=x=>(x+5)/10*W, toY=y=>H-((y+5)/10*H)
    const grad=ctx.createLinearGradient(0,0,W,0); grad.addColorStop(0,'#6366F1'); grad.addColorStop(1,'#06B6D4')
    ctx.beginPath(); ctx.moveTo(toX(-5),toY(m*-5+b)); ctx.lineTo(toX(5),toY(m*5+b)); ctx.strokeStyle=grad; ctx.lineWidth=3; ctx.stroke()
    const ix=toX(0),iy=toY(b); ctx.beginPath(); ctx.arc(ix,iy,6,0,Math.PI*2); ctx.fillStyle='#8B5CF6'; ctx.fill(); ctx.strokeStyle='white'; ctx.lineWidth=2; ctx.stroke()
  },[m,b,theme])

  return (
    <div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>Slope (m) = <span className="text-indigo-400 font-bold">{m.toFixed(1)}</span></label>
          <input type="range" min="-3" max="3" step="0.1" value={m} onChange={e=>setM(parseFloat(e.target.value))} className="w-full accent-indigo-500"/>
          <p className={`text-xs mt-1 ${theme==='dark'?'text-gray-600':'text-gray-400'}`}>{m>0?'Positive: line goes up ↗':m<0?'Negative: line goes down ↘':'Zero: flat line →'}</p>
        </div>
        <div>
          <label className={`text-xs font-medium block mb-1.5 ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>Intercept (b) = <span className="text-purple-400 font-bold">{b.toFixed(1)}</span></label>
          <input type="range" min="-3" max="3" step="0.1" value={b} onChange={e=>setB(parseFloat(e.target.value))} className="w-full accent-purple-500"/>
          <p className={`text-xs mt-1 ${theme==='dark'?'text-gray-600':'text-gray-400'}`}>{b>0?'Line crosses Y-axis above origin':b<0?'Line crosses Y-axis below origin':'Line passes through origin'}</p>
        </div>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
        style={{maxWidth:W,height:H}} />
      <p className={`text-xs mt-2 text-center font-mono ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>
        Y = {m.toFixed(1)}X + {b.toFixed(1)} &nbsp;•&nbsp; 🟣 = y-intercept (where line crosses Y-axis)
      </p>
    </div>
  )
}

function CostSurface3D({ theme }) {
  const canvasRef = useRef(null)
  const [angle, setAngle] = useState(30)
  const W=380, H=280

  useEffect(()=>{
    const canvas=canvasRef.current; if(!canvas) return
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    const toIso=(x,y,z)=>{const rad=angle*Math.PI/180;const ix=(x-y)*Math.cos(rad);const iy=(x+y)*Math.sin(rad)*0.5-z;return[ix*40+W/2,iy*30+H*0.7]}
    const cost=(m,b)=>{const pts=[[1,2],[2,4],[3,5],[4,7],[5,9]];return pts.reduce((s,[x,y])=>s+(y-(m*x+b))**2,0)/pts.length}
    const steps=12
    for(let i=steps-1;i>=0;i--){for(let j=steps-1;j>=0;j--){
      const m1=i/steps*4-2,b1=j/steps*4-2,m2=(i+1)/steps*4-2,b2=(j+1)/steps*4-2
      const c=cost((m1+m2)/2,(b1+b2)/2),t=Math.min(c/20,1)
      const r=Math.round(99+t*156),g=Math.round(102-t*40),bl=Math.round(241-t*200)
      const [x0,y0]=toIso(m1,b1,cost(m1,b1)*0.3),[x1,y1]=toIso(m2,b1,cost(m2,b1)*0.3),[x2,y2]=toIso(m2,b2,cost(m2,b2)*0.3),[x3,y3]=toIso(m1,b2,cost(m1,b2)*0.3)
      ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.lineTo(x2,y2);ctx.lineTo(x3,y3);ctx.closePath()
      ctx.fillStyle=`rgba(${r},${g},${bl},0.8)`;ctx.fill()
      ctx.strokeStyle=theme==='dark'?'rgba(0,0,0,0.3)':'rgba(255,255,255,0.3)';ctx.lineWidth=0.5;ctx.stroke()
    }}
    const [mx,my]=toIso(2,0,cost(2,0)*0.3)
    ctx.beginPath();ctx.arc(mx,my,6,0,Math.PI*2);ctx.fillStyle='#10B981';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke()
  },[angle,theme])

  return (
    <div>
      <div className="mb-3">
        <label className={`text-xs font-medium block mb-1.5 ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>Rotation: {angle}°</label>
        <input type="range" min="10" max="60" value={angle} onChange={e=>setAngle(parseInt(e.target.value))} className="w-full accent-indigo-500"/>
      </div>
      <canvas ref={canvasRef} width={W} height={H}
        className={`w-full rounded-xl border ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
        style={{maxWidth:W,height:H}} />
      <p className={`text-xs mt-2 text-center ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>Cost(m, b) surface • 🟢 = global minimum (optimal m and b)</p>
    </div>
  )
}

export default function LinearRegression() {
  const { theme } = useApp()
  const S=`rounded-2xl border p-6 mb-6 ${theme==='dark'?'bg-slate-900/80 border-white/10':'bg-white border-gray-200'}`
  const LBL=`text-xs font-semibold uppercase tracking-wider mb-3 ${theme==='dark'?'text-indigo-400':'text-indigo-600'}`
  const H2=`text-xl font-bold mb-2 ${theme==='dark'?'text-white':'text-gray-900'}`
  const BODY=`text-sm leading-relaxed ${theme==='dark'?'text-gray-400':'text-gray-600'}`

  return (
    <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} className="max-w-3xl">
      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{opacity:0,scale:0.9}} animate={{opacity:1,scale:1}}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/20 border border-indigo-500/30 mb-4">
          <span className="text-xs text-indigo-400 font-medium">Machine Learning • Regression</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme==='dark'?'text-white':'text-gray-900'}`}>
          Linear <span className="gradient-text">Regression</span>
        </h1>
        <p className={`text-lg mb-4 ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>
          Find the <strong className={theme==='dark'?'text-white':'text-gray-900'}>best-fit straight line</strong> through data to predict a continuous output — the foundational supervised learning algorithm.
        </p>
        <Callout type="analogy" title="Real-world analogy">
          Imagine plotting house sizes vs prices on graph paper and drawing the best straight line through the points with a ruler. Linear regression does exactly that — but mathematically, for any number of features.
        </Callout>
      </div>

      {/* Core theory */}
      <TheoryBlock title="Core Theory" cards={[
        { icon: '📐', title: 'The Line Equation', body: 'Every straight line is defined by slope and intercept. For each extra unit of X, Y increases by m. b sets where the line starts on the Y-axis.', mono: 'Y = mX + b' },
        { icon: '📉', title: 'Residuals', body: 'The vertical distance from each actual data point to the predicted line. These are the "errors" the model is trying to minimise.', mono: 'residual = yᵢ − ŷᵢ' },
        { icon: '🎯', title: 'Least Squares', body: 'The most common method to fit the line. It finds m and b that minimise the sum of squared residuals — the Ordinary Least Squares (OLS) solution.', mono: 'min Σ(yᵢ − (mxᵢ+b))²' },
        { icon: '📊', title: 'R² Score', body: 'Proportion of variance in Y explained by the model. R²=1 is a perfect fit, R²=0 is no better than guessing the mean. Values between 0–1 are typical.', mono: 'R² = 1 − (SS_res/SS_tot)' },
        { icon: '⛰️', title: 'MSE Cost Function', body: 'Mean Squared Error — the average of all squared residuals. Training minimises MSE by adjusting m and b using Gradient Descent.', mono: 'MSE = (1/n) Σ(yᵢ − ŷᵢ)²' },
        { icon: '🔢', title: 'Multiple Features', body: 'Real datasets have many features. Multiple Linear Regression extends to: Y = m₁X₁ + m₂X₂ + … + b (one weight per feature).', mono: 'Y = Xw + b  (vector form)' },
      ]} />

      {/* Equation breakdown */}
      <div className={`${S} text-center`}>
        <p className={LBL}>Equation Breakdown</p>
        <div className="flex items-center justify-center gap-4 flex-wrap mb-4">
          {[
            {sym:'Y', desc:'Predicted output', color:'#10B981'},
            {sym:'=', desc:'', color:theme==='dark'?'#9CA3AF':'#6B7280'},
            {sym:'m', desc:'Slope (learned)', color:'#6366F1'},
            {sym:'X', desc:'Input feature', color:'#06B6D4'},
            {sym:'+', desc:'', color:theme==='dark'?'#9CA3AF':'#6B7280'},
            {sym:'b', desc:'Y-intercept (learned)', color:'#8B5CF6'},
          ].map((item,i)=>(
            item.desc
              ? <div key={i} className="flex flex-col items-center gap-1">
                  <span className="text-4xl font-bold font-mono" style={{color:item.color}}>{item.sym}</span>
                  <span className={`text-xs ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>{item.desc}</span>
                </div>
              : <span key={i} className="text-3xl font-bold" style={{color:item.color}}>{item.sym}</span>
          ))}
        </div>
        <Callout type="info" title="What does 'learning' mean here?">
          The algorithm doesn't know m and b in advance. It starts with random values, checks how wrong the predictions are (MSE), and gradually nudges m and b toward the values that minimise the error. That process is Gradient Descent.
        </Callout>
      </div>

      {/* Interactive scatter */}
      <div className={S}>
        <p className={LBL}>Interactive Scatter Plot</p>
        <h2 className={H2}>Drag points — line updates in real time</h2>
        <p className={`${BODY} mb-3`}>
          The blue gradient line is the <strong className={theme==='dark'?'text-white':'text-gray-900'}>best-fit line</strong> computed via ordinary least squares. Red dashed lines are residuals — the distances being minimised.
        </p>
        <Callout type="warning" title="Try this">
          Add an outlier far from the cluster. Watch how dramatically it pulls the regression line. This is why outlier detection matters before training.
        </Callout>
        <LinearRegressionCanvas theme={theme} />
      </div>

      {/* Slope/Intercept */}
      <div className={S}>
        <p className={LBL}>Slope & Intercept Explorer</p>
        <h2 className={H2}>Understand m and b visually</h2>
        <p className={`${BODY} mb-3`}>Adjust slope and intercept to see exactly how each parameter changes the line. Notice how slope controls the angle and intercept shifts the line up or down.</p>
        <SlopeInterceptPlayground theme={theme} />
        <DeepDive title="How are m and b computed analytically?">
          <p className={`text-sm ${BODY} mb-3`}>For simple linear regression (one feature), there's a closed-form solution called the Normal Equation:</p>
          <TheoryBlock title="" cards={[
            { icon: '📐', title: 'Normal Equation', body: 'Directly solves for the optimal m and b in one step — no iteration needed. Works well for small datasets.', mono: 'm = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²' },
            { icon: '⛰️', title: 'Gradient Descent', body: 'For large datasets, iterative gradient descent is preferred — it scales better than the Normal Equation (which requires a matrix inverse).', mono: 'θ := θ − α·∂J/∂θ' },
          ]} />
        </DeepDive>
      </div>

      {/* Gradient Descent */}
      <div className={S}>
        <p className={LBL}>Gradient Descent & Cost Surface</p>
        <h2 className={H2}>Visualising how training finds the minimum</h2>
        <p className={`${BODY} mb-3`}>
          The 3D surface shows the MSE cost for every possible (m, b) combination. Training is the process of navigating this surface downhill to the green dot — the global minimum.
        </p>
        <Callout type="formula" title="The update rule" mono="m := m − α·(∂MSE/∂m)   b := b − α·(∂MSE/∂b)">
          α (learning rate) controls step size. ∂MSE/∂m tells us which direction is "downhill" for m. The same applies to b.
        </Callout>
        <CostSurface3D theme={theme} />
        <DeepDive title="What is the learning rate α?">
          <Callout type="warning" title="Too large α">If the learning rate is too large, gradient descent overshoots the minimum and may diverge — loss increases instead of decreasing.</Callout>
          <Callout type="success" title="Good α">A well-chosen α allows loss to steadily decrease each epoch, converging smoothly to the minimum.</Callout>
          <p className={`text-xs ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>Common starting values: 0.001, 0.01, 0.1. Use learning rate schedules to decay it over time.</p>
        </DeepDive>
      </div>

      {/* Assumptions */}
      <div className={S}>
        <p className={LBL}>When to Use Linear Regression</p>
        <p className={`${BODY} mb-3`}>Linear regression assumes a linear relationship between features and target. Violating these assumptions degrades performance.</p>
        <div className="grid sm:grid-cols-2 gap-3">
          {[
            { icon:'✅', title:'Use when', items:['Relationship between X and Y is approximately linear','Output is continuous (price, score, temperature)','You need an interpretable model','Dataset is small-to-medium sized'] },
            { icon:'❌', title:'Avoid when', items:['Relationship is clearly non-linear','Output is categorical (class labels)','Many correlated features (multicollinearity)','Heavy outliers in the dataset'] },
          ].map((col,i)=>(
            <div key={i} className={`p-4 rounded-xl border ${theme==='dark'?'bg-slate-800/60 border-white/10':'bg-gray-50 border-gray-200'}`}>
              <p className={`font-semibold text-sm mb-3 flex items-center gap-2 ${theme==='dark'?'text-white':'text-gray-900'}`}>{col.icon} {col.title}</p>
              <ul className="space-y-1">
                {col.items.map(item=>(
                  <li key={item} className={`text-xs flex items-start gap-2 ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>
                    <span className="mt-0.5">•</span><span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Code */}
      <div className={S}>
        <p className={LBL}>Python Example</p>
        <h2 className={H2}>Train, evaluate, and predict</h2>
        <p className={`${BODY} mb-3`}>This example includes the full workflow: data splitting, training, evaluation with MSE and R², and prediction.</p>
        <CodeBlock code={PYTHON_CODE} />
        <DeepDive title="What is R² and why does it matter?">
          <p className={`text-sm ${BODY}`}>R² (R-squared / coefficient of determination) measures how well the model explains the variance in the target variable. An R² of 0.95 means 95% of the variation in Y is explained by the model. Unlike MSE, R² is scale-independent — making it easy to compare models on different datasets.</p>
        </DeepDive>
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme==='dark'?'bg-indigo-500/10 border-indigo-500/20':'bg-indigo-50 border-indigo-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme==='dark'?'text-white':'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>5 questions • Earn 100 XP + Regression Explorer badge</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="linear-regression" />
      </div>
    </motion.div>
  )
}
