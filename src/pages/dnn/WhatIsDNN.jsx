import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../context/useApp'
import CodeBlock from '../../components/CodeBlock'
import Quiz from '../../components/Quiz'
import TheoryBlock from '../../components/TheoryBlock'
import Callout from '../../components/Callout'
import DeepDive from '../../components/DeepDive'

const PYTHON_CODE = `import tensorflow as tf
import numpy as np

# Build a DNN for loan approval (3 input features)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(8,  activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Training (dummy data for illustration)
X_train = np.random.rand(1000, 3)
y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(float)

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

# Inference: age=28, income=55k, credit=720 (normalised 0-1)
x_new = np.array([[0.17, 0.27, 0.63]])
prob = model.predict(x_new)[0][0]
print(f"Approval probability: {prob:.2%}")`

const QUIZ_QUESTIONS = [
  { question: 'What is a Deep Neural Network?', options: ['A network with exactly 2 layers','A neural network with multiple hidden layers','A type of recurrent network','A linear regression model'], correct: 1, explanation: 'A DNN has multiple hidden layers, enabling it to learn increasingly abstract representations of data.' },
  { question: 'What is the role of hidden layers?', options: ['To store training data','To learn hierarchical feature representations','To reduce model size','To output predictions directly'], correct: 1, explanation: 'Hidden layers learn intermediate features — early layers detect simple patterns, deeper layers detect complex abstractions.' },
  { question: 'Forward propagation means:', options: ['Moving data from output to input','Updating weights using gradients','Passing inputs through each layer to compute output','Removing neurons'], correct: 2, explanation: 'Forward propagation passes input through each layer sequentially, computing activations until the final output.' },
  { question: 'Which activation is most commonly used in hidden layers?', options: ['Sigmoid','Tanh','ReLU','Linear'], correct: 2, explanation: 'ReLU (max(0,x)) is preferred — it avoids vanishing gradients and is computationally efficient.' },
  { question: 'Backpropagation computes:', options: ['The model architecture','Gradients of the loss w.r.t. each weight','The learning rate','The number of layers'], correct: 1, explanation: 'Backpropagation applies the chain rule to compute ∂Loss/∂w for every weight, enabling gradient descent to update them.' },
]

const DNN_LAYERS = [
  { neurons: 3, label: 'Input Layer', x: 0.12 },
  { neurons: 4, label: 'Hidden 1',   x: 0.38 },
  { neurons: 4, label: 'Hidden 2',   x: 0.62 },
  { neurons: 2, label: 'Output',     x: 0.88 },
]

function NeuralNetDiagram({ theme, activeNeurons }) {
  const canvasRef = useRef(null)
  const W=380, H=280
  const layers=DNN_LAYERS

  useEffect(()=>{
    const canvas=canvasRef.current; if(!canvas) return
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H)
    const pos=(li,ni,total)=>({x:layers[li].x*W,y:(ni+1)/(total+1)*H})
    layers.forEach((layer,li)=>{
      if(li>=layers.length-1) return
      const next=layers[li+1]
      for(let ni=0;ni<layer.neurons;ni++){
        for(let nj=0;nj<next.neurons;nj++){
          const a=pos(li,ni,layer.neurons),b=pos(li+1,nj,next.neurons)
          const active=activeNeurons.includes(`${li}-${ni}`)&&activeNeurons.includes(`${li+1}-${nj}`)
          ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y)
          ctx.strokeStyle=active?'rgba(99,102,241,0.7)':theme==='dark'?'rgba(255,255,255,0.06)':'rgba(0,0,0,0.08)'
          ctx.lineWidth=active?1.5:0.8; ctx.stroke()
        }
      }
    })
    layers.forEach((layer,li)=>{
      for(let ni=0;ni<layer.neurons;ni++){
        const p=pos(li,ni,layer.neurons)
        const active=activeNeurons.includes(`${li}-${ni}`)
        if(active){ctx.beginPath();ctx.arc(p.x,p.y,18,0,Math.PI*2);ctx.fillStyle='rgba(99,102,241,0.2)';ctx.fill()}
        ctx.beginPath();ctx.arc(p.x,p.y,12,0,Math.PI*2)
        ctx.fillStyle=active?li===0?'#06B6D4':li===layers.length-1?'#10B981':'#6366F1':theme==='dark'?'#1E293B':'#F1F5F9'
        ctx.fill(); ctx.strokeStyle=active?'white':theme==='dark'?'#334155':'#CBD5E1'; ctx.lineWidth=2; ctx.stroke()
      }
      ctx.fillStyle=theme==='dark'?'#6B7280':'#9CA3AF'; ctx.font='9px sans-serif'; ctx.textAlign='center'
      ctx.fillText(layer.label,layer.x*W,H-4)
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[activeNeurons,theme])

  return <canvas ref={canvasRef} width={W} height={H}
    className={`w-full rounded-xl border ${theme==='dark'?'bg-slate-900 border-white/10':'bg-gray-50 border-gray-200'}`}
    style={{maxWidth:W,height:H}} />
}

function ForwardPropDemo({ theme }) {
  const [age,setAge]=useState(28), [income,setIncome]=useState(55), [credit,setCredit]=useState(720)
  const [activeNeurons,setActiveNeurons]=useState([])
  const [running,setRunning]=useState(false), [result,setResult]=useState(null)
  const sigmoid=z=>1/(1+Math.exp(-z))

  const run=async()=>{
    if(running)return; setRunning(true); setResult(null); setActiveNeurons([])
    const delay=ms=>new Promise(r=>setTimeout(r,ms))
    setActiveNeurons(['0-0','0-1','0-2']); await delay(600)
    setActiveNeurons(p=>[...p,'1-0','1-1','1-2','1-3']); await delay(600)
    setActiveNeurons(p=>[...p,'2-0','2-1','2-2','2-3']); await delay(600)
    const norm=(age-18)/60+(income/150)+(credit-500)/350
    const prob=sigmoid(norm-1.2)
    setActiveNeurons(p=>[...p,'3-0','3-1']); await delay(400)
    setResult(prob); setRunning(false)
  }

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        {[
          {label:'Age',value:age,set:setAge,min:18,max:65,unit:'yrs',color:'#6366F1'},
          {label:'Income',value:income,set:setIncome,min:20,max:200,unit:'k',color:'#8B5CF6'},
          {label:'Credit Score',value:credit,set:setCredit,min:300,max:850,unit:'',color:'#06B6D4'},
        ].map(({label,value,set,min,max,unit,color})=>(
          <div key={label} className={`p-3 rounded-xl ${theme==='dark'?'bg-slate-800':'bg-gray-50'}`}>
            <div className="flex justify-between mb-1">
              <span className={`text-xs font-medium ${theme==='dark'?'text-gray-300':'text-gray-600'}`}>{label}</span>
              <span className="text-xs font-bold" style={{color}}>{value}{unit}</span>
            </div>
            <input type="range" min={min} max={max} value={value} onChange={e=>set(parseInt(e.target.value))} className="w-full" style={{accentColor:color}}/>
          </div>
        ))}
      </div>
      <NeuralNetDiagram theme={theme} activeNeurons={activeNeurons} />
      <div className="flex items-center gap-4 mt-4">
        <button onClick={run} disabled={running}
          className="px-5 py-2.5 rounded-xl bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 text-white text-sm font-medium transition-colors flex items-center gap-2">
          {running?'⚡ Propagating...':'▶ Run Forward Pass'}
        </button>
        {result!==null&&(
          <motion.div initial={{opacity:0,scale:0.8}} animate={{opacity:1,scale:1}}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${result>0.5?'bg-emerald-500/20 border-emerald-500/30 text-emerald-300':'bg-red-500/20 border-red-500/30 text-red-300'}`}>
            <span className="text-lg">{result>0.5?'✅':'❌'}</span>
            <div>
              <p className="text-xs font-bold">{result>0.5?'Approved':'Rejected'}</p>
              <p className="text-xs opacity-75">{(result*100).toFixed(1)}% confidence</p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default function WhatIsDNN() {
  const { theme } = useApp()
  const S=`rounded-2xl border p-6 mb-6 ${theme==='dark'?'bg-slate-900/80 border-white/10':'bg-white border-gray-200'}`
  const LBL=`text-xs font-semibold uppercase tracking-wider mb-3 ${theme==='dark'?'text-cyan-400':'text-cyan-600'}`
  const H2=`text-xl font-bold mb-2 ${theme==='dark'?'text-white':'text-gray-900'}`
  const BODY=`text-sm leading-relaxed ${theme==='dark'?'text-gray-400':'text-gray-600'}`

  const layers=[
    {label:'Input Layer',icon:'📥',desc:'Receives raw features — numbers representing real-world attributes.',color:'from-cyan-500 to-blue-500',count:'N neurons'},
    {label:'Hidden Layer 1',icon:'🔵',desc:'Learns simple combinations of input features (edges, basic patterns).',color:'from-indigo-500 to-purple-500',count:'M neurons'},
    {label:'Hidden Layer 2',icon:'🟣',desc:'Learns complex abstractions by combining hidden layer 1 outputs.',color:'from-purple-500 to-pink-500',count:'K neurons'},
    {label:'Output Layer',icon:'📤',desc:'Produces the final prediction — probability, class, or numeric value.',color:'from-emerald-500 to-teal-500',count:'C neurons'},
  ]

  return (
    <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} className="max-w-3xl">
      {/* Hero */}
      <div className="mb-8">
        <motion.div initial={{opacity:0,scale:0.9}} animate={{opacity:1,scale:1}}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/20 border border-cyan-500/30 mb-4">
          <span className="text-xs text-cyan-400 font-medium">Deep Neural Networks • Fundamentals</span>
        </motion.div>
        <h1 className={`text-4xl font-bold tracking-tight mb-4 ${theme==='dark'?'text-white':'text-gray-900'}`}>
          What is a <span className="gradient-text">Deep Neural Network?</span>
        </h1>
        <p className={`text-lg mb-4 ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>
          A computational model loosely inspired by the brain — composed of <strong className={theme==='dark'?'text-white':'text-gray-900'}>layers of interconnected neurons</strong> that learn hierarchical representations from data.
        </p>
        <Callout type="analogy" title="Why 'deep'?">
          "Depth" = number of layers. A shallow network has 1 hidden layer; a deep network has many. GPT-4 has 96 layers. Each layer transforms the data into a higher-level representation — words → phrases → meaning.
        </Callout>
      </div>

      <TheoryBlock title="Core Theory" cards={[
        { icon: '🧠', title: 'Biological Inspiration', body: 'Artificial neurons loosely mimic biological neurons. Each computes a weighted sum of inputs, applies an activation function, and passes the result to the next layer.', mono: 'output = f(Σ wᵢxᵢ + b)' },
        { icon: '📦', title: 'Hierarchical Features', body: 'Early layers detect low-level features (edges, tones). Middle layers combine these into shapes. Deep layers recognise complex objects. This hierarchy is the core power of deep learning.' },
        { icon: '🎯', title: 'Backpropagation', body: 'After a forward pass, the error signal flows backward through all layers. The chain rule computes gradients, telling each weight how much to change.', mono: '∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w' },
        { icon: '⚡', title: 'Activation Functions', body: 'Non-linear functions applied after the weighted sum. Without them, any stack of linear layers collapses to a single linear transformation — killing the network\'s power.', mono: 'ReLU(z) = max(0, z)' },
        { icon: '📉', title: 'Loss & Optimisation', body: 'A loss function measures prediction error. An optimiser (Adam, SGD) uses gradients from backprop to update weights after each mini-batch of training examples.', mono: 'w := w − α·∂L/∂w' },
        { icon: '🔢', title: 'Universal Approximation', body: 'A single hidden layer with enough neurons can approximate any continuous function. Depth makes this practical — fewer parameters needed for the same accuracy.', mono: 'f(x) ≈ DNN(x)  ∀f' },
      ]} />

      {/* Architecture */}
      <div className={S}>
        <p className={LBL}>Network Architecture</p>
        <h2 className={H2}>Every DNN has this structure</h2>
        <p className={`${BODY} mb-4`}>Data flows left to right through each layer. Each neuron in a layer is connected to every neuron in the next layer (fully connected / dense).</p>
        <div className="flex flex-col items-center gap-2 mb-4">
          {layers.map((l,i)=>(
            <div key={i} className="flex flex-col items-center w-full">
              <motion.div initial={{opacity:0,x:-20}} animate={{opacity:1,x:0}} transition={{delay:i*0.1}}
                className={`w-full max-w-sm px-5 py-3 rounded-xl border flex items-center gap-4 ${theme==='dark'?'bg-slate-800/60 border-white/10':'bg-gray-50 border-gray-200'}`}>
                <span className="text-2xl">{l.icon}</span>
                <div className="flex-1">
                  <p className={`font-semibold text-sm ${theme==='dark'?'text-white':'text-gray-900'}`}>{l.label}</p>
                  <p className={`text-xs ${theme==='dark'?'text-gray-500':'text-gray-400'}`}>{l.desc}</p>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full bg-gradient-to-r ${l.color} text-white font-medium shrink-0`}>{l.count}</span>
              </motion.div>
              {i<layers.length-1&&<div className="text-2xl text-indigo-500 leading-none py-1">↓</div>}
            </div>
          ))}
        </div>
        <Callout type="info" title="Fully Connected vs other architectures">
          What we're building here is a fully connected (dense) network. Other specialised architectures exist: CNNs for images (use local filters), RNNs for sequences (have memory), and Transformers for text (use attention). All share the same backprop/gradient descent training loop.
        </Callout>
        <DeepDive title="How many neurons and layers should you use?">
          <p className={`text-sm ${BODY} mb-3`}>There's no one-size-fits-all answer. Common starting points:</p>
          <TheoryBlock title="" cards={[
            { icon: '🔢', title: 'Layer Count', body: '2–3 hidden layers solve most tabular problems. Image/text tasks typically need 10–100+ layers.', mono: 'start with 2–3' },
            { icon: '⚖️', title: 'Neuron Count', body: 'Pyramid shape often works: wide early layers, narrower deeper layers. Start with 64–256 neurons.', mono: 'e.g. 256→128→64→1' },
            { icon: '🧪', title: 'Experimentation', body: 'Use validation loss to guide architecture decisions. Increase capacity if underfitting; add dropout/regularisation if overfitting.' },
          ]} />
        </DeepDive>
      </div>

      {/* Forward Prop Demo */}
      <div className={S}>
        <p className={LBL}>Forward Propagation Demo</p>
        <h2 className={H2}>Loan Approval Network — watch data flow</h2>
        <p className={`${BODY} mb-3`}>Adjust inputs and click Run. Watch the activation signal propagate layer-by-layer from inputs to the final approval probability.</p>
        <Callout type="info" title="What happens at each neuron?">
          Each neuron: (1) multiplies inputs by weights, (2) sums them with bias → z, (3) applies ReLU(z) or σ(z), (4) sends result to all neurons in the next layer. This is one "forward pass".
        </Callout>
        <ForwardPropDemo theme={theme} />
      </div>

      {/* Training */}
      <div className={S}>
        <p className={LBL}>How a DNN Learns</p>
        <h2 className={H2}>The complete training loop</h2>
        <div className="grid sm:grid-cols-2 gap-3 mb-4">
          {[
            {step:'1',title:'Initialise Weights',body:'Start with small random weights. Too large → neurons saturate. Too small → vanishing gradients.',color:'#6366F1'},
            {step:'2',title:'Forward Pass',body:'Feed a mini-batch through all layers. Compute predictions ŷ.',color:'#8B5CF6'},
            {step:'3',title:'Compute Loss',body:'Measure error between ŷ and true labels y using the loss function (e.g. cross-entropy).',color:'#06B6D4'},
            {step:'4',title:'Backpropagation',body:'Apply chain rule backwards through all layers to compute ∂L/∂w for every weight.',color:'#10B981'},
            {step:'5',title:'Update Weights',body:'Optimiser (Adam) adjusts each weight: w := w − α·∂L/∂w. Repeat for all batches.',color:'#F59E0B'},
            {step:'6',title:'Repeat Epochs',body:'One epoch = one complete pass through training data. Repeat 10–1000s of epochs until loss converges.',color:'#EF4444'},
          ].map(s=>(
            <div key={s.step} className={`p-4 rounded-xl border ${theme==='dark'?'bg-slate-800/60 border-white/10':'bg-gray-50 border-gray-200'}`}>
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 rounded-full text-white text-xs font-bold flex items-center justify-center shrink-0" style={{background:s.color}}>{s.step}</span>
                <span className={`font-semibold text-sm ${theme==='dark'?'text-white':'text-gray-900'}`}>{s.title}</span>
              </div>
              <p className={`text-xs leading-relaxed ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>{s.body}</p>
            </div>
          ))}
        </div>
        <Callout type="formula" mono="Loss ↓  →  Backprop  →  Weight update  →  Repeat">
          Each iteration of this loop is called a training step. After enough steps, the network learns to produce accurate predictions.
        </Callout>
        <DeepDive title="What is the Adam optimiser?">
          <p className={`text-sm ${BODY} mb-2`}>Adam (Adaptive Moment Estimation) is the most popular DNN optimiser. It adapts the learning rate for each weight individually using estimates of the first and second moments of the gradients. This makes it much more robust than vanilla SGD and typically requires less hyperparameter tuning.</p>
          <Callout type="formula" mono="mₜ = β₁mₜ₋₁ + (1−β₁)gₜ  |  vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²">β₁=0.9, β₂=0.999 are the default momentum parameters. g is the gradient.</Callout>
        </DeepDive>
      </div>

      {/* Code */}
      <div className={S}>
        <p className={LBL}>Python Example — TensorFlow/Keras</p>
        <h2 className={H2}>Build a DNN in 10 lines</h2>
        <p className={`${BODY} mb-3`}>Keras provides a high-level API for building and training DNNs. Sequential stacks layers; Dense adds a fully connected layer.</p>
        <CodeBlock code={PYTHON_CODE} />
        <DeepDive title="TensorFlow vs PyTorch">
          <TheoryBlock title="" cards={[
            { icon: '🌊', title: 'TensorFlow / Keras', body: 'High-level, production-ready. Great for deployment (TF Serving, TFLite). Keras makes building models very simple.', mono: 'model.fit() one-liner' },
            { icon: '🔥', title: 'PyTorch', body: 'More Pythonic, preferred in research. Dynamic computation graph gives more flexibility. Dominant in academic papers.', mono: 'model.forward()' },
          ]} />
        </DeepDive>
      </div>

      {/* Quiz */}
      <div className="mb-6">
        <div className={`rounded-2xl border p-4 mb-4 flex items-center gap-3 ${theme==='dark'?'bg-cyan-500/10 border-cyan-500/20':'bg-cyan-50 border-cyan-200'}`}>
          <span className="text-xl">📝</span>
          <div>
            <p className={`font-semibold ${theme==='dark'?'text-white':'text-gray-900'}`}>Knowledge Check</p>
            <p className={`text-xs ${theme==='dark'?'text-gray-400':'text-gray-600'}`}>5 questions • Earn Neural Network Starter badge!</p>
          </div>
        </div>
        <Quiz questions={QUIZ_QUESTIONS} topicId="what-is-dnn" />
      </div>
    </motion.div>
  )
}
