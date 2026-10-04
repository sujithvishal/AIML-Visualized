import { AnimatePresence, motion } from 'framer-motion'
import { AppProvider } from './context/AppContext'
import { useApp } from './context/useApp'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import GlobalSearch from './components/GlobalSearch'
import XPPanel from './components/XPPanel'
import ParticleBackground from './components/ParticleBackground'

// ML pages
import WhatIsML from './pages/ml/WhatIsML'
import LinearRegression from './pages/ml/LinearRegression'
import LogisticRegression from './pages/ml/LogisticRegression'
import DecisionTree from './pages/ml/DecisionTree'
import KNN from './pages/ml/KNN'
// DNN pages
import WhatIsDNN from './pages/dnn/WhatIsDNN'
import Perceptron from './pages/dnn/Perceptron'
import MLP from './pages/dnn/MLP'
import LinearNeuronRegression from './pages/dnn/LinearNeuronRegression'
import LinearNeuronClassification from './pages/dnn/LinearNeuronClassification'
import CNN from './pages/dnn/CNN'
import DeepCNN from './pages/dnn/DeepCNN'
import TransferLearning from './pages/dnn/TransferLearning'
import RNN from './pages/dnn/RNN'
import DeepRNN from './pages/dnn/DeepRNN'
import Attention from './pages/dnn/Attention'
import Transformer from './pages/dnn/Transformer'
import OptimizationDeepModels from './pages/dnn/OptimizationDeepModels'
import RegularizationDeepModels from './pages/dnn/RegularizationDeepModels'

const PAGE_MAP = {
  'what-is-ml': WhatIsML,
  'linear-regression': LinearRegression,
  'logistic-regression': LogisticRegression,
  'decision-tree': DecisionTree,
  'knn': KNN,
  'what-is-dnn': WhatIsDNN,
  'perceptron': Perceptron,
  'mlp': MLP,
  'linear-neuron-regression': LinearNeuronRegression,
  'linear-neuron-classification': LinearNeuronClassification,
  'cnn': CNN,
  'deep-cnn': DeepCNN,
  'transfer-learning': TransferLearning,
  'rnn': RNN,
  'deep-rnn': DeepRNN,
  'attention': Attention,
  'transformer': Transformer,
  'optimization-deep-models': OptimizationDeepModels,
  'regularization-deep-models': RegularizationDeepModels,
}

function MainContent() {
  const { topic, sidebarOpen, theme } = useApp()
  const Page = PAGE_MAP[topic] || WhatIsML

  return (
    <div
      className={`transition-all duration-300 min-h-screen ${
        theme === 'dark' ? 'bg-slate-950' : 'bg-white'
      } ${sidebarOpen ? 'md:ml-64' : 'ml-0'}`}
    >
      <div>
        <div className="max-w-5xl mx-auto px-5 sm:px-8 lg:px-10 pt-24 pb-20">
          <AnimatePresence mode="wait">
            <motion.div
              key={topic}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.25 }}
            >
              <Page />
            </motion.div>
          </AnimatePresence>

          {/* XP Panel at bottom */}
          <div className="mt-12">
            <XPPanel />
          </div>
        </div>
      </div>
    </div>
  )
}

function App() {
  return (
    <AppProvider>
      <AppInner />
    </AppProvider>
  )
}

function AppInner() {
  const { theme } = useApp()
  return (
    <div className={theme}>
      <ParticleBackground />
      <Navbar />
      <Sidebar />
      <GlobalSearch />
      <MainContent />
    </div>
  )
}

export default App
