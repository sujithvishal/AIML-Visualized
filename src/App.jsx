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
// DNN pages
import WhatIsDNN from './pages/dnn/WhatIsDNN'
import Perceptron from './pages/dnn/Perceptron'
import MLP from './pages/dnn/MLP'

const PAGE_MAP = {
  'what-is-ml': WhatIsML,
  'linear-regression': LinearRegression,
  'logistic-regression': LogisticRegression,
  'what-is-dnn': WhatIsDNN,
  'perceptron': Perceptron,
  'mlp': MLP,
}

function MainContent() {
  const { topic, sidebarOpen, theme } = useApp()
  const Page = PAGE_MAP[topic] || WhatIsML

  return (
    <div
      className={`transition-all duration-300 min-h-screen ${
        sidebarOpen ? 'md:ml-64' : 'ml-0'
      }`}
    >
      <div className={`min-h-screen ${
        theme === 'dark'
          ? 'bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950/30'
          : 'bg-gradient-to-br from-gray-50 via-white to-indigo-50/50'
      }`}>
        <div className="max-w-4xl mx-auto px-6 pt-24 pb-16">
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
