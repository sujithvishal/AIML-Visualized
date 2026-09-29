import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import { useApp } from '../context/useApp'

/**
 * DeepDive — collapsible "want to know more?" block.
 * Props: title (string), children (JSX content inside)
 */
export default function DeepDive({ title = 'Deep Dive', children }) {
  const { theme } = useApp()
  const [open, setOpen] = useState(false)

  return (
    <div className={`rounded-xl border mb-4 overflow-hidden ${
      theme === 'dark' ? 'border-white/10 bg-slate-800/40' : 'border-gray-200 bg-gray-50'
    }`}>
      <button
        onClick={() => setOpen(o => !o)}
        className={`w-full flex items-center justify-between px-4 py-3 text-sm font-medium transition-colors ${
          theme === 'dark'
            ? 'text-gray-300 hover:text-white hover:bg-white/5'
            : 'text-gray-700 hover:text-gray-900 hover:bg-gray-100'
        }`}
      >
        <div className="flex items-center gap-2">
          <span className="text-indigo-400">🔬</span>
          <span>{title}</span>
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            theme === 'dark' ? 'bg-indigo-500/20 text-indigo-400' : 'bg-indigo-100 text-indigo-600'
          }`}>optional</span>
        </div>
        <motion.div animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown size={16} className="text-gray-500" />
        </motion.div>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className={`px-4 pb-4 pt-2 border-t ${
              theme === 'dark' ? 'border-white/10' : 'border-gray-200'
            }`}>
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
