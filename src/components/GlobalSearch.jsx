import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, X, ArrowRight } from 'lucide-react'
import { useApp } from '../context/useApp'
import { topics } from '../context/topics'

const ALL_ITEMS = [
  ...topics.ml.map(t => ({ ...t, section: 'ml', sectionLabel: 'Machine Learning' })),
  ...topics.dnn.map(t => ({ ...t, section: 'dnn', sectionLabel: 'Deep Neural Networks' })),
  { id: 'sigmoid', label: 'Sigmoid Function', icon: '📊', section: 'ml', sectionLabel: 'Concepts' },
  { id: 'gradient-descent', label: 'Gradient Descent', icon: '⛰️', section: 'ml', sectionLabel: 'Concepts' },
  { id: 'activation-functions', label: 'Activation Functions', icon: '⚡', section: 'dnn', sectionLabel: 'Concepts' },
  { id: 'backpropagation', label: 'Backpropagation', icon: '🔄', section: 'dnn', sectionLabel: 'Concepts' },
]

export default function GlobalSearch() {
  const { theme, searchOpen, setSearchOpen, setSection, setTopic } = useApp()
  const [query, setQuery] = useState('')
  const inputRef = useRef(null)

  useEffect(() => {
    if (searchOpen) {
      const timer = setTimeout(() => inputRef.current?.focus(), 100)
      return () => clearTimeout(timer)
    }
  }, [searchOpen])

  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); setSearchOpen(true) }
      if (e.key === 'Escape') setSearchOpen(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [setSearchOpen])

  const results = query.length > 0
    ? ALL_ITEMS.filter(item =>
        item.label.toLowerCase().includes(query.toLowerCase()) ||
        item.sectionLabel.toLowerCase().includes(query.toLowerCase())
      )
    : ALL_ITEMS

  const handleSelect = (item) => {
    if (topics[item.section]) {
      setSection(item.section)
      const t = topics[item.section].find(t => t.id === item.id)
      if (t) setTopic(t.id)
    }
    setSearchOpen(false)
  }

  return (
    <AnimatePresence>
      {searchOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={() => setSearchOpen(false)}
          className="fixed inset-0 z-[100] flex items-start justify-center pt-24 px-4"
          style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            onClick={e => e.stopPropagation()}
            className={`w-full max-w-xl rounded-2xl border overflow-hidden shadow-2xl ${
              theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-white border-gray-200'
            }`}
          >
            {/* Input */}
            <div className={`flex items-center gap-3 px-4 py-3 border-b ${
              theme === 'dark' ? 'border-white/10' : 'border-gray-200'
            }`}>
              <Search size={16} className="text-gray-500 shrink-0" />
              <input
                ref={inputRef}
                type="text"
                placeholder="Search topics, concepts, code..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                className={`flex-1 bg-transparent text-sm outline-none ${
                  theme === 'dark' ? 'text-white placeholder-gray-600' : 'text-gray-900 placeholder-gray-400'
                }`}
              />
              <button onClick={() => setSearchOpen(false)} className="text-gray-500 hover:text-gray-300 transition-colors">
                <X size={16} />
              </button>
            </div>

            {/* Results */}
            <div className="max-h-80 overflow-y-auto py-2">
              {results.length === 0 ? (
                <p className={`text-center py-8 text-sm ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
                  No results found
                </p>
              ) : results.map(item => (
                <button
                  key={`${item.section}-${item.id}`}
                  onClick={() => handleSelect(item)}
                  className={`w-full flex items-center gap-3 px-4 py-3 text-sm transition-colors ${
                    theme === 'dark'
                      ? 'hover:bg-white/5 text-gray-300'
                      : 'hover:bg-gray-50 text-gray-700'
                  }`}
                >
                  <span className="text-lg">{item.icon}</span>
                  <div className="flex-1 text-left">
                    <div className={`font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{item.label}</div>
                    <div className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{item.sectionLabel}</div>
                  </div>
                  <ArrowRight size={14} className="text-gray-500" />
                </button>
              ))}
            </div>

            {/* Footer */}
            <div className={`px-4 py-2 border-t flex items-center gap-4 ${
              theme === 'dark' ? 'border-white/10' : 'border-gray-200'
            }`}>
              <span className={`text-xs ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
                ↵ Select &nbsp; ↑↓ Navigate &nbsp; Esc Close
              </span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
