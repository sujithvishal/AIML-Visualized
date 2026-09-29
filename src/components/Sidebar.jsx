import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle2, Search, ChevronRight } from 'lucide-react'
import { useApp } from '../context/useApp'
import { topics } from '../context/topics'

export default function Sidebar() {
  const { theme, section, topic, setTopic, sidebarOpen, progress } = useApp()
  const [search, setSearch] = useState('')

  const currentTopics = topics[section] || []
  const filtered = currentTopics.filter(t =>
    t.label.toLowerCase().includes(search.toLowerCase())
  )

  const sectionLabel = section === 'ml' ? 'Machine Learning' : 'Deep Neural Networks'
  const totalDone = currentTopics.filter(t => progress[t.id]).length

  return (
    <AnimatePresence>
      {sidebarOpen && (
        <motion.aside
          initial={{ x: -280, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: -280, opacity: 0 }}
          transition={{ type: 'spring', bounce: 0.1, duration: 0.4 }}
          className={`fixed left-0 top-16 bottom-0 w-64 z-40 flex flex-col border-r ${
            theme === 'dark'
              ? 'bg-slate-950/95 border-white/10'
              : 'bg-white/95 border-gray-200'
          } backdrop-blur-xl overflow-hidden`}
        >
          {/* Section header */}
          <div className="p-4 border-b border-inherit">
            <div className="flex items-center justify-between mb-3">
              <span className={`text-xs font-semibold uppercase tracking-wider ${
                theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'
              }`}>{sectionLabel}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                theme === 'dark' ? 'bg-slate-800 text-gray-400' : 'bg-gray-100 text-gray-500'
              }`}>{totalDone}/{currentTopics.length}</span>
            </div>
            {/* Progress bar */}
            <div className={`h-1 rounded-full ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-200'}`}>
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
                animate={{ width: `${(totalDone / currentTopics.length) * 100}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>

          {/* Search */}
          <div className="px-3 py-2 border-b border-inherit">
            <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
              theme === 'dark' ? 'bg-slate-800' : 'bg-gray-100'
            }`}>
              <Search size={13} className="text-gray-500 shrink-0" />
              <input
                type="text"
                placeholder="Filter topics..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className={`bg-transparent text-sm outline-none w-full ${
                  theme === 'dark' ? 'text-gray-300 placeholder-gray-600' : 'text-gray-700 placeholder-gray-400'
                }`}
              />
            </div>
          </div>

          {/* Topics list */}
          <nav className="flex-1 overflow-y-auto py-2 px-2">
            {filtered.map((t, i) => {
              const done = progress[t.id]
              const active = topic === t.id
              return (
                <motion.button
                  key={t.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  onClick={() => setTopic(t.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm mb-0.5 transition-all duration-150 group ${
                    active
                      ? theme === 'dark'
                        ? 'bg-indigo-500/20 text-white border border-indigo-500/30'
                        : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                      : theme === 'dark'
                        ? 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <span className="text-base">{t.icon}</span>
                  <span className="flex-1 text-left font-medium">{t.label}</span>
                  {done ? (
                    <CheckCircle2 size={15} className="text-emerald-400 shrink-0" />
                  ) : (
                    <ChevronRight size={14} className={`shrink-0 transition-transform group-hover:translate-x-0.5 ${
                      active ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                    }`} />
                  )}
                </motion.button>
              )
            })}
          </nav>

          {/* Footer */}
          <div className={`p-4 border-t ${theme === 'dark' ? 'border-white/10' : 'border-gray-200'}`}>
            <p className={`text-xs ${theme === 'dark' ? 'text-gray-600' : 'text-gray-400'}`}>
              {currentTopics.filter(t => progress[t.id]).length} of {currentTopics.length} topics completed
            </p>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}
