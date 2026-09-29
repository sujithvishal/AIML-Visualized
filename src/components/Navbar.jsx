import { motion } from 'framer-motion'
import { Sun, Moon, Search, Menu, X, Zap } from 'lucide-react'
import { useApp } from '../context/useApp'
import { topics } from '../context/topics'

const NAV_ITEMS = [
  { id: 'ml', label: 'ML', active: true },
  { id: 'dnn', label: 'DNN', active: true },
  { id: 'nlp', label: 'NLP', active: false },
  { id: 'llm', label: 'LLM', active: false },
]

export default function Navbar() {
  const { theme, toggleTheme, section, setSection, setTopic, setSidebarOpen, sidebarOpen, setSearchOpen, xp } = useApp()

  const handleSectionClick = (item) => {
    if (!item.active) return
    setSection(item.id)
    setTopic(topics[item.id][0].id)
  }

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-16">
      <div className={`h-full px-4 flex items-center justify-between border-b ${
        theme === 'dark'
          ? 'bg-slate-950/80 backdrop-blur-xl border-white/10'
          : 'bg-white/80 backdrop-blur-xl border-gray-200'
      }`}>
        {/* Left */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setSidebarOpen(o => !o)}
            className={`p-2 rounded-lg transition-colors ${
              theme === 'dark' ? 'hover:bg-white/10 text-gray-400 hover:text-white' : 'hover:bg-gray-100 text-gray-500 hover:text-gray-900'
            }`}
          >
            {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
              <Zap size={14} className="text-white" />
            </div>
            <span className={`font-semibold text-sm tracking-tight ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              AI Learning Hub
            </span>
          </div>
        </div>

        {/* Center Nav */}
        <nav className="hidden md:flex items-center gap-1">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              onClick={() => handleSectionClick(item)}
              disabled={!item.active}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 relative ${
                !item.active
                  ? theme === 'dark' ? 'text-gray-600 cursor-not-allowed' : 'text-gray-400 cursor-not-allowed'
                  : section === item.id
                  ? theme === 'dark' ? 'text-white' : 'text-gray-900'
                  : theme === 'dark' ? 'text-gray-400 hover:text-white hover:bg-white/5' : 'text-gray-500 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              {section === item.id && item.active && (
                <motion.div
                  layoutId="nav-indicator"
                  className={`absolute inset-0 rounded-lg ${
                    theme === 'dark' ? 'bg-white/10 border border-white/15' : 'bg-indigo-50 border border-indigo-200'
                  }`}
                  transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
                />
              )}
              <span className="relative z-10 flex items-center gap-1.5">
                {item.label}
                {!item.active && (
                  <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                    theme === 'dark' ? 'bg-gray-800 text-gray-500' : 'bg-gray-100 text-gray-400'
                  }`}>Soon</span>
                )}
              </span>
            </button>
          ))}
        </nav>

        {/* Right */}
        <div className="flex items-center gap-2">
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/20 border border-indigo-500/30">
            <span className="text-xs text-indigo-300 font-medium">⚡ {xp} XP</span>
          </div>
          <button
            onClick={() => setSearchOpen(true)}
            className={`p-2 rounded-lg transition-colors ${
              theme === 'dark' ? 'hover:bg-white/10 text-gray-400 hover:text-white' : 'hover:bg-gray-100 text-gray-500 hover:text-gray-900'
            }`}
          >
            <Search size={16} />
          </button>
          <a
            href="https://www.linkedin.com/in/sujith-vishal/"
            target="_blank"
            rel="noopener noreferrer"
            className={`p-2 rounded-lg transition-colors ${
              theme === 'dark' ? 'hover:bg-white/10 text-gray-400 hover:text-white' : 'hover:bg-gray-100 text-gray-500 hover:text-gray-900'
            }`}
            aria-label="LinkedIn profile"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
            </svg>
          </a>
          <button
            onClick={toggleTheme}
            className={`p-2 rounded-lg transition-colors ${
              theme === 'dark' ? 'hover:bg-white/10 text-gray-400 hover:text-white' : 'hover:bg-gray-100 text-gray-500 hover:text-gray-900'
            }`}
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>
    </header>
  )
}
