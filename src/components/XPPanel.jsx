import { motion } from 'framer-motion'
import { useApp } from '../context/useApp'

const BADGE_META = {
  'ml-beginner': { label: 'ML Beginner', icon: '🧠', desc: 'Completed What is ML?' },
  'regression-explorer': { label: 'Regression Explorer', icon: '📈', desc: 'Mastered Linear Regression' },
  'neural-network-starter': { label: 'Neural Network Starter', icon: '🕸️', desc: 'Completed What is DNN?' },
  'ml-master': { label: 'ML Master', icon: '🏆', desc: 'Completed all ML topics' },
  'dnn-master': { label: 'DNN Master', icon: '🎓', desc: 'Completed all DNN topics' },
}

export default function XPPanel() {
  const { theme, xp, badges } = useApp()
  const level = Math.floor(xp / 200) + 1
  const xpInLevel = xp % 200
  const xpToNext = 200

  return (
    <div className={`rounded-2xl border p-5 ${
      theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-white border-gray-200'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Your Progress</h3>
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/20 border border-indigo-500/30">
          <span className="text-xs text-indigo-300 font-semibold">⚡ {xp} XP</span>
        </div>
      </div>

      {/* Level */}
      <div className="mb-4">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}>Level {level}</span>
          <span className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}>{xpInLevel}/{xpToNext} XP</span>
        </div>
        <div className={`h-2 rounded-full ${theme === 'dark' ? 'bg-slate-800' : 'bg-gray-200'}`}>
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
            animate={{ width: `${(xpInLevel / xpToNext) * 100}%` }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Badges */}
      <div>
        <p className={`text-xs font-medium mb-2 ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
          BADGES ({badges.length})
        </p>
        <div className="flex flex-wrap gap-2">
          {Object.entries(BADGE_META).map(([id, meta]) => {
            const earned = badges.includes(id)
            return (
              <div
                key={id}
                title={meta.desc}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs transition-all ${
                  earned
                    ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300'
                    : theme === 'dark'
                      ? 'bg-slate-800/50 border-white/5 text-gray-600 grayscale'
                      : 'bg-gray-50 border-gray-200 text-gray-400 grayscale'
                }`}
              >
                <span>{meta.icon}</span>
                <span className="font-medium">{meta.label}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
