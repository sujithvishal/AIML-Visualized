import { motion } from 'framer-motion'
import { useApp } from '../context/useApp'

/**
 * TheoryBlock — compact card-grid for key concepts.
 * Props:
 *   title: string          — section label (ALL CAPS, small)
 *   cards: Array<{
 *     icon: string,
 *     title: string,
 *     body: string,
 *     mono?: string        — optional formula / monospace line
 *     accent?: string      — tailwind text-color class for icon bg, e.g. 'indigo'
 *   }>
 */
export default function TheoryBlock({ title = 'Key Concepts', cards, items, cols }) {
  const { theme } = useApp()

  const bg = theme === 'dark' ? 'bg-slate-900/80 border-white/10' : 'bg-white border-gray-200'
  const cardBg = theme === 'dark' ? 'bg-slate-800/60 border-white/8' : 'bg-gray-50 border-gray-200'
  const titleColor = theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
  const headColor = theme === 'dark' ? 'text-white' : 'text-gray-900'
  const bodyColor = theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
  const monoColor = theme === 'dark' ? 'text-indigo-300' : 'text-indigo-600'

  const list = cards || items || []
  const normalizedCards = list.map(item => ({
    icon: item.icon || '',
    title: item.title || '',
    body: item.body || item.content || '',
    mono: item.mono || '',
  }))

  let colsClass = 'grid-cols-1 sm:grid-cols-3'
  if (cols) {
    if (cols === 2) colsClass = 'grid-cols-1 sm:grid-cols-2'
    else if (cols === 1) colsClass = 'grid-cols-1'
    else if (cols === 4) colsClass = 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4'
  } else {
    if (normalizedCards.length === 2) colsClass = 'grid-cols-1 sm:grid-cols-2'
    else if (normalizedCards.length === 1) colsClass = 'grid-cols-1'
  }

  return (
    <div className={`rounded-2xl border p-5 mb-6 ${bg}`}>
      {title && (
        <p className={`text-xs font-semibold uppercase tracking-wider mb-4 ${titleColor}`}>{title}</p>
      )}
      <div className={`grid gap-3 ${colsClass}`}>
        {normalizedCards.map((card, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.07 }}
            className={`rounded-xl border p-4 ${cardBg}`}
          >
            <div className="flex items-center gap-2.5 mb-2">
              {card.icon && <span className="text-xl leading-none">{card.icon}</span>}
              <span className={`font-semibold text-sm ${headColor}`}>{card.title}</span>
            </div>
            <p className={`text-xs leading-relaxed ${bodyColor}`}>{card.body}</p>
            {card.mono && (
              <p className={`mt-2 font-mono text-xs ${monoColor}`}>{card.mono}</p>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  )
}
