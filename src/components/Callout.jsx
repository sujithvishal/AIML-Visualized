import { useApp } from '../context/useApp'

const VARIANTS = {
  info:    { icon: '💡', border: 'border-indigo-500/30', bg: 'bg-indigo-500/10', text: 'text-indigo-300', heading: 'text-indigo-200' },
  formula: { icon: '📐', border: 'border-cyan-500/30',   bg: 'bg-cyan-500/10',   text: 'text-cyan-300',   heading: 'text-cyan-200'   },
  warning: { icon: '⚠️', border: 'border-amber-500/30',  bg: 'bg-amber-500/10',  text: 'text-amber-300',  heading: 'text-amber-200'  },
  success: { icon: '✅', border: 'border-emerald-500/30',bg: 'bg-emerald-500/10',text: 'text-emerald-300',heading: 'text-emerald-200'},
  analogy: { icon: '🎯', border: 'border-purple-500/30', bg: 'bg-purple-500/10', text: 'text-purple-300',  heading: 'text-purple-200' },
}

export default function Callout({ type = 'info', title, children, mono }) {
  const { theme } = useApp()
  const v = VARIANTS[type]
  const lightBg = type === 'formula' ? 'bg-cyan-50 border-cyan-200' :
                  type === 'warning' ? 'bg-amber-50 border-amber-200' :
                  type === 'success' ? 'bg-emerald-50 border-emerald-200' :
                  type === 'analogy' ? 'bg-purple-50 border-purple-200' :
                  'bg-indigo-50 border-indigo-200'
  const lightText = type === 'formula' ? 'text-cyan-700' : type === 'warning' ? 'text-amber-700' :
                    type === 'success' ? 'text-emerald-700' : type === 'analogy' ? 'text-purple-700' : 'text-indigo-700'

  return (
    <div className={`rounded-xl border p-4 mb-4 ${theme === 'dark' ? `${v.bg} ${v.border}` : lightBg}`}>
      <div className="flex items-start gap-3">
        <span className="text-lg mt-0.5 shrink-0">{v.icon}</span>
        <div className="flex-1 min-w-0">
          {title && (
            <p className={`font-semibold text-sm mb-1 ${theme === 'dark' ? v.heading : lightText}`}>{title}</p>
          )}
          <p className={`text-sm leading-relaxed ${theme === 'dark' ? v.text : lightText}`}>{children}</p>
          {mono && (
            <p className={`mt-2 font-mono text-sm font-medium ${theme === 'dark' ? 'text-white/80' : 'text-gray-900'}`}>{mono}</p>
          )}
        </div>
      </div>
    </div>
  )
}
