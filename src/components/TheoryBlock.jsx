import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useApp } from '../context/useApp'

/** A compact, selectable concept explorer used throughout the lessons. */
export default function TheoryBlock({ title = 'Key Concepts', cards, items }) {
  const { theme } = useApp()
  const [active, setActive] = useState(0)
  const dark = theme === 'dark'
  const list = (cards || items || []).map(item => ({
    icon: item.icon || '',
    title: item.title || '',
    body: item.body || item.content || '',
    mono: item.mono || '',
  }))

  if (!list.length) return null
  const selected = list[Math.min(active, list.length - 1)]

  return (
    <section className={`mb-8 ${dark ? 'text-slate-100' : 'text-slate-900'}`} aria-label={title || 'Concept explorer'}>
      {title && <div className="mb-3 flex items-center gap-3">
        <span className={`text-[11px] font-semibold uppercase tracking-[0.18em] ${dark ? 'text-indigo-300' : 'text-indigo-600'}`}>{title}</span>
        <span className={`h-px flex-1 ${dark ? 'bg-white/10' : 'bg-slate-200'}`} />
        <span className={`text-[11px] tabular-nums ${dark ? 'text-slate-500' : 'text-slate-400'}`}>{String(active + 1).padStart(2, '0')} / {String(list.length).padStart(2, '0')}</span>
      </div>}

      <div className={`overflow-hidden rounded-2xl border ${dark ? 'border-white/10 bg-slate-900/55' : 'border-slate-200 bg-white/80'}`}>
        <div className={`flex gap-1 overflow-x-auto px-2 pt-2 ${dark ? 'border-b border-white/8' : 'border-b border-slate-100'}`} role="tablist" aria-label={title || 'Concepts'}>
          {list.map((item, i) => (
            <button key={`${item.title}-${i}`} type="button" role="tab" aria-selected={active === i} onClick={() => setActive(i)}
              className={`relative flex shrink-0 items-center gap-2 rounded-t-xl px-3.5 py-3 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-indigo-400 ${active === i ? (dark ? 'text-white' : 'text-indigo-700') : (dark ? 'text-slate-400 hover:text-slate-200' : 'text-slate-500 hover:text-slate-800')}`}>
              {item.icon && <span aria-hidden="true">{item.icon}</span>}
              <span className="font-medium">{item.title}</span>
              {active === i && <motion.span layoutId={`concept-tab-${title}`} className="absolute inset-x-3 bottom-0 h-0.5 rounded-full bg-indigo-500" />}
            </button>
          ))}
        </div>

        <div className="min-h-32 px-5 py-5 sm:px-6">
          <AnimatePresence mode="wait">
            <motion.div key={active} role="tabpanel" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.16 }}>
              <h3 className={`mb-2 text-lg font-semibold tracking-tight ${dark ? 'text-white' : 'text-slate-900'}`}>{selected.icon && <span className="mr-2">{selected.icon}</span>}{selected.title}</h3>
              <p className={`max-w-3xl text-sm leading-7 ${dark ? 'text-slate-300' : 'text-slate-600'}`}>{selected.body}</p>
              {selected.mono && <p className={`mt-4 inline-flex rounded-lg px-3 py-2 font-mono text-xs ${dark ? 'bg-indigo-400/10 text-indigo-200' : 'bg-indigo-50 text-indigo-700'}`}>{selected.mono}</p>}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
      <div className="mt-2 flex justify-end gap-2">
        <button type="button" onClick={() => setActive((active - 1 + list.length) % list.length)} aria-label="Previous concept" className={`rounded-full px-3 py-1.5 text-xs transition-colors ${dark ? 'text-slate-400 hover:bg-white/8 hover:text-white' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'}`}>← Previous</button>
        <button type="button" onClick={() => setActive((active + 1) % list.length)} aria-label="Next concept" className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${dark ? 'text-indigo-200 hover:bg-indigo-400/10' : 'text-indigo-700 hover:bg-indigo-50'}`}>Next →</button>
      </div>
    </section>
  )
}
