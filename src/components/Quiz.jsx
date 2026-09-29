import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle2, XCircle, ChevronRight, Trophy } from 'lucide-react'
import { useApp } from '../context/useApp'

export default function Quiz({ questions, topicId }) {
  const { theme, addXp, completeTopic } = useApp()
  const [current, setCurrent] = useState(0)
  const [selected, setSelected] = useState(null)
  const [answered, setAnswered] = useState(false)
  const [score, setScore] = useState(0)
  const [done, setDone] = useState(false)
  const [results, setResults] = useState([])

  const q = questions[current]

  const handleSelect = (idx) => {
    if (answered) return
    setSelected(idx)
    setAnswered(true)
    const correct = idx === q.correct
    if (correct) {
      setScore(s => s + 1)
      addXp(20)
    }
    setResults(r => [...r, correct])
  }

  const handleNext = () => {
    if (current + 1 >= questions.length) {
      setDone(true)
      if (score + (results[current] ? 1 : 0) >= Math.ceil(questions.length * 0.6)) {
        completeTopic(topicId)
      }
    } else {
      setCurrent(c => c + 1)
      setSelected(null)
      setAnswered(false)
    }
  }

  const handleRestart = () => {
    setCurrent(0); setSelected(null); setAnswered(false)
    setScore(0); setDone(false); setResults([])
  }

  const card = `rounded-2xl border p-6 ${
    theme === 'dark' ? 'bg-slate-900 border-white/10' : 'bg-white border-gray-200'
  }`

  if (done) {
    const pct = Math.round(((score) / questions.length) * 100)
    const passed = score >= Math.ceil(questions.length * 0.6)
    return (
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className={card}>
        <div className="text-center">
          <div className="text-5xl mb-4">{passed ? '🏆' : '📚'}</div>
          <h3 className={`text-xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
            {passed ? 'Excellent Work!' : 'Keep Practicing!'}
          </h3>
          <p className={`mb-6 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            You scored {score}/{questions.length} ({pct}%)
          </p>
          <div className="flex gap-2 justify-center mb-6">
            {results.map((r, i) => (
              <div key={i} className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold ${r ? 'bg-emerald-500' : 'bg-red-500'}`}>
                {i + 1}
              </div>
            ))}
          </div>
          {passed && (
            <div className="flex items-center justify-center gap-2 mb-6 p-3 rounded-xl bg-indigo-500/20 border border-indigo-500/30">
              <Trophy size={16} className="text-indigo-400" />
              <span className="text-sm text-indigo-300 font-medium">+100 XP earned • Topic Completed!</span>
            </div>
          )}
          <button
            onClick={handleRestart}
            className="px-6 py-2.5 rounded-xl bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-medium transition-colors"
          >
            Try Again
          </button>
        </div>
      </motion.div>
    )
  }

  return (
    <div className={card}>
      <div className="flex items-center justify-between mb-5">
        <span className={`text-xs font-medium uppercase tracking-wider ${theme === 'dark' ? 'text-indigo-400' : 'text-indigo-600'}`}>
          Knowledge Check
        </span>
        <div className="flex items-center gap-2">
          <span className={`text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>{current + 1} / {questions.length}</span>
          <div className="flex gap-1">
            {questions.map((_, i) => (
              <div key={i} className={`w-1.5 h-1.5 rounded-full transition-colors ${
                i < current ? 'bg-indigo-500' : i === current ? 'bg-indigo-400' : theme === 'dark' ? 'bg-slate-700' : 'bg-gray-200'
              }`} />
            ))}
          </div>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={current}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.2 }}
        >
          <p className={`text-base font-semibold mb-5 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
            {q.question}
          </p>

          <div className="space-y-2.5 mb-5">
            {q.options.map((opt, idx) => {
              const isCorrect = idx === q.correct
              const isSelected = idx === selected
              let style = theme === 'dark'
                ? 'border-white/10 hover:border-indigo-400/50 hover:bg-indigo-500/10'
                : 'border-gray-200 hover:border-indigo-300 hover:bg-indigo-50'
              if (answered) {
                if (isCorrect) style = 'border-emerald-500/50 bg-emerald-500/10'
                else if (isSelected && !isCorrect) style = 'border-red-500/50 bg-red-500/10'
                else style = theme === 'dark' ? 'border-white/5 opacity-50' : 'border-gray-100 opacity-50'
              }
              return (
                <button
                  key={idx}
                  onClick={() => handleSelect(idx)}
                  disabled={answered}
                  className={`w-full flex items-center justify-between px-4 py-3 rounded-xl border text-sm text-left transition-all duration-150 ${style} ${
                    !answered ? 'cursor-pointer' : 'cursor-default'
                  } ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}
                >
                  <span>{opt}</span>
                  {answered && isCorrect && <CheckCircle2 size={16} className="text-emerald-400 shrink-0" />}
                  {answered && isSelected && !isCorrect && <XCircle size={16} className="text-red-400 shrink-0" />}
                </button>
              )
            })}
          </div>

          {answered && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`p-3 rounded-xl mb-4 text-sm ${
                selected === q.correct
                  ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300'
                  : 'bg-red-500/10 border border-red-500/20 text-red-300'
              }`}
            >
              {q.explanation}
            </motion.div>
          )}
        </motion.div>
      </AnimatePresence>

      {answered && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={handleNext}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-medium transition-colors"
        >
          {current + 1 >= questions.length ? 'See Results' : 'Next Question'}
          <ChevronRight size={14} />
        </motion.button>
      )}
    </div>
  )
}
