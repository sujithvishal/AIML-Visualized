import { useState, useEffect } from 'react'
import AppContext from './AppContextInstance'
import { topics } from './topics'

export function AppProvider({ children }) {
  const [theme, setTheme] = useState('dark')
  const [section, setSection] = useState('ml')
  const [topic, setTopic] = useState('what-is-ml')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [searchOpen, setSearchOpen] = useState(false)
  const [progress, setProgress] = useState(() => {
    try { return JSON.parse(localStorage.getItem('aiml-progress') || '{}') } catch { return {} }
  })
  const [xp, setXp] = useState(() => {
    try { return parseInt(localStorage.getItem('aiml-xp') || '0') } catch { return 0 }
  })
  const [badges, setBadges] = useState(() => {
    try { return JSON.parse(localStorage.getItem('aiml-badges') || '[]') } catch { return [] }
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  useEffect(() => {
    localStorage.setItem('aiml-progress', JSON.stringify(progress))
  }, [progress])

  useEffect(() => {
    localStorage.setItem('aiml-xp', xp)
  }, [xp])

  useEffect(() => {
    localStorage.setItem('aiml-badges', JSON.stringify(badges))
  }, [badges])

  const completeTopic = (topicId) => {
    if (progress[topicId]) return
    setProgress(p => ({ ...p, [topicId]: true }))
    addXp(100)
    checkBadges({ ...progress, [topicId]: true })
  }

  const addXp = (amount) => setXp(x => x + amount)

  const checkBadges = (newProgress) => {
    const mlDone = topics.ml.every(t => newProgress[t.id])
    const dnnDone = topics.dnn.every(t => newProgress[t.id])
    const earned = []
    if (newProgress['what-is-ml'] && !badges.includes('ml-beginner')) earned.push('ml-beginner')
    if (newProgress['linear-regression'] && !badges.includes('regression-explorer')) earned.push('regression-explorer')
    if (newProgress['what-is-dnn'] && !badges.includes('neural-network-starter')) earned.push('neural-network-starter')
    if (mlDone && !badges.includes('ml-master')) earned.push('ml-master')
    if (dnnDone && !badges.includes('dnn-master')) earned.push('dnn-master')
    if (earned.length) setBadges(b => [...b, ...earned])
  }

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  return (
    <AppContext.Provider value={{
      theme, toggleTheme,
      section, setSection,
      topic, setTopic,
      sidebarOpen, setSidebarOpen,
      searchOpen, setSearchOpen,
      progress, completeTopic,
      xp, addXp,
      badges,
    }}>
      {children}
    </AppContext.Provider>
  )
}
