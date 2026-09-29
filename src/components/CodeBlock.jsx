import { useState } from 'react'
import { Check, Copy } from 'lucide-react'
import { useApp } from '../context/useApp'

// Token-based syntax highlighter — never re-scans already-emitted HTML
const KEYWORDS = new Set([
  'from','import','def','class','return','if','else','elif','for','while',
  'in','not','and','or','True','False','None','print','as','with','pass',
  'try','except','finally','raise','lambda','yield','global','nonlocal',
])

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function span(color, text) {
  return `<span style="color:${color}">${text}</span>`
}

function tokenizeLine(raw) {
  // Tokenize one line into: comment | string | number | identifier/keyword | punctuation
  let i = 0, out = ''
  while (i < raw.length) {
    // Comment
    if (raw[i] === '#') {
      out += span('#6A9955', escHtml(raw.slice(i)))
      break
    }
    // Triple-quoted strings (simplified: won't cross lines)
    if ((raw[i] === '"' && raw.slice(i,i+3) === '"""') ||
        (raw[i] === "'" && raw.slice(i,i+3) === "'''")) {
      const q = raw.slice(i,i+3)
      const end = raw.indexOf(q, i+3)
      const tok = end === -1 ? raw.slice(i) : raw.slice(i, end+3)
      out += span('#CE9178', escHtml(tok))
      i += tok.length
      continue
    }
    // Single/double-quoted strings
    if (raw[i] === '"' || raw[i] === "'") {
      const q = raw[i]
      let j = i + 1
      while (j < raw.length && raw[j] !== q) {
        if (raw[j] === '\\') j++ // skip escaped char
        j++
      }
      const tok = raw.slice(i, j+1)
      out += span('#CE9178', escHtml(tok))
      i = j + 1
      continue
    }
    // f-string prefix: f" or f'
    if ((raw[i] === 'f' || raw[i] === 'F') && (raw[i+1] === '"' || raw[i+1] === "'")) {
      const q = raw[i+1]
      let j = i + 2
      while (j < raw.length && raw[j] !== q) {
        if (raw[j] === '\\') j++
        j++
      }
      const tok = raw.slice(i, j+1)
      out += span('#CE9178', escHtml(tok))
      i = j + 1
      continue
    }
    // Numbers
    if (/[0-9]/.test(raw[i]) || (raw[i] === '.' && /[0-9]/.test(raw[i+1] || ''))) {
      let j = i
      while (j < raw.length && /[0-9._eExX]/.test(raw[j])) j++
      out += span('#B5CEA8', escHtml(raw.slice(i, j)))
      i = j
      continue
    }
    // Identifiers / keywords
    if (/[a-zA-Z_]/.test(raw[i])) {
      let j = i
      while (j < raw.length && /[a-zA-Z0-9_]/.test(raw[j])) j++
      const word = raw.slice(i, j)
      if (KEYWORDS.has(word)) {
        out += span('#C586C0', escHtml(word))
      } else if (/^[A-Z]/.test(word)) {
        // Class names
        out += span('#4EC9B0', escHtml(word))
      } else {
        out += span('#9CDCFE', escHtml(word))
      }
      i = j
      continue
    }
    // Operators & punctuation
    if (/[=+\-*/%<>!&|^~@]/.test(raw[i])) {
      out += span('#D4D4D4', escHtml(raw[i]))
      i++
      continue
    }
    // Everything else (spaces, parens, brackets, commas, dots)
    out += escHtml(raw[i])
    i++
  }
  return out
}

function highlight(code) {
  return code.split('\n').map(tokenizeLine).join('\n')
}

export default function CodeBlock({ code, language = 'python' }) {
  const { theme } = useApp()
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={`rounded-xl overflow-hidden border ${
      theme === 'dark' ? 'bg-[#1e1e2e] border-white/10' : 'bg-gray-900 border-gray-700'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-black/20 border-b border-white/5">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/70" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
            <div className="w-3 h-3 rounded-full bg-green-500/70" />
          </div>
          <span className="text-xs text-gray-500 ml-2 font-mono">{language}</span>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
        >
          {copied ? <Check size={13} className="text-green-400" /> : <Copy size={13} />}
          <span>{copied ? 'Copied!' : 'Copy'}</span>
        </button>
      </div>
      {/* Code */}
      <div className="overflow-x-auto">
        <pre className="p-4 text-sm leading-relaxed font-mono text-gray-300">
          <code dangerouslySetInnerHTML={{ __html: highlight(code) }} />
        </pre>
      </div>
    </div>
  )
}
