import { useEffect, useMemo, useState } from 'react'
import { Search, Volume2 } from 'lucide-react'
import { motion } from 'framer-motion'
import ConfidenceBar from '../ui/ConfidenceBar'
import { useAppStore } from '../../store/useAppStore'
import { createSpeechUtterance } from '../../hooks/speech'

function historyItem(entry) {
  const timestamp = new Date(entry.createdAt)
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(today.getDate() - 1)
  const day = timestamp.toDateString() === today.toDateString()
    ? 'Today'
    : timestamp.toDateString() === yesterday.toDateString()
      ? 'Yesterday'
      : timestamp.toLocaleDateString()
  return {
    id: entry.id,
    day,
    pet: entry.petName,
    emoji: entry.emoji,
    time: timestamp.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
    behavior: entry.behavior,
    text: entry.translation,
    confidence: entry.confidence,
  }
}

export default function TranslationHistory() {
  const [search, setSearch] = useState('')
  const [petFilter, setPetFilter] = useState('All Pets')
  const [speakingId, setSpeakingId] = useState(null)
  const pets = useAppStore((state) => state.pets)
  const voice = useAppStore((state) => state.voice)
  const speed = useAppStore((state) => state.speed)
  const translationHistory = useAppStore((state) => state.translationHistory)
  const petNames = ['All Pets', ...pets.map((pet) => pet.name)]
  const entries = useMemo(() => translationHistory.map(historyItem), [translationHistory])
  const filtered = useMemo(() => entries.filter((item) => (petFilter === 'All Pets' || item.pet === petFilter) && `${item.pet} ${item.behavior} ${item.text}`.toLowerCase().includes(search.toLowerCase())), [entries, search, petFilter])

  useEffect(() => () => { window.speechSynthesis?.cancel() }, [])

  const speak = (item) => {
    if (speakingId === item.id) {
      window.speechSynthesis?.cancel()
      setSpeakingId(null)
      return
    }
    if (!window.speechSynthesis) return
    window.speechSynthesis.cancel()
    const utterance = createSpeechUtterance(item.text, voice, speed)
    utterance.onend = () => setSpeakingId(null)
    setSpeakingId(item.id)
    window.speechSynthesis.speak(utterance)
  }

  const groups = [...new Set(filtered.map((item) => item.day))]
  return <div className="history-screen">
    <div className="history-toolbar"><label className="history-search"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search interpretations…" aria-label="Search interpretations" />{search && <button onClick={() => setSearch('')} aria-label="Clear search">×</button>}</label><select className="history-filter" value={petFilter} onChange={(event) => setPetFilter(event.target.value)} aria-label="Filter by pet">{petNames.map((name) => <option key={name}>{name}</option>)}</select></div>
    <p className="history-demo-note">Recent API interpretations are stored in this browser. Original recordings are not retained by the frontend.</p>
    {filtered.length ? groups.map((day) => <section className="history-group" key={day}><div className="history-date"><span>{day}</span><i /></div>{filtered.filter((item) => item.day === day).map((item, index) => <motion.article className="history-item" key={item.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, delay: index * 0.05 }}><div className="history-animal">{item.emoji}</div><div className="history-copy"><div className="history-meta"><strong>{item.emoji} {item.pet}</strong><span>{item.time}</span><span>{item.behavior}</span></div><p>{item.text}</p><div className="history-actions"><span className="history-confidence">{Math.round(item.confidence * 100)}% confidence</span><ConfidenceBar value={item.confidence} /><button className={`tiny-action ${speakingId === item.id ? 'active' : ''}`} aria-label={`Speak interpretation for ${item.pet}`} onClick={() => speak(item)}><Volume2 size={14} /></button></div></div></motion.article>)}</section>) : <div className="empty-state"><span>⌕</span><strong>No saved interpretations</strong><p>Record or upload an animal vocalization to create your first result.</p></div>}
  </div>
}
