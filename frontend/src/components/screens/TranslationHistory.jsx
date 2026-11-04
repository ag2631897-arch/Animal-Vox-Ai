import { useEffect, useMemo, useState } from 'react'
import { Search, Volume2 } from 'lucide-react'
import { motion } from 'framer-motion'
import ConfidenceBar from '../ui/ConfidenceBar'
import { PetAvatar, SpeciesMark } from '../ui/SpeciesMark'
import { useAppStore } from '../../store/useAppStore'
import { createSpeechUtterance } from '../../hooks/speech'
import { readableSignal, readableSpecies } from '../../lib/labels'

function historyItem(entry, pets) {
  const timestamp = new Date(entry.createdAt)
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(today.getDate() - 1)
  const day = timestamp.toDateString() === today.toDateString() ? 'Today' : timestamp.toDateString() === yesterday.toDateString() ? 'Yesterday' : timestamp.toLocaleDateString()
  const pet = pets.find((item) => item.id === entry.petId)
  const isQuick = !entry.petId
  return {
    id: entry.id,
    day,
    pet,
    isQuick,
    name: isQuick ? 'Quick translate' : pet?.name || entry.petName || 'Saved translation',
    species: isQuick ? readableSpecies(entry.petName, '') : readableSpecies(pet?.species, ''),
    time: timestamp.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
    behavior: readableSignal(entry.behavior),
    text: entry.translation,
    confidence: typeof entry.confidence === 'number' ? entry.confidence : null,
  }
}

export default function TranslationHistory() {
  const [search, setSearch] = useState('')
  const [petFilter, setPetFilter] = useState('All pets')
  const [speakingId, setSpeakingId] = useState(null)
  const pets = useAppStore((state) => state.pets)
  const voice = useAppStore((state) => state.voice)
  const speed = useAppStore((state) => state.speed)
  const translationHistory = useAppStore((state) => state.translationHistory)
  const petNames = ['All pets', ...pets.map((pet) => pet.name)]
  const entries = useMemo(() => {
    const seen = new Set()
    return translationHistory.reduce((unique, entry) => {
      const duplicateKey = `${entry.petId || entry.petName}|${entry.behavior}|${entry.translation}`
      if (seen.has(duplicateKey)) return unique
      seen.add(duplicateKey)
      unique.push(historyItem(entry, pets))
      return unique
    }, [])
  }, [translationHistory, pets])
  const filtered = useMemo(() => entries.filter((item) => (petFilter === 'All pets' || item.name === petFilter) && `${item.name} ${item.species} ${item.behavior} ${item.text}`.toLowerCase().includes(search.toLowerCase())), [entries, search, petFilter])

  useEffect(() => () => { window.speechSynthesis?.cancel() }, [])
  const speak = (item) => {
    if (speakingId === item.id) { window.speechSynthesis?.cancel(); setSpeakingId(null); return }
    if (!window.speechSynthesis) return
    window.speechSynthesis.cancel()
    const utterance = createSpeechUtterance(item.text, voice, speed)
    utterance.onend = () => setSpeakingId(null)
    setSpeakingId(item.id)
    window.speechSynthesis.speak(utterance)
  }

  const groups = [...new Set(filtered.map((item) => item.day))]
  return <div className="history-screen">
    <div className="history-toolbar"><label className="history-search"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search interpretations" aria-label="Search interpretations" />{search && <button type="button" onClick={() => setSearch('')} aria-label="Clear search">Clear</button>}</label><select className="history-filter" value={petFilter} onChange={(event) => setPetFilter(event.target.value)} aria-label="Filter by pet">{petNames.map((name) => <option key={name}>{name}</option>)}</select></div>
    <p className="history-demo-note">Interpretations are stored in this browser. The frontend does not retain original recordings.</p>
    {filtered.length ? groups.map((day) => <section className="history-group" key={day}><div className="history-date"><span>{day}</span><i /></div>{filtered.filter((item) => item.day === day).map((item, index) => <motion.article className="history-item" key={item.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2, delay: index * 0.04 }}>
      {item.pet ? <PetAvatar pet={item.pet} className="history-animal" size={18} /> : <div className="history-animal"><SpeciesMark species={item.species} size={18} /></div>}
      <div className="history-copy"><div className="history-meta"><strong>{item.name}</strong>{item.isQuick && item.species && <span>{item.species}</span>}<span>{item.behavior}</span></div><p>{item.text}</p><div className="history-actions">{item.confidence !== null && <><span className="history-confidence">{Math.round(item.confidence * 100)}% confidence</span><ConfidenceBar value={item.confidence} /></>}<button type="button" className={`tiny-action ${speakingId === item.id ? 'active' : ''}`} aria-label={`Speak interpretation for ${item.name}`} onClick={() => speak(item)}><Volume2 size={14} /></button></div></div>
      <time className="history-time">{item.time}</time>
    </motion.article>)}</section>) : <div className="empty-state"><Search size={24} /><strong>No saved interpretations</strong><p>Record or upload an animal vocalization to create your first result.</p></div>}
  </div>
}
