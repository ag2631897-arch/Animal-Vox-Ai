import { useEffect, useState } from 'react'
import { CalendarDays, Smile } from 'lucide-react'
import GlassCard from '../ui/GlassCard'
import { Shimmer } from '../ui/SkeletonCard'
import { useAppStore } from '../../store/useAppStore'
import { getEmotionWeekly } from '../../lib/api'

const emotionColors = ['#00C9A7', '#4CAF6B', '#F5A623', '#EF4444', '#A78BFA', '#35C6D8']
const emotionNames = {
  E01: 'Contentment', E02: 'Stress', E03: 'Fear', E04: 'Hunger', E05: 'Territorial / Defensive',
  E06: 'Social bonding', E07: 'Play / excitement', E08: 'Frustration', E09: 'Curiosity', E10: 'Isolation',
}

function EmotionSkeleton() {
  return <div className="emotion-skeleton"><Shimmer className="skeleton-line w-40" /><Shimmer className="skeleton-emotion-stack" /><div className="skeleton-emotion-legend">{[1, 2, 3, 4].map((item) => <Shimmer key={item} />)}</div></div>
}

export default function EmotionTimeline() {
  const petId = useAppStore((state) => state.activePetId)
  const pet = useAppStore((state) => state.pets.find((item) => item.id === petId))
  const [loading, setLoading] = useState(true)
  const [digest, setDigest] = useState(null)

  useEffect(() => {
    let active = true
    setLoading(true)
    getEmotionWeekly(petId).then((result) => { if (active) setDigest(result) }).catch(() => { if (active) setDigest(null) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [petId])

  if (loading) return <div className="emotion-screen"><EmotionSkeleton /></div>
  if (!digest) return <div className="emotion-screen"><div className="empty-state"><Smile size={24} /><strong>No emotional diary for {pet?.name || 'this profile'} yet</strong><p>Record vocalizations to build a behavior-aligned diary over time.</p></div></div>

  const distribution = Object.entries(digest.emotion_distribution || {}).map(([id, value], index) => ({ id, name: emotionNames[id] || id, value: Math.round(value * 100), color: emotionColors[index % emotionColors.length] })).sort((a, b) => b.value - a.value)
  return <div className="emotion-screen">
    <div className="screen-intro emotion-controls"><div><span className="eyebrow">MOOD SIGNALS / WEEKLY DIARY</span><p>Patterns are clues to context, not a diagnosis.</p></div><span className="local-badge"><CalendarDays size={14} /> {digest.total_entries} entries</span></div>
    <section className="emotion-distribution"><div className="section-heading compact-heading"><div><span className="eyebrow">EMOTION DISTRIBUTION</span><h2>Signals for {pet?.name || 'this profile'}</h2></div></div><div className="emotion-stack" aria-label="Stacked emotion distribution">{distribution.map((item) => <div key={item.id} className="emotion-segment" style={{ width: `${item.value}%`, background: item.color }} title={`${item.name}: ${item.value}%`} />)}</div><div className="emotion-legend">{distribution.map((item) => <span key={item.id}><i style={{ background: item.color }} />{item.name}<b>{item.value}%</b></span>)}</div></section>
    <section className="notable-section"><div className="section-heading compact-heading"><div><span className="eyebrow">WEEKLY SUMMARY</span><h2>Behavioral observations</h2></div></div><div className="event-list"><GlassCard className="event-card"><i /><div><small>{digest.week_start?.slice(0, 10)} to {digest.week_end?.slice(0, 10)}</small><strong>Dominant signal: {emotionNames[digest.dominant_emotion] || digest.dominant_emotion}</strong><p>{digest.stress_episodes} stress episode{digest.stress_episodes === 1 ? '' : 's'} · {digest.contentment_episodes} contentment episode{digest.contentment_episodes === 1 ? '' : 's'} · {Math.round((digest.mood_stability || 0) * 100)}% mood stability.</p></div></GlassCard>{(digest.anomalies || []).map((anomaly) => <GlassCard className="event-card" key={anomaly}><i /><div><strong>Pattern to review</strong><p>{anomaly}</p></div></GlassCard>)}</div></section>
  </div>
}
