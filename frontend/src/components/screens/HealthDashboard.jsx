import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowDownRight, ArrowRight, ArrowUpRight, CheckCircle2, HeartPulse } from 'lucide-react'
import GlassCard from '../ui/GlassCard'
import FuturisticGauge from '../ui/FuturisticGauge'
import { Shimmer } from '../ui/SkeletonCard'
import ConfidenceBar from '../ui/ConfidenceBar'
import { useAppStore } from '../../store/useAppStore'
import { getHealthWeekly, getWellbeing } from '../../lib/api'

const TrendIcon = ({ trend }) => trend === 'improving' || trend === 'up' ? <ArrowUpRight size={14} /> : trend === 'declining' || trend === 'down' ? <ArrowDownRight size={14} /> : <ArrowRight size={14} />

function HealthSkeleton() {
  return <div className="health-skeleton"><Shimmer className="skeleton-gauge" /><div className="skeleton-dimensions">{[1, 2, 3, 4].map((n) => <Shimmer key={n} className="skeleton-dimension" />)}</div><Shimmer className="skeleton-line w-full" /></div>
}

function formatDimension(dimension) {
  const lowerIsBetter = dimension.direction === 'lower_is_better'
  return { ...dimension, status: lowerIsBetter ? (dimension.score < 20 ? 'Low' : dimension.score < 50 ? 'Moderate' : 'Elevated') : (dimension.score >= 60 ? 'Good' : dimension.score >= 35 ? 'Moderate' : 'Low') }
}

export default function HealthDashboard() {
  const pet = useAppStore((state) => state.pets.find((item) => item.id === state.activePetId) || state.pets[0])
  const petId = pet?.id || ''
  const [loading, setLoading] = useState(true)
  const [wellbeing, setWellbeing] = useState(null)
  const [health, setHealth] = useState(null)
  const [message, setMessage] = useState('')

  useEffect(() => {
    if (!petId) {
      setWellbeing(null)
      setHealth(null)
      setMessage('Create or select a pet profile to start tracking health signals.')
      setLoading(false)
      return
    }
    let active = true
    setLoading(true)
    setMessage('')
    Promise.allSettled([getWellbeing(petId), getHealthWeekly(petId)]).then(([wellbeingResult, healthResult]) => {
      if (!active) return
      if (wellbeingResult.status === 'fulfilled') setWellbeing(wellbeingResult.value)
      else setWellbeing(null)
      if (healthResult.status === 'fulfilled') setHealth(healthResult.value)
      else setHealth(null)
      if (wellbeingResult.status === 'rejected') setMessage('No wellbeing history yet. Record vocalizations with this profile to build a baseline.')
      setLoading(false)
    })
    return () => { active = false }
  }, [petId])

  if (loading) return <div className="health-screen"><HealthSkeleton /></div>
  if (!wellbeing) return <div className="health-screen"><div className="empty-state"><HeartPulse size={24} /><strong>No wellbeing report for {pet?.name || 'this profile'} yet</strong><p>{message || 'Record a vocalization to begin building wellbeing signals.'}</p></div></div>

  const dimensions = Object.values(wellbeing.dimensions || {}).map(formatDimension)
  const baseline = health?.baseline
  return <div className="health-screen">
    <div className="health-overview"><div className="gauge-panel"><span className="hud-corner tl" /><span className="hud-corner tr" /><span className="hud-corner bl" /><span className="hud-corner br" /><span className="eyebrow">WELLBEING INDEX</span><FuturisticGauge value={wellbeing.composite_score} /><span className="health-summary">{wellbeing.composite_label} · Trend: {wellbeing.trend}</span></div>
      <div className="health-dimensions">{dimensions.map((dimension, index) => <motion.div key={dimension.name} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, delay: index * 0.06 }}><GlassCard className="dimension-card"><span className="dimension-name">{dimension.name}</span><div className="dimension-score"><strong>{Math.round(dimension.score)}</strong><span>/100</span><TrendIcon trend={dimension.trend} /></div><div className={`health-status ${dimension.status.toLowerCase()}`}>{dimension.status}</div><ConfidenceBar value={dimension.score / 100} color={dimension.direction === 'lower_is_better' ? 'var(--accent-3)' : 'var(--accent-2)'} /></GlassCard></motion.div>)}</div>
    </div>
    <div className="health-bottom"><section className="flags-panel"><span className="eyebrow">PROFILE BASELINE</span><h2>Health Signals</h2>{baseline ? <div className="no-flags"><CheckCircle2 size={17} /> {baseline.clip_count} clip{baseline.clip_count === 1 ? '' : 's'} recorded · {baseline.established ? 'baseline established' : 'baseline still building'}</div> : <div className="no-flags"><HeartPulse size={17} /> No health baseline available yet</div>}</section>
      <section className="activity-panel"><div className="section-heading compact-heading"><div><span className="eyebrow">RECOMMENDATIONS</span><h2>Current guidance</h2></div><HeartPulse size={18} /></div><div className="event-list">{wellbeing.recommendations?.length ? wellbeing.recommendations.map((recommendation) => <GlassCard className="event-card" key={recommendation}><i /><div><strong>AnimalVox guidance</strong><p>{recommendation}</p></div></GlassCard>) : <p>No recommendations available.</p>}</div></section></div>
  </div>
}
