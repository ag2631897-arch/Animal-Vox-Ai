import { motion } from 'framer-motion'

export default function FuturisticGauge({ value = 74, size = 180, label = 'Overall Wellbeing' }) {
  const radius = 72
  const circumference = 2 * Math.PI * radius
  const progress = circumference * (1 - Math.max(0, Math.min(100, value)) / 100)
  return <div className="gauge-wrap" style={{ width: size, height: size }}>
    <svg viewBox="0 0 180 180" className="gauge-svg" aria-label={`${label}: ${value} out of 100`} role="img">
      <circle className="gauge-glow" cx="90" cy="90" r={radius} />
      <circle className="gauge-track" cx="90" cy="90" r={radius} />
      <motion.circle className="gauge-value" cx="90" cy="90" r={radius} strokeDasharray={circumference} initial={{ strokeDashoffset: circumference }} animate={{ strokeDashoffset: progress }} transition={{ duration: 0.8, ease: 'easeOut' }} />
    </svg>
    <div className="gauge-center"><strong>{value}</strong><span>/100</span><small>{label}</small></div>
  </div>
}
