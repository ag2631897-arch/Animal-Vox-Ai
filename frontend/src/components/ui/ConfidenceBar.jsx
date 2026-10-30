import { motion } from 'framer-motion'

export default function ConfidenceBar({ value, color = 'var(--accent)', className = '' }) {
  const percent = Math.max(0, Math.min(100, value <= 1 ? value * 100 : value))
  return <div className={`confidence-track ${className}`} role="meter" aria-valuenow={Math.round(percent)} aria-valuemin="0" aria-valuemax="100">
    <motion.span className="confidence-fill" style={{ background: color, boxShadow: `0 0 12px ${color}55` }} initial={{ width: 0 }} whileInView={{ width: `${percent}%` }} viewport={{ once: true }} transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }} />
  </div>
}
