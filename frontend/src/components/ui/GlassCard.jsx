import { motion } from 'framer-motion'

export default function GlassCard({ children, className = '', ...props }) {
  return <motion.section className={`glass-card ${className}`} {...props}>{children}</motion.section>
}
