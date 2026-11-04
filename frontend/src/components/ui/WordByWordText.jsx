import { motion } from 'framer-motion'

export default function WordByWordText({ text, className = '' }) {
  return <span className={className} aria-label={text}>{text.split(/(\s+)/).map((part, index) => part.trim() ? <motion.span key={`${part}-${index}`} className="reveal-word" initial={{ opacity: 0, filter: 'blur(4px)', y: 4 }} animate={{ opacity: 1, filter: 'blur(0px)', y: 0 }} transition={{ duration: 0.22, delay: Math.floor(index / 2) * 0.05 }} aria-hidden="true">{part}</motion.span> : <span key={`space-${index}`} aria-hidden="true">{part}</span>)}</span>
}
