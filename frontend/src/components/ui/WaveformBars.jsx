import { useEffect, useState } from 'react'

export default function WaveformBars({ active = false, count = 20, compact = false }) {
  const [heights, setHeights] = useState(() => Array.from({ length: count }, (_, i) => 18 + ((i * 29) % 35)))
  useEffect(() => {
    if (!active) return undefined
    const timer = window.setInterval(() => setHeights(Array.from({ length: count }, () => 8 + Math.round(Math.random() * 54))), 110)
    return () => window.clearInterval(timer)
  }, [active, count])
  return <div className={`waveform-bars ${compact ? 'waveform-compact' : ''} ${active ? 'waveform-live' : ''}`} aria-hidden="true">{heights.map((height, index) => <i key={index} style={{ height: `${active ? height : Math.max(12, height * 0.55)}%`, animationDelay: `${index * 0.07}s` }} />)}</div>
}
