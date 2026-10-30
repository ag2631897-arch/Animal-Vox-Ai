import { useEffect, useState } from 'react'

export default function RetroDataPanel({ result }) {
  const [visible, setVisible] = useState(0)
  const primaryBehavior = result?.behaviors?.[0]
  const confidence = primaryBehavior?.score ?? 0
  const lines = [
    ['SPECIES_GROUP', result?.species || 'unknown'],
    ['BEHAVIOR', primaryBehavior?.name || result?.vocalization_type || 'unknown'],
    ['CONFIDENCE', Number(confidence).toFixed(4)],
    ['INTENSITY', Number(result?.intensity || 0).toFixed(4)],
    ['URGENCY', Number(result?.urgency || 0).toFixed(4)],
    ['PROCESSING_MS', String(result?.processing_time_ms || 0)],
  ]
  useEffect(() => {
    setVisible(0)
    const timer = window.setInterval(() => setVisible((value) => Math.min(value + 1, lines.length)), 120)
    return () => window.clearInterval(timer)
  }, [])
  return <div className="retro-panel" aria-label="Raw translation data">
    {lines.slice(0, visible).map(([key, value], i) => <div className="retro-line" key={key} style={{ animationDelay: `${i * 0.02}s` }}><span>&gt; {key.padEnd(17, ' ')}</span><b>{value}</b></div>)}
    <span className="terminal-cursor" aria-hidden="true">▋</span>
  </div>
}
