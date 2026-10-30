import { Radar } from 'lucide-react'

export default function AutoSpeciesToggle({ enabled, onToggle }) {
  return (
    <div className="auto-species-toggle">
      <div className="ast-left">
        <Radar size={15} className={enabled ? 'ast-icon-active' : ''} />
        <div className="ast-text">
          <span className="ast-label">Auto Species Detection</span>
          <span className="ast-status">
            {enabled ? 'Detecting species automatically' : 'Using manual selection'}
          </span>
        </div>
      </div>
      <button
        className={`toggle-switch ${enabled ? 'on' : ''}`}
        onClick={onToggle}
        role="switch"
        aria-checked={enabled}
        aria-label="Toggle auto species detection"
      >
        <span className="toggle-knob" />
      </button>
    </div>
  )
}
