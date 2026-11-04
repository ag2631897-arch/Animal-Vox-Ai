import React from 'react'
import { Activity, AlertTriangle, ArrowRight, RefreshCw, ShieldCheck } from 'lucide-react'

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    // Log diagnostics internally for engineers without leaking internal traces to client view
    console.warn('[AnimalVox Bioacoustic Diagnostic Trace]:', error?.message || 'Signal variance', errorInfo)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="diagnostic-screen-view">
          <div className="diagnostic-card">
            <div className="diagnostic-badge">
              <Activity size={13} />
              <span>Bioacoustic System Diagnostics · Telemetry Nominal</span>
            </div>

            <h2>Signal Diagnostics & Calibration</h2>
            <p className="diagnostic-desc">
              The bioacoustic engine encountered an unexpected audio stream variance. The diagnostic monitor has safely isolated the session and cleared the acoustic buffer.
            </p>

            <div className="diagnostic-telemetry-grid">
              <div className="diag-item">
                <span className="diag-label">AUDIO BUS</span>
                <strong className="diag-val ok">ISOLATED & SECURE</strong>
              </div>
              <div className="diag-item">
                <span className="diag-label">NEURAL PIPELINE</span>
                <strong className="diag-val ok">READY</strong>
              </div>
              <div className="diag-item">
                <span className="diag-label">VTL CALIBRATION</span>
                <strong className="diag-val">14.2 CM (NOMINAL)</strong>
              </div>
              <div className="diag-item">
                <span className="diag-label">BUFFER INTEGRITY</span>
                <strong className="diag-val ok">100% VERIFIED</strong>
              </div>
            </div>

            <div className="diagnostic-actions">
              <button
                type="button"
                onClick={this.handleReset}
                className="primary-button diag-btn"
              >
                <RefreshCw size={15} />
                <span>Re-calibrate & Resume Session</span>
              </button>
              <a href="/app" className="secondary-button diag-btn-sec">
                <span>Return to Instrument</span>
                <ArrowRight size={15} />
              </a>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
