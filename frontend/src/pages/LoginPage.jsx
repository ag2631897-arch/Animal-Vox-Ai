import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, AudioLines, Eye, EyeOff, Lock, Mail, PawPrint, User } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../store/useAppStore'

export default function LoginPage() {
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const login = useAppStore((s) => s.login)
  const isAuthenticated = useAppStore((s) => s.isAuthenticated)

  // Redirect if already logged in
  if (isAuthenticated) { navigate('/app', { replace: true }); return null }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (!email.trim() || !password.trim()) { setError('Please fill in all fields.'); return }
    if (mode === 'signup' && !name.trim()) { setError('Please enter your name.'); return }
    if (password.length < 4) { setError('Password must be at least 4 characters.'); return }

    setLoading(true)
    await new Promise((r) => setTimeout(r, 900)) // simulate network

    const users = (() => { try { return JSON.parse(localStorage.getItem('animalvox-users') || '{}') } catch { return {} } })()

    if (mode === 'signup') {
      if (users[email]) { setError('An account with this email already exists.'); setLoading(false); return }
      users[email] = { name: name.trim(), email: email.trim(), password, createdAt: new Date().toISOString() }
      localStorage.setItem('animalvox-users', JSON.stringify(users))
      login({ name: name.trim(), email: email.trim() })
    } else {
      const user = users[email]
      if (!user || user.password !== password) { setError('Invalid email or password.'); setLoading(false); return }
      login({ name: user.name, email: user.email })
    }

    setLoading(false)
    navigate('/app')
  }

  const switchMode = () => { setMode(mode === 'login' ? 'signup' : 'login'); setError('') }

  return (
    <div className="login-page">
      <div className="login-bg-grid" />
      <div className="login-bg-glow" />
      <div className="login-bg-glow login-bg-glow-2" />

      <div className="login-container">
        {/* Brand */}
        <motion.div
          className="login-brand"
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <span className="brand-icon">
            <PawPrint size={30} />
            <AudioLines size={15} />
          </span>
          <h1>AnimalVox</h1>
          <p>Hear what they're saying</p>
        </motion.div>

        {/* Card */}
        <motion.div
          className="login-card"
          initial={{ opacity: 0, y: 20, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.35, delay: 0.08 }}
        >
          {/* Tabs */}
          <div className="login-tabs">
            <button className={mode === 'login' ? 'active' : ''} onClick={() => switchMode()}>Sign In</button>
            <button className={mode === 'signup' ? 'active' : ''} onClick={() => switchMode()}>Create Account</button>
            <span className="login-tab-indicator" style={{ transform: `translateX(${mode === 'login' ? '0%' : '100%'})` }} />
          </div>

          <form onSubmit={handleSubmit} autoComplete="off">
            <AnimatePresence mode="wait">
              {mode === 'signup' && (
                <motion.label
                  key="name-field"
                  className="login-field"
                  initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                  animate={{ opacity: 1, height: 'auto', marginBottom: 14 }}
                  exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <User size={16} />
                  <input type="text" placeholder="Full name" value={name} onChange={(e) => setName(e.target.value)} />
                </motion.label>
              )}
            </AnimatePresence>

            <label className="login-field">
              <Mail size={16} />
              <input type="email" placeholder="Email address" value={email} onChange={(e) => setEmail(e.target.value)} required />
            </label>

            <label className="login-field">
              <Lock size={16} />
              <input
                type={showPw ? 'text' : 'password'}
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button type="button" className="pw-toggle" onClick={() => setShowPw((v) => !v)} tabIndex={-1}>
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </label>

            <AnimatePresence>
              {error && (
                <motion.p
                  className="login-error"
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  {error}
                </motion.p>
              )}
            </AnimatePresence>

            <button type="submit" className="login-submit" disabled={loading}>
              {loading ? (
                <span className="login-spinner" />
              ) : (
                <>
                  {mode === 'login' ? 'Sign In' : 'Create Account'}
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </form>

          <p className="login-switch">
            {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
            <button onClick={switchMode}>
              {mode === 'login' ? 'Create one' : 'Sign in'}
            </button>
          </p>
        </motion.div>

        <p className="login-disclaimer">Demo authentication · data stored locally</p>
      </div>
    </div>
  )
}
