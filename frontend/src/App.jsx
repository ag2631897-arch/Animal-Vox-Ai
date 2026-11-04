import { Navigate, Route, Routes } from 'react-router-dom'
import { useAppStore } from './store/useAppStore'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import AppPage from './pages/AppPage'

function ProtectedRoute({ children }) {
  const isAuthenticated = useAppStore((s) => s.isAuthenticated)
  if (!isAuthenticated) return <Navigate to="/login" replace />
  return children
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/landing" element={<LandingPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/app/*" element={<ProtectedRoute><AppPage /></ProtectedRoute>} />
      <Route path="*" element={<Navigate to="/landing" replace />} />
    </Routes>
  )
}
