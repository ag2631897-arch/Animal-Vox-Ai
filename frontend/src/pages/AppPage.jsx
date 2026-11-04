import { useLocation } from 'react-router-dom'
import AppShell from '../components/layout/AppShell'
import RecordScreen from '../components/screens/RecordScreen'
import QuickTranslate from '../components/screens/QuickTranslate'
import DetectionScreen from '../components/screens/DetectionScreen'
import HealthDashboard from '../components/screens/HealthDashboard'
import BreedIdentifier from '../components/screens/BreedIdentifier'
import EmotionTimeline from '../components/screens/EmotionTimeline'
import TranslationHistory from '../components/screens/TranslationHistory'
import Settings from '../components/screens/Settings'

const screens = {
  '/app': RecordScreen,
  '/app/': RecordScreen,
  '/app/quick': QuickTranslate,
  '/app/detect': DetectionScreen,
  '/app/health': HealthDashboard,
  '/app/breed': BreedIdentifier,
  '/app/emotions': EmotionTimeline,
  '/app/history': TranslationHistory,
  '/app/settings': Settings,
}

export default function AppPage() {
  const location = useLocation()
  const Screen = screens[location.pathname] || RecordScreen
  return <AppShell><Screen /></AppShell>
}
