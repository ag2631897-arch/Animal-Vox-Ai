import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import ErrorBoundary from './components/ui/ErrorBoundary'
import './styles/globals.css'
import './styles/animations.css'
import './styles/retro.css'
import './styles/screens.css'
import './styles/features.css'
import './styles/app-redesign.css'

const container = document.getElementById('root')

const appElement = (
  <React.StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ErrorBoundary>
  </React.StrictMode>
)

// Support both SSR hydration and client-side rendering
if (container && container.hasChildNodes() && container.innerHTML.trim().length > 0) {
  ReactDOM.hydrateRoot(container, appElement)
} else if (container) {
  ReactDOM.createRoot(container).render(appElement)
}
