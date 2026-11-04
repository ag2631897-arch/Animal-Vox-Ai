import React from 'react'
import ReactDOMServer from 'react-dom/server'
import { StaticRouter } from 'react-router-dom/server.js'
import App from './App'
import ErrorBoundary from './components/ui/ErrorBoundary'

export function render(url) {
  const html = ReactDOMServer.renderToString(
    <React.StrictMode>
      <ErrorBoundary>
        <StaticRouter location={url}>
          <App />
        </StaticRouter>
      </ErrorBoundary>
    </React.StrictMode>
  )
  return { html }
}
