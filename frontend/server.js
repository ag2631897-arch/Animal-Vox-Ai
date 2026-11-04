import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer as createViteServer } from 'vite'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const isProd = process.env.NODE_ENV === 'production'
const PORT = Number(process.env.PORT) || 3000
const API_BACKEND = process.env.API_BACKEND || 'http://127.0.0.1:8000'

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.webm': 'audio/webm',
  '.xml': 'application/xml; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
}

async function createServer() {
  let vite
  if (!isProd) {
    vite = await createViteServer({
      server: { middlewareMode: true, hmr: { port: 3001 } },
      appType: 'custom',
    })
  }

  const server = http.createServer(async (req, res) => {
    const url = req.url || '/'

    // 1. Proxy API endpoints to the FastAPI backend
    if (url.startsWith('/api/')) {
      const targetUrl = new URL(url, API_BACKEND)
      const proxyReq = http.request(
        targetUrl,
        {
          method: req.method,
          headers: {
            ...req.headers,
            host: targetUrl.host,
          },
        },
        (proxyRes) => {
          res.writeHead(proxyRes.statusCode || 200, proxyRes.headers)
          proxyRes.pipe(res)
        }
      )

      proxyReq.on('error', (err) => {
        console.warn('[SSR Proxy Warning]:', err.message)
        res.writeHead(502, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'backend_unavailable', message: 'FastAPI backend connection error' }))
      })

      req.pipe(proxyReq)
      return
    }

    // 1b. Technical SEO Static Handlers (robots.txt and sitemap.xml)
    if (url === '/robots.txt') {
      const robotsPath = path.resolve(__dirname, isProd ? 'dist/client/robots.txt' : 'public/robots.txt')
      const targetFile = fs.existsSync(robotsPath) ? robotsPath : path.resolve(__dirname, 'public/robots.txt')
      if (fs.existsSync(targetFile)) {
        res.writeHead(200, {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'public, max-age=86400',
        })
        fs.createReadStream(targetFile).pipe(res)
        return
      }
    }

    if (url === '/sitemap.xml') {
      const sitemapPath = path.resolve(__dirname, isProd ? 'dist/client/sitemap.xml' : 'public/sitemap.xml')
      const targetFile = fs.existsSync(sitemapPath) ? sitemapPath : path.resolve(__dirname, 'public/sitemap.xml')
      if (fs.existsSync(targetFile)) {
        res.writeHead(200, {
          'Content-Type': 'application/xml; charset=utf-8',
          'Cache-Control': 'public, max-age=86400',
        })
        fs.createReadStream(targetFile).pipe(res)
        return
      }
    }

    // 2. In dev mode, let Vite middleware serve assets; in prod, serve from dist/client
    if (isProd) {
      const cleanUrl = url.split('?')[0]
      const staticCandidate = path.join(__dirname, 'dist/client', cleanUrl)
      if (fs.existsSync(staticCandidate) && fs.statSync(staticCandidate).isFile()) {
        const ext = path.extname(cleanUrl).toLowerCase()
        const mime = MIME_TYPES[ext] || 'application/octet-stream'
        res.writeHead(200, {
          'Content-Type': mime,
          'Cache-Control': ext === '.html' ? 'no-cache' : 'public, max-age=31536000, immutable',
        })
        fs.createReadStream(staticCandidate).pipe(res)
        return
      }
    } else if (!req.headers.accept?.includes('text/html') && url !== '/' && !url.startsWith('/app') && !url.startsWith('/landing') && !url.startsWith('/login')) {
      vite.middlewares(req, res, () => {
        // Fallback
        res.writeHead(404)
        res.end()
      })
      return
    }

    // 3. Server-Side Rendering (SSR) for HTML routes
    try {
      let template
      let render

      if (!isProd) {
        template = fs.readFileSync(path.resolve(__dirname, 'index.html'), 'utf-8')
        template = await vite.transformIndexHtml(url, template)
        const entry = await vite.ssrLoadModule('/src/entry-server.jsx')
        render = entry.render
      } else {
        template = fs.readFileSync(path.resolve(__dirname, 'dist/client/index.html'), 'utf-8')
        const entry = await import('./dist/server/entry-server.js')
        render = entry.render
      }

      let appHtml = ''
      try {
        const rendered = render(url)
        appHtml = rendered.html || ''
      } catch (renderErr) {
        // Safe isolation: fall back gracefully to client-side hydration without exposing internal stack traces
        console.warn('[SSR Render Isolation]: Hydrating on client side.', renderErr.message)
      }

      const html = template.replace('<!--ssr-outlet-->', appHtml)

      res.writeHead(200, {
        'Content-Type': 'text/html; charset=utf-8',
        'X-Powered-By': 'AnimalVox-SSR-Engine/1.1',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
      })
      res.end(html)
    } catch (e) {
      if (!isProd && vite) {
        vite.ssrFixStacktrace(e)
      }
      console.warn('[SSR Server Error]:', e.message)
      res.writeHead(500, {
        'Content-Type': 'text/html; charset=utf-8',
        'X-Content-Type-Options': 'nosniff',
      })
      res.end('<!doctype html><html><body><div id="root"></div><script type="module" src="/src/main.jsx"></script></body></html>')
    }
  })

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`[AnimalVox SSR Server] Running at http://localhost:${PORT}`)
  })

  const shutdown = (signal) => {
    console.log(`[AnimalVox SSR Server] ${signal} received: closing HTTP server...`)
    server.close(() => {
      console.log('[AnimalVox SSR Server] HTTP server closed cleanly.')
      process.exit(0)
    })
    setTimeout(() => {
      console.error('[AnimalVox SSR Server] Forced shutdown after timeout.')
      process.exit(1)
    }, 10000).unref()
  }

  process.on('SIGTERM', () => shutdown('SIGTERM'))
  process.on('SIGINT', () => shutdown('SIGINT'))
}

createServer()
