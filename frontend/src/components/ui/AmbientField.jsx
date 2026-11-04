import { useEffect, useRef } from 'react'

function hexToRgb(value) {
  const match = /^#([\da-f]{6})$/i.exec(value?.trim() || '')
  if (!match) return [0, 201, 167]
  const parsed = Number.parseInt(match[1], 16)
  return [(parsed >> 16) & 255, (parsed >> 8) & 255, parsed & 255]
}

export default function AmbientField({ intensity = 'full' }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const context = canvas.getContext('2d', { alpha: true })
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
    const coarsePointer = window.matchMedia('(pointer: coarse)')
    const state = {
      frame: 0,
      lastFrame: 0,
      width: 0,
      height: 0,
      dpr: 1,
      pointerX: window.innerWidth / 2,
      pointerY: window.innerHeight / 2,
      lightX: window.innerWidth / 2,
      lightY: window.innerHeight / 2,
      idle: true,
      idleTimer: 0,
      hidden: document.hidden,
    }

    const resize = () => {
      state.dpr = Math.min(window.devicePixelRatio || 1, 2)
      state.width = window.innerWidth
      state.height = window.innerHeight
      canvas.width = Math.round(state.width * state.dpr)
      canvas.height = Math.round(state.height * state.dpr)
      canvas.style.width = `${state.width}px`
      canvas.style.height = `${state.height}px`
      context.setTransform(state.dpr, 0, 0, state.dpr, 0, 0)
    }

    const handlePointer = (event) => {
      state.pointerX = event.clientX
      state.pointerY = event.clientY
      state.idle = false
      window.clearTimeout(state.idleTimer)
      state.idleTimer = window.setTimeout(() => { state.idle = true }, 280)
    }
    const visibility = () => {
      state.hidden = document.hidden
      if (state.hidden) cancelAnimationFrame(state.frame)
      else if (!reducedMotion.matches) state.frame = requestAnimationFrame(draw)
    }
    const draw = (now = 0) => {
      if (state.hidden) return
      const minFrameTime = state.idle ? 1000 / 45 : 1000 / 60
      if (now - state.lastFrame >= minFrameTime || reducedMotion.matches) {
        state.lastFrame = now
        const styles = getComputedStyle(canvas)
        const [r, g, b] = hexToRgb(styles.getPropertyValue('--accent'))
        const amplitudeMultiplier = intensity === 'low' ? 0.4 : intensity === 'minimal' ? 0 : 1
        const lineCount = coarsePointer.matches ? 8 : 16
        const time = now * 0.00035
        const targetX = coarsePointer.matches ? state.width * 0.5 + Math.sin(time * 0.4) * state.width * 0.12 : state.pointerX
        const targetY = coarsePointer.matches ? state.height * 0.45 + Math.cos(time * 0.35) * state.height * 0.08 : state.pointerY
        state.lightX += (targetX - state.lightX) * 0.08
        state.lightY += (targetY - state.lightY) * 0.08
        context.clearRect(0, 0, state.width, state.height)
        const light = context.createRadialGradient(state.lightX, state.lightY, 0, state.lightX, state.lightY, 360)
        light.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${intensity === 'minimal' ? 0.06 : 0.1})`)
        light.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`)
        context.globalCompositeOperation = 'screen'
        context.fillStyle = light
        context.fillRect(0, 0, state.width, state.height)
        if (amplitudeMultiplier > 0 && !reducedMotion.matches) {
          context.globalCompositeOperation = 'source-over'
          context.lineWidth = 1
          context.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.075)`
          for (let line = 0; line < lineCount; line += 1) {
            const y = ((line + 1) / (lineCount + 1)) * state.height
            const phase = time * (1 + line * 0.03) + line * 0.68
            context.beginPath()
            for (let x = 0; x <= state.width; x += 24) {
              const distanceX = x - state.pointerX
              const distanceY = y - state.pointerY
              const pull = Math.exp(-((distanceX * distanceX) + (distanceY * distanceY)) / (2 * 220 * 220))
              const wave = Math.sin((x * 0.008) + phase) * (5 + pull * 15) * amplitudeMultiplier
              if (x === 0) context.moveTo(x, y + wave)
              else context.lineTo(x, y + wave)
            }
            context.stroke()
          }
        }
      }
      if (!reducedMotion.matches) state.frame = requestAnimationFrame(draw)
    }

    resize()
    window.addEventListener('resize', resize)
    window.addEventListener('pointermove', handlePointer, { passive: true })
    document.addEventListener('visibilitychange', visibility)
    state.frame = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(state.frame)
      window.clearTimeout(state.idleTimer)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointermove', handlePointer)
      document.removeEventListener('visibilitychange', visibility)
    }
  }, [intensity])

  return <canvas ref={canvasRef} className="ambient-field" aria-hidden="true" />
}
