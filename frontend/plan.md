# AnimalVox AI — Frontend Implementation Plan

## Scope and implementation approach

Build a responsive, client-only single-page web app with React 18 and Vite. There will be no backend, database, authentication, or real service/API requests. All animal translations, health metrics, detection results, events, and profiles use the specified mock data. User interactions are implemented locally in the browser; pet/theme/profile preferences may be retained in local browser storage. Recording uses the native browser media APIs when permission is available, with a clear local-demo fallback if unavailable; playback and speech use browser-native capabilities where supported. No server capability is needed.

The app will use the requested React/Vite + Tailwind/custom CSS stack, Framer Motion for transitions and micro-interactions, Lucide React icons, Zustand for global navigation/pets/theme state, and the supplied Instrument Serif, Inter, and JetBrains Mono typography system. No UI component or charting libraries will be introduced. Charts and gauges will be built with CSS and SVG.

### Route and navigation structure

- `/landing` — marketing landing page; `/` resolves to the landing page. The navbar links to How it works, Species and Open App and intentionally has no Docs link; the landing surface uses richer glass, ambient teal/moss lighting, pointer-responsive glow and restrained motion while app dashboards remain calmer.
- `/app` — default app experience (Translate/Record); `/app/:screen` — in-app screen routes for `detect`, `health`, `breed`, `emotions`, `history`, and `settings`.
- Desktop uses a fixed 64px icon rail; mobile replaces it with a bottom tab bar. Sidebar and in-app links navigate between all screens. The pet selector uses the same global pet state across views.
- `public/manus-routes.json` will declare the complete page route set before starting Vite.

## Project structure

- `src/main.jsx`, `src/App.jsx` — app bootstrap and route composition.
- `src/pages/AppPage.jsx` — application route and screen composition.
- `src/components/ui/` — reusable GlassCard, SkeletonCard, RetroDataPanel, FuturisticGauge, SonarRings, WaveformBars, ConfidenceBar, and WordByWordText primitives.
- `src/components/layout/` — AppShell, Sidebar/mobile tab navigation, top bars and pet selector.
- `src/components/screens/` — RecordScreen, DetectionScreen, HealthDashboard, BreedIdentifier, EmotionTimeline, TranslationHistory, and Settings.
- `src/pages/LandingPage.jsx` — complete responsive landing narrative and its method/species/stat/feature/persona/CTA sections.
- `src/store/useAppStore.js` — Zustand state for active pet, pets, theme, voice settings and app actions.
- `src/hooks/` — native audio recording, shared browser speech profiles, word-by-word reveal, and keyed skeleton-state hooks.
- `src/data/mockData.js` — specified pet profiles, translation, detection, wellbeing, history, breed and timeline demo content.
- `src/styles/globals.css`, `animations.css`, `retro.css`, `screens.css` — design tokens, responsive landing/app layouts, motion, screen details and terminal effects.
- `public/manus-routes.json` — route manifest.

## Design direction

### Design movement

A restrained bioacoustic instrument-panel aesthetic: a quiet forest-dark workspace punctuated by teal signal light, with a terminal language reserved for raw evidence and diagnostics.

### Core principles

1. Keep the primary task legible and calm; use generous spacing and restrained UI chrome.
2. Use glass for floating/result surfaces, species/persona cards and the landing navbar, not indiscriminately.
3. Keep technical raw data visibly distinct through terminal green, monospace typography, scanlines and a blinking cursor.
4. Give the landing page a lived-in signal atmosphere through layered teal/moss light, subtle ambient drift and interactive card lighting; give signals and interactions motion with purpose through sonar, staggered reveals and shimmer skeletons.

### Color philosophy

The near-black evergreen base (#070D0A / #0C1610) evokes a quiet listening environment; teal (#00C9A7) signals active interpretation, with moss for supportive states and amber/red reserved for attention or urgency. The terminal palette remains isolated to evidence displays. Theme presets and live color controls update CSS variables without changing the screen's semantic hierarchy.

### Layout paradigm

The landing page is a vertical editorial narrative—hero, method, species, evidence, features, use cases, and CTA. The app is a compact instrument workspace: fixed icon rail or mobile tab bar, clear screen heading, and content arranged around the task rather than a generic admin template. Wide data sections may use responsive columns; the record view remains centered around the listening control.

### Signature elements

- Concentric sonar rings and small waveform traces around the record control.
- Minimal HUD corner marks, a very faint drifting hex pattern and soft moving light pools on the landing page.
- A green-on-black terminal surface reserved for raw model signals and environment status.

### Interaction philosophy

Navigation and form controls should be predictable, keyboard-operable, and immediately responsive. Recording toggles between idle, listening, processing (skeleton), and result; unsupported permissions produce a useful demo state rather than a broken screen. Search, pet/week filters, save/share actions, inline pet edits, theme presets/color inputs, voice choice, speed, and result toggles all respond locally. Motion should clarify state changes and respect reduced-motion preferences.

### Animation

Use short Framer Motion screen transitions, landing-section/card entry staggers, button hover/tap feedback, a 50ms-per-word reveal, and gentle glass-light sweeps/hover highlights. Skeletons shimmer in place and swap to content. Sonar pulse, subtle ambient-light drift and slow hex movement are intentional looping exceptions. Prefer CSS for continuous effects, avoid introducing unnecessary motion, and disable/reduce nonessential motion when `prefers-reduced-motion` is set.

### Typography system

Instrument Serif carries landing headlines and animal translations; Inter handles navigation, labels, forms and UI text; JetBrains Mono is limited to measured values, confidence, history timestamps and terminal output. Keep each screen to a compact hierarchy of no more than three font sizes as requested.

### Brand essence and voice

AnimalVox is a science-grounded listening interface for curious pet owners, birdwatchers and bioacoustic researchers—measured, empathetic and precise. Its voice is warm, direct and evidence-aware. Example lines: “Hear what they're saying.” “A signal is a clue, not the whole story.”

### Wordmark and signature color

Use a custom lockup pairing “AnimalVox” in Instrument Serif with a compact Lucide paw/sound-wave mark; the mark can also stand alone in the app rail. Signature color: signal teal `#00C9A7`.

## Delivery constraints

Keep managed server/database features off. All UI uses supplied sample content, and API-shaped/raw-data views are illustrative mock output only. Preserve dark screens, responsive navigation, skeleton states, accessible labels, and reduced-motion behavior. No decorative custom imagery is required for this internal bioacoustics tool; visuals are created from CSS, SVG, typography and Lucide icons.
