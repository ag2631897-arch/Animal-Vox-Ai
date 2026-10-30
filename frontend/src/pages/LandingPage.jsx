import { useEffect, useRef, useState } from 'react'
import { motion, useInView } from 'framer-motion'
import { ArrowDown, ArrowRight, AudioLines, Bird, BrainCircuit, Check, ChevronDown, Dna, HeartPulse, PawPrint, Radio, ShieldCheck, Sparkles, Waves } from 'lucide-react'
import { Link } from 'react-router-dom'
import WaveformBars from '../components/ui/WaveformBars'
import WordByWordText from '../components/ui/WordByWordText'

const layers = [
  { title: 'Preprocess', text: 'Separate the vocal signal from wind, room tone, and the rest of the world.', icon: Waves },
  { title: 'Extract', text: 'Measure pitch, rhythm, texture, and the details ears can miss.', icon: AudioLines },
  { title: 'Classify', text: 'Match acoustic patterns to species and behavioral signatures.', icon: Bird },
  { title: 'Reason', text: 'Place a sound in context without flattening it into a simple label.', icon: BrainCircuit },
  { title: 'Translate', text: 'Offer a human-readable interpretation, with confidence attached.', icon: Sparkles },
]
const species = [
  { emoji: '🐕', name: 'Dogs', phrase: '“I’m here. Are you coming back?”', tag: 'Contact call' },
  { emoji: '🐈', name: 'Cats', phrase: '“The room is quiet. I’d like company.”', tag: 'Social signal' },
  { emoji: '🐦', name: 'Songbirds', phrase: '“A shadow moved above the trees.”', tag: 'Alarm call' },
  { emoji: '🐬', name: 'Dolphins', phrase: '“Stay close as we move together.”', tag: 'Coordination' },
  { emoji: '🐘', name: 'Elephants', phrase: '“I’m calling to the herd nearby.”', tag: 'Long-range contact' },
  { emoji: '🐺', name: 'Wolves', phrase: '“Our group is gathering over here.”', tag: 'Pack call' },
]
const features = [
  { title: 'Sound, in context', text: 'Read the behavior around a call, not just a transcript.', icon: AudioLines },
  { title: 'Confidence included', text: 'Every interpretation comes with a signal-strength estimate.', icon: Check },
  { title: 'Behavioral science', text: 'Grounded in decades of observation and vocal research.', icon: BrainCircuit },
  { title: 'Listen in the moment', text: 'Record a call and explore its likely meaning right away.', icon: Radio },
  { title: 'A clearer picture', text: 'Follow patterns across health, emotion, and time.', icon: HeartPulse },
  { title: 'Made for curiosity', text: 'For people who want to notice more, not assume more.', icon: ShieldCheck },
]
const people = [
  { initial: 'B', name: 'Bea M.', role: 'Birdwatcher', quote: 'I started noticing the rhythm of calls around my trail. It feels like paying attention in a new way.' },
  { initial: 'A', name: 'Aryan K.', role: 'Pet owner', quote: 'Rio’s sounds tell me when he wants to play or when something has caught his attention.' },
  { initial: 'M', name: 'Dr. Mara L.', role: 'Marine researcher', quote: 'A helpful way to explore vocal patterns while remembering there is always more to learn.' },
]

function Stat({ value, label, delay }) {
  const [count, setCount] = useState('0')
  const statRef = useRef(null)
  const inView = useInView(statRef, { once: true, margin: '-40px' })
  useEffect(() => {
    if (!inView) return undefined
    const numeric = Number.parseInt(value.replace(/\D/g, ''), 10)
    const prefix = value.match(/^[^\d]*/)?.[0] || ''
    const suffix = value.replace(/^[^\d]*/, '').replace(/^\d+/, '')
    if (!Number.isFinite(numeric)) { setCount(value); return undefined }
    const start = performance.now()
    let frame
    const tick = (now) => { const progress = Math.min(1, (now - start) / 680); const current = Math.round(numeric * (1 - (1 - progress) ** 3)); setCount(`${prefix}${current}${suffix}`); if (progress < 1) frame = requestAnimationFrame(tick) }
    const timer = window.setTimeout(() => { frame = requestAnimationFrame(tick) }, delay)
    return () => { clearTimeout(timer); cancelAnimationFrame(frame) }
  }, [inView, value, delay])
  return <div className="stat-item" ref={statRef}><strong>{count}</strong><span>{label}</span></div>
}

export default function LandingPage() {
  const [scrolled, setScrolled] = useState(false)
  const [demoPlayed, setDemoPlayed] = useState(false)
  useEffect(() => { const update = () => setScrolled(window.scrollY > 50); update(); window.addEventListener('scroll', update); return () => window.removeEventListener('scroll', update) }, [])
  const moveLandingLight = (event) => {
    if (event.pointerType === 'touch') return
    event.currentTarget.style.setProperty('--landing-pointer-x', `${event.clientX}px`)
    event.currentTarget.style.setProperty('--landing-pointer-y', `${event.clientY}px`)
  }
  return <div className="landing-page" onPointerMove={moveLandingLight}>
    <header className={`landing-nav ${scrolled ? 'nav-scrolled' : ''}`}><Link to="/landing" className="brand-lockup"><span className="brand-icon"><PawPrint size={21} /><AudioLines size={12} /></span><span>AnimalVox</span></Link><nav><a href="#how-it-works">How it works</a><a href="#species">Species</a><Link to="/app" className="nav-cta">Open App <ArrowRight size={15} /></Link></nav></header>
    <main>
      <section className="hero-section"><div className="hex-grid" /><div className="hero-bracket hero-bracket-tl" /><div className="hero-bracket hero-bracket-tr" /><div className="hero-bracket hero-bracket-bl" /><div className="hero-bracket hero-bracket-br" />
        <div className="hero-content"><motion.span className="eyebrow hero-eyebrow" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>BIOACOUSTICS / MADE HUMAN</motion.span>
          <h1>{['Hear', 'What', "They're", 'Saying'].map((word, index) => <motion.span key={word} initial={{ opacity: 0, y: 34, filter: 'blur(6px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0)' }} transition={{ duration: 0.28, delay: 0.1 + index * 0.075 }}>{word}{index < 3 && ' '}</motion.span>)}</h1>
          <motion.p className="hero-copy" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.28, delay: 0.48 }}>AnimalVox AI translates animal vocalizations into human language, grounded in 80 years of behavioral science.</motion.p>
          <motion.div className="hero-actions" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.28, delay: 0.55 }}><Link to="/app" className="primary-button hero-primary">Start Translating <ArrowRight size={16} /></Link><a href="#demo-card" className="secondary-button" onClick={() => setDemoPlayed(true)}>Watch Demo <ChevronDown size={15} /></a></motion.div>
          <motion.div id="demo-card" className="hero-demo" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.28, delay: 0.62 }}><div className="demo-top"><div className="demo-animal"><span>🐕</span><div><strong>Rio</strong><small>CANINE / SIGNAL 018</small></div></div><span className="demo-live"><i /> TRANSLATED</span></div><WaveformBars active={demoPlayed} count={22} compact /><div className="demo-quote">“<WordByWordText text={demoPlayed ? 'Stay away. This is my territory.' : 'Stay away. This is my territory.'} />”</div><div className="demo-card-bottom"><span className="confidence-chip"><span />82% confidence</span><span className="behavior-chip">Territorial Warning</span><button aria-label={demoPlayed ? 'Pause demo' : 'Play demo'} onClick={() => setDemoPlayed((value) => !value)}>{demoPlayed ? 'Pause' : 'Listen'} <AudioLines size={14} /></button></div></motion.div>
        </div><a className="scroll-cue" href="#how-it-works"><ArrowDown size={14} /><span>SCROLL TO EXPLORE</span></a>
      </section>
      <section className="layers-section section-wrap" id="how-it-works"><div className="section-overline"><span>01 / THE METHOD</span><span>FIVE SIGNAL LAYERS</span></div><div className="landing-section-title"><div><span className="eyebrow">FROM SOUND TO SENSE</span><h2>Five Layers of Intelligence</h2></div><p>One vocalization can hold more than a single meaning. Each layer helps keep the interpretation grounded.</p></div><div className="layer-row">{layers.map((layer, index) => { const Icon = layer.icon; return <motion.article className="layer-card" key={layer.title} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.26, delay: index * 0.06 }}><span className="layer-index">0{index + 1}</span><span className="layer-icon"><Icon size={18} /></span><h3>{layer.title}</h3><p>{layer.text}</p>{index < layers.length - 1 && <span className="layer-connector" />}</motion.article> })}</div></section>
      <section className="species-section section-wrap" id="species"><div className="section-overline"><span>02 / WHO WE LISTEN TO</span><span>SPECIES LIBRARY</span></div><div className="landing-section-title"><div><span className="eyebrow">A SHARED VOCAL WORLD</span><h2>Six Species. Real Science.</h2></div><p>Different voices, different contexts. Explore illustrative interpretations inspired by observed behavior.</p></div><div className="species-grid">{species.map((animal, index) => <motion.article className="species-card" key={animal.name} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.26, delay: index * 0.05 }}><div className="species-head"><span className="species-emoji">{animal.emoji}</span><span className="species-index">0{index + 1}</span></div><h3>{animal.name}</h3><p>{animal.phrase}</p><span className="species-tag">{animal.tag}</span></motion.article>)}</div><p className="science-caveat">Example interpretations are illustrative. Animal communication is complex, contextual, and still being studied.</p></section>
      <section className="stats-band"><div className="hex-grid hex-grid-band"/><div className="stats-inner"><div className="stats-intro"><span className="eyebrow">THE PLATFORM, AT A GLANCE</span><p>Designed to make listening more accessible.</p></div><div className="stats-grid"><Stat value="6" label="Species supported" delay={0} /><Stat value="12" label="Behavior classes" delay={50} /><Stat value="700K+" label="Training recordings" delay={100} /><Stat value="$0" label="Monthly cost" delay={150} /></div></div></section>
      <section className="features-section section-wrap"><div className="section-overline"><span>03 / BUILT FOR ATTENTION</span><span>TOOLS FOR BETTER LISTENING</span></div><div className="landing-section-title"><div><span className="eyebrow">SIGNALS WITH CONTEXT</span><h2>Curiosity, with a little more clarity.</h2></div><p>A companion for observing vocal patterns, not a substitute for the science of animal behavior.</p></div><div className="feature-grid">{features.map((feature, index) => { const Icon = feature.icon; return <motion.article className="feature-card" key={feature.title} initial={{ opacity: 0, x: -8 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ duration: 0.24, delay: index * 0.04 }}><Icon size={19} /><div><h3>{feature.title}</h3><p>{feature.text}</p></div></motion.article> })}</div></section>
      <section className="voices-section section-wrap"><div className="section-overline"><span>04 / IN THE FIELD</span><span>THREE WAYS TO LISTEN</span></div><div className="landing-section-title"><div><span className="eyebrow">PEOPLE, PETS, AND PLACE</span><h2>For the ones who notice.</h2></div><p>Different perspectives, one shared instinct: to understand the world a little better.</p></div><div className="testimonial-grid">{people.map((person, index) => <motion.article className="testimonial-card" key={person.name} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.25, delay: index * 0.07 }}><span className="testimonial-mark">“</span><p>{person.quote}</p><div className="person"><span>{person.initial}</span><div><strong>{person.name}</strong><small>{person.role}</small></div></div></motion.article>)}</div></section>
      <section className="footer-cta"><div className="footer-cta-content"><span className="eyebrow">TAKE A MOMENT TO LISTEN</span><h2>Ready to hear what they're saying?</h2><p>Open the demo and explore the signals for yourself.</p><Link to="/app" className="primary-button">Open AnimalVox <ArrowRight size={16} /></Link></div><footer><Link to="/landing" className="brand-lockup footer-brand"><span className="brand-icon"><PawPrint size={20} /><AudioLines size={11} /></span><span>AnimalVox</span></Link><p>Listen closely. Interpret carefully.</p><nav><a href="#how-it-works">Method</a><a href="#species">Species</a><Link to="/app/settings">Settings</Link></nav><small>© 2026 AnimalVox AI · Illustrative demo</small></footer></section>
    </main>
  </div>
}
