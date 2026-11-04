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
  { image: '/images/animals/dog.png', name: 'Canine Dogs (Canis lupus)', phrase: '“I’m here. Are you coming back?”', tag: 'Contact call', alt: 'Dog bark translator bioacoustic analysis' },
  { image: '/images/animals/cat.png', name: 'Feline Cats (Felis catus)', phrase: '“The room is quiet. I’d like company.”', tag: 'Social signal', alt: 'Cat meow and purr acoustic translation' },
  { image: '/images/animals/bird.png', name: 'Avian Songbirds (Syrinx)', phrase: '“A shadow moved above the trees.”', tag: 'Alarm call', alt: 'Avian bird song and whistle identification' },
  { image: '/images/animals/dolphin.png', name: 'Marine Dolphins (Delphinidae)', phrase: '“Stay close as we move together.”', tag: 'Coordination', alt: 'Dolphin signature whistle translation' },
  { image: '/images/animals/whale.png', name: 'Baleen Whales (Mysticeti)', phrase: '“Deep pulse traveling across the ocean basin.”', tag: 'Long-range contact', alt: 'Whale vocal song deep bioacoustic analysis' },
  { image: '/images/animals/frog.png', name: 'Anuran Frogs (Amphibia)', phrase: '“Joining the rhythmic chorus across the wetland.”', tag: 'Choral mating call', alt: 'Frog anuran chorusing bioacoustic decode' },
]
const features = [
  { title: 'Sound, in context', text: 'Read the behavior around a call, not just a transcript.', icon: AudioLines },
  { title: 'Confidence included', text: 'Every interpretation comes with a signal-strength estimate.', icon: Check },
  { title: 'Behavioral science', text: 'Grounded in decades of observation and vocal research.', icon: BrainCircuit },
  { title: 'Listen in the moment', text: 'Record a call and explore its likely meaning right away.', icon: Radio },
  { title: 'A clearer picture', text: 'Follow patterns across health, emotion, and time.', icon: HeartPulse },
  { title: 'Made for curiosity', text: 'For people who want to notice more, not assume more.', icon: ShieldCheck },
]
const faqs = [
  {
    q: "How does AnimalVox translate animal vocalizations?",
    a: "AnimalVox uses a 90M-parameter BEATs acoustic CNN-Transformer trained on over 700,000 bioacoustic recordings. It extracts spectral tilt, pitch harmonics, formants, and temporal rhythm to classify emotional valence, urgency, and ethological behavioral context, translating them into human language."
  },
  {
    q: "Can AnimalVox distinguish human voices or mock barking from real animals?",
    a: "Yes. AnimalVox incorporates a Tier-0 acoustic anti-spoofing engine analyzing Vocal Tract Length (VTL) physical estimates, formant dispersal (ΔF), and vocal fry subharmonics. If a human imitates a dog bark or bird whistle, AnimalVox filters out the human voice and alerts the user rather than generating a false translation."
  },
  {
    q: "Which animal species are currently supported by AnimalVox AI?",
    a: "AnimalVox supports 6 distinct taxonomic groups: Dogs (Canis lupus familiaris), Cats (Felis catus), Songbirds (Avian Syrinx), Dolphins (Delphinidae), Whales (Mysticeti / Odontoceti), and Frogs (Anuran Chorus)."
  },
  {
    q: "Is AnimalVox AI free to use for pet owners and researchers?",
    a: "Yes, the Quick Translate feature, species auto-detection, and basic bioacoustic analysis tools are completely free to use directly in your browser without requiring a subscription."
  },
  {
    q: "How does AnimalVox protect animal welfare and data privacy?",
    a: "AnimalVox operates with zero audio telemetry by default. Audio processed through the interface uses SHA-256 fingerprinting for high-speed local caching and never shares raw microphone recordings with third parties."
  }
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
  const [openFaq, setOpenFaq] = useState(0)
  useEffect(() => { const update = () => setScrolled(window.scrollY > 50); update(); window.addEventListener('scroll', update); return () => window.removeEventListener('scroll', update) }, [])
  const moveLandingLight = (event) => {
    if (event.pointerType === 'touch') return
    event.currentTarget.style.setProperty('--landing-pointer-x', `${event.clientX}px`)
    event.currentTarget.style.setProperty('--landing-pointer-y', `${event.clientY}px`)
  }
  return <div className="landing-page" onPointerMove={moveLandingLight}>
    <header className={`landing-nav ${scrolled ? 'nav-scrolled' : ''}`}><Link to="/landing" className="brand-lockup"><span className="brand-icon"><PawPrint size={21} /><AudioLines size={12} /></span><span>AnimalVox</span></Link><nav><a href="#how-it-works">How it works</a><a href="#species">Species</a><a href="#faq">FAQ</a><Link to="/app" className="nav-cta">Open App <ArrowRight size={15} /></Link></nav></header>
    <main>
      <section className="hero-section"><div className="hex-grid" /><div className="hero-bracket hero-bracket-tl" /><div className="hero-bracket hero-bracket-tr" /><div className="hero-bracket hero-bracket-bl" /><div className="hero-bracket hero-bracket-br" />
        <div className="hero-content"><motion.span className="eyebrow hero-eyebrow" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>BIOACOUSTICS / MADE HUMAN</motion.span>
          <h1>{['Hear', 'What', "They're", 'Saying'].map((word, index) => <motion.span key={word} initial={{ opacity: 0, y: 34, filter: 'blur(6px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0)' }} transition={{ duration: 0.28, delay: 0.1 + index * 0.075 }}>{word}{index < 3 && ' '}</motion.span>)}</h1>
          <motion.p className="hero-copy" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.28, delay: 0.48 }}>AnimalVox AI translates animal vocalizations into human language, grounded in 80 years of behavioral science.</motion.p>
          <motion.div className="hero-actions" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.28, delay: 0.55 }}><Link to="/app" className="primary-button hero-primary">Start Translating <ArrowRight size={16} /></Link><a href="#demo-card" className="secondary-button" onClick={() => setDemoPlayed(true)}>Watch Demo <ChevronDown size={15} /></a></motion.div>
          <motion.div id="demo-card" className="hero-demo" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.28, delay: 0.62 }}>
            <div className="demo-top">
              <div className="demo-animal">
                <span className="demo-animal-wrap"><img src="/images/animals/dog.png" alt="Canine" className="demo-animal-img" /></span>
                <div><strong>Rio</strong><small>CANINE / SIGNAL 018</small></div>
              </div>
              <span className="demo-live"><i /> TRANSLATED</span>
            </div>
            <WaveformBars active={demoPlayed} count={22} compact />
            <div className="demo-quote">“<WordByWordText text={demoPlayed ? 'Stay away. This is my territory.' : 'Stay away. This is my territory.'} />”</div>
            <div className="demo-card-bottom">
              <span className="confidence-chip"><span />82% confidence</span>
              <span className="behavior-chip">Territorial Warning</span>
              <button aria-label={demoPlayed ? 'Pause demo' : 'Play demo'} onClick={() => setDemoPlayed((value) => !value)}>
                {demoPlayed ? 'Pause' : 'Listen'} <AudioLines size={14} />
              </button>
            </div>
          </motion.div>
        </div><a className="scroll-cue" href="#how-it-works"><ArrowDown size={14} /><span>SCROLL TO EXPLORE</span></a>
      </section>
      <section className="layers-section section-wrap" id="how-it-works"><div className="section-overline"><span>01 / THE METHOD</span><span>FIVE SIGNAL LAYERS</span></div><div className="landing-section-title"><div><span className="eyebrow">FROM SOUND TO SENSE</span><h2>Five Layers of Intelligence</h2></div><p>One vocalization can hold more than a single meaning. Each layer helps keep the interpretation grounded.</p></div><div className="layer-row">{layers.map((layer, index) => { const Icon = layer.icon; return <motion.article className="layer-card" key={layer.title} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.26, delay: index * 0.06 }}><span className="layer-index">0{index + 1}</span><span className="layer-icon"><Icon size={18} /></span><h3>{layer.title}</h3><p>{layer.text}</p>{index < layers.length - 1 && <span className="layer-connector" />}</motion.article> })}</div></section>
      <section className="species-section section-wrap" id="species">
        <div className="section-overline"><span>02 / WHO WE LISTEN TO</span><span>SPECIES LIBRARY</span></div>
        <div className="landing-section-title">
          <div><span className="eyebrow">A SHARED VOCAL WORLD</span><h2>Six Species. Real Science.</h2></div>
          <p>Different voices, different contexts. Explore illustrative interpretations inspired by observed behavior.</p>
        </div>
        <div className="species-grid">
          {species.map((animal, index) => (
            <motion.article className="species-card" key={animal.name} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.26, delay: index * 0.05 }}>
              <div className="species-head">
                <span className="species-img-wrap">
                  <img src={animal.image} alt={animal.alt || animal.name} className="species-img" width="72" height="72" loading="lazy" />
                </span>
                <span className="species-index">0{index + 1}</span>
              </div>
              <h3>{animal.name}</h3>
              <p>{animal.phrase}</p>
              <span className="species-tag">{animal.tag}</span>
            </motion.article>
          ))}
        </div>
        <p className="science-caveat">Example interpretations are illustrative. Animal communication is complex, contextual, and still being studied.</p>
      </section>
      <section className="stats-band"><div className="hex-grid hex-grid-band"/><div className="stats-inner"><div className="stats-intro"><span className="eyebrow">THE PLATFORM, AT A GLANCE</span><p>Designed to make listening more accessible.</p></div><div className="stats-grid"><Stat value="6" label="Species supported" delay={0} /><Stat value="12" label="Behavior classes" delay={50} /><Stat value="700K+" label="Training recordings" delay={100} /><Stat value="$0" label="Monthly cost" delay={150} /></div></div></section>
      <section className="features-section section-wrap"><div className="section-overline"><span>03 / BUILT FOR ATTENTION</span><span>TOOLS FOR BETTER LISTENING</span></div><div className="landing-section-title"><div><span className="eyebrow">SIGNALS WITH CONTEXT</span><h2>Curiosity, with a little more clarity.</h2></div><p>A companion for observing vocal patterns, not a substitute for the science of animal behavior.</p></div><div className="feature-grid">{features.map((feature, index) => { const Icon = feature.icon; return <motion.article className="feature-card" key={feature.title} initial={{ opacity: 0, x: -8 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ duration: 0.24, delay: index * 0.04 }}><Icon size={19} /><div><h3>{feature.title}</h3><p>{feature.text}</p></div></motion.article> })}</div></section>
      <section className="voices-section section-wrap"><div className="section-overline"><span>04 / IN THE FIELD</span><span>THREE WAYS TO LISTEN</span></div><div className="landing-section-title"><div><span className="eyebrow">PEOPLE, PETS, AND PLACE</span><h2>For the ones who notice.</h2></div><p>Different perspectives, one shared instinct: to understand the world a little better.</p></div><div className="testimonial-grid">{people.map((person, index) => <motion.article className="testimonial-card" key={person.name} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.25, delay: index * 0.07 }}><span className="testimonial-mark">“</span><p>{person.quote}</p><div className="person"><span>{person.initial}</span><div><strong>{person.name}</strong><small>{person.role}</small></div></div></motion.article>)}</div></section>

      {/* ─── High Intent SEO FAQ Section ─── */}
      <section className="faq-section section-wrap" id="faq">
        <div className="section-overline"><span>05 / FREQUENTLY ASKED QUESTIONS</span><span>BIOACOUSTIC FAQ</span></div>
        <div className="landing-section-title">
          <div><span className="eyebrow">CLEAR ANSWERS</span><h2>Frequently Asked Questions</h2></div>
          <p>Scientific methodology, anti-spoofing detection, supported species, and privacy practices behind AnimalVox AI.</p>
        </div>
        <div className="faq-accordion">
          {faqs.map((faq, index) => {
            const isOpen = openFaq === index
            return (
              <article key={faq.q} className={`faq-item ${isOpen ? 'faq-item-open' : ''}`}>
                <button
                  type="button"
                  className="faq-question"
                  onClick={() => setOpenFaq(isOpen ? -1 : index)}
                  aria-expanded={isOpen}
                >
                  <span>{faq.q}</span>
                  <span className="faq-icon-wrap">
                    <ChevronDown size={18} />
                  </span>
                </button>
                {isOpen && (
                  <motion.div
                    className="faq-answer"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    transition={{ duration: 0.22 }}
                  >
                    <p>{faq.a}</p>
                  </motion.div>
                )}
              </article>
            )
          })}
        </div>
      </section>

      <section className="footer-cta">
        <div className="footer-cta-content">
          <span className="eyebrow">TAKE A MOMENT TO LISTEN</span>
          <h2>Ready to hear what they're saying?</h2>
          <p>Open the bioacoustic interface and explore animal vocalizations with real-time neural decoding.</p>
          <Link to="/app" className="primary-button">Open AnimalVox <ArrowRight size={16} /></Link>
        </div>
      </section>

      {/* ─── Comprehensive Site Footer ─── */}
      <footer className="site-footer">
        <div className="site-footer-inner">
          <div className="footer-col brand-col">
            <Link to="/landing" className="brand-lockup footer-brand">
              <span className="brand-icon"><PawPrint size={22} /><AudioLines size={12} /></span>
              <span>AnimalVox</span>
            </Link>
            <p className="footer-mission">
              Bioacoustic neural translation grounded in 80 years of peer-reviewed animal behavior research. Translating acoustics into human context with rigorous anti-spoofing and ethological fidelity.
            </p>
            <div className="footer-status-pill">
              <span className="status-dot-pulse" />
              <span>Bioacoustic Core Active (6 Species Online)</span>
            </div>
          </div>

          <div className="footer-col">
            <h4>Supported Species</h4>
            <ul>
              <li><Link to="/app/quick">🐕 Dogs (Canis lupus familiaris)</Link></li>
              <li><Link to="/app/quick">🐈 Cats (Felis catus)</Link></li>
              <li><Link to="/app/quick">🦜 Songbirds (Avian Syrinx)</Link></li>
              <li><Link to="/app/quick">🐬 Dolphins (Delphinidae)</Link></li>
              <li><Link to="/app/quick">🐋 Whales (Mysticeti / Odontoceti)</Link></li>
              <li><Link to="/app/quick">🐸 Frogs (Anuran Chorus)</Link></li>
            </ul>
          </div>

          <div className="footer-col">
            <h4>Intelligence Stack</h4>
            <ul>
              <li><a href="#how-it-works">BEATs CNN-Transformer (90M)</a></li>
              <li><a href="#how-it-works">Tier-0 Anti-Spoofing (VTL)</a></li>
              <li><a href="#how-it-works">Morton's Structural Rules</a></li>
              <li><a href="#how-it-works">SHA-256 Dynamic Cache</a></li>
              <li><a href="#how-it-works">Sliding Rate Limiter & Cookies</a></li>
              <li><Link to="/app/health">Longitudinal Health Monitor</Link></li>
            </ul>
          </div>

          <div className="footer-col">
            <h4>Field Notes & Legal</h4>
            <ul>
              <li><a href="#species">Bioacoustic Ethology</a></li>
              <li><a href="#faq">Bioacoustic FAQ</a></li>
              <li><a href="#how-it-works">Vocal Biomarkers</a></li>
              <li><Link to="/app/settings">Privacy & Cookies Policy</Link></li>
              <li><Link to="/app/settings">Device Storage & Cache</Link></li>
              <li><a href="#demo-card">Bioacoustic Safety Caveat</a></li>
            </ul>
          </div>
        </div>

        <div className="site-footer-bottom">
          <p>© 2026 AnimalVox AI. Grounded in ethological science and bioacoustics. Interpreted with care.</p>
          <div className="footer-badges">
            <span className="footer-chip">HTTP-Only Cookies</span>
            <span className="footer-chip">Dynamic Audio Caching</span>
            <span className="footer-chip">Zero Audio Telemetry</span>
          </div>
        </div>
      </footer>
    </main>
  </div>
}
