export const initialPets = [
  { id: 'rio', name: 'Rio', species: 'Dog', breed: 'German Shepherd', age: '3 years', gender: 'Male', emoji: '🐕' },
  { id: 'luna', name: 'Luna', species: 'Cat', breed: 'Siamese', age: '2 years', gender: 'Female', emoji: '🐈' },
]

export const translation = {
  pet: 'Rio', ownerDirected: true, ownerName: 'Aryan',
  text: 'Aryan, come here — I need you right now. Where did you go?',
  behavior: 'B02 — Territorial Warning', confidence: 0.82,
  intensity: 0.78, urgency: 0.45, species: 'German Shepherd', snrDb: 18.3,
  quality: 'good', processingMs: 2341,
}

export const detections = [
  { emoji: '🐦', species: 'American Robin', confidence: 0.94, color: '#00C9A7', translation: "Freeze! There's a hawk directly overhead. Don't move." },
  { emoji: '🐦', species: 'House Sparrow', confidence: 0.87, color: '#4CAF6B', translation: "I'm over here — where is everyone? Come find me." },
  { emoji: '🐱', species: 'Cat', confidence: 0.89, color: '#A78BFA', translation: 'I can see the birds and I cannot get to them. Torture.' },
]

export const wellbeing = {
  overall: 74,
  dimensions: [
    { name: 'Physical Distress', value: 12, trend: 'down', status: 'Low' },
    { name: 'Environmental', value: 34, trend: 'steady', status: 'Moderate' },
    { name: 'Isolation', value: 18, trend: 'down', status: 'Low' },
    { name: 'Contentment', value: 71, trend: 'up', status: 'Good' },
  ],
  healthFlags: [],
  weeklyActivity: [18, 24, 12, 31, 22, 28, 16],
}

export const historyItems = [
  { id: 1, day: 'Today', pet: 'Rio', emoji: '🐕', time: '3:45 PM', behavior: 'Territorial Warning', text: 'Aryan, come here — I need you right now. Where did you go?', confidence: 0.82 },
  { id: 2, day: 'Today', pet: 'Rio', emoji: '🐕', time: '1:18 PM', behavior: 'Contact Call', text: 'I hear you in the garden. Are you coming back inside?', confidence: 0.91 },
  { id: 3, day: 'Today', pet: 'Luna', emoji: '🐈', time: '11:06 AM', behavior: 'Play Invitation', text: 'There is a ribbon nearby and I am ready to pounce.', confidence: 0.88 },
  { id: 4, day: 'Yesterday', pet: 'Rio', emoji: '🐕', time: '8:32 PM', behavior: 'Attention Seeking', text: 'Something moved near the back door. Come listen with me.', confidence: 0.79 },
  { id: 5, day: 'Yesterday', pet: 'Luna', emoji: '🐈', time: '6:10 PM', behavior: 'Contentment', text: 'This warm spot is exactly where I want to be.', confidence: 0.93 },
  { id: 6, day: 'Yesterday', pet: 'Rio', emoji: '🐕', time: '9:12 AM', behavior: 'Greeting', text: 'You are home. I have been waiting to see you.', confidence: 0.86 },
]

export const emotionDistribution = [
  { name: 'Contentment', value: 42, color: '#00C9A7' },
  { name: 'Excitement', value: 26, color: '#4CAF6B' },
  { name: 'Hunger', value: 14, color: '#F5A623' },
  { name: 'Stress', value: 8, color: '#EF4444' },
  { name: 'Other', value: 10, color: '#3D5C42' },
]

export const emotionDays = [
  { day: 'Monday', moods: ['😊', '😌', '🐾', '😋', '😊', '🎾', '😌', '😊'] },
  { day: 'Tuesday', moods: ['😌', '🐕', '🎾', '😊', '😋', '😊', '🐾', '😌'] },
  { day: 'Wednesday', moods: ['😊', '😌', '😟', '🐾', '😊', '😋', '🎾', '😌'] },
  { day: 'Thursday', moods: ['🐾', '😊', '😌', '🎾', '😊', '😋', '😌', '😊'] },
  { day: 'Friday', moods: ['😊', '🎾', '😌', '😊', '🐾', '😋', '😊', '😌'] },
  { day: 'Saturday', moods: ['😌', '😊', '🎾', '🐾', '😊', '😋', '😊', '😌'] },
  { day: 'Sunday', moods: ['😊', '😌', '🐾', '😊', '😋', '🎾', '😊', '😌'] },
]

export const notableEvents = [
  { time: 'Today · 2:14 PM', title: 'Play invitation', description: 'A bright, repeated call followed a familiar voice nearby.', color: '#4CAF6B' },
  { time: 'Yesterday · 8:31 PM', title: 'Environmental alert', description: 'Rio noticed a sudden sound near the back door.', color: '#F5A623' },
  { time: 'Yesterday · 6:08 PM', title: 'Settled contentment', description: 'Long, low vocalizations while resting with Luna nearby.', color: '#00C9A7' },
]

export const breeds = [
  { name: 'German Shepherd', confidence: 82, clues: 'Deep resonant bark, measured breath, and a strong mid-range harmonic profile.' },
  { name: 'Belgian Malinois', confidence: 11, clues: 'Similar working-breed resonance and alert cadence.' },
  { name: 'Dutch Shepherd', confidence: 7, clues: 'Overlapping pitch range with a slightly lighter bark profile.' },
]
