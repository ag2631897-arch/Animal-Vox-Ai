import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

export const themePresets = [
  { name: 'Midnight Forest', colors: { bg: '#070D0A', surface: '#101E14', accent: '#00C9A7', accent2: '#4CAF6B', text: '#E8F0E9' } },
  { name: 'Ocean Deep', colors: { bg: '#071014', surface: '#10222A', accent: '#35C6D8', accent2: '#4D9EAA', text: '#E4F5F4' } },
  { name: 'Neon Savanna', colors: { bg: '#100D08', surface: '#241B10', accent: '#F5A623', accent2: '#B9D34B', text: '#F6F0E4' } },
  { name: 'Arctic', colors: { bg: '#090D12', surface: '#141D28', accent: '#76BDF2', accent2: '#6DD5B0', text: '#ECF4FA' } },
]

const defaults = {
  bg: '#070D0A', surface: '#101E14', accent: '#00C9A7', accent2: '#4CAF6B', text: '#E8F0E9',
}

export const useAppStore = create(persist((set, get) => ({
  /* ── Auth ── */
  isAuthenticated: false,
  authUser: null,
  login: (user) => set({ isAuthenticated: true, authUser: user }),
  logout: () => set({ isAuthenticated: false, authUser: null }),

  /* ── Onboarding ── */
  hasCompletedOnboarding: false,
  setOnboarded: () => set({ hasCompletedOnboarding: true }),

  /* ── Auto Species Detection ── */
  autoSpeciesDetection: true,
  toggleAutoSpeciesDetection: () => set((state) => ({ autoSpeciesDetection: !state.autoSpeciesDetection })),

  /* ── Screens ── */
  activeScreen: 'record',
  setActiveScreen: (activeScreen) => set({ activeScreen }),

  /* ── Pets (empty by default — first-time popup creates the first one) ── */
  pets: [],
  activePetId: '',
  setActivePet: (activePetId) => set({ activePetId }),
  getActivePet: () => get().pets.find((pet) => pet.id === get().activePetId) || get().pets[0],
  savePet: (pet) => set((state) => ({
    pets: state.pets.some((item) => item.id === pet.id)
      ? state.pets.map((item) => (item.id === pet.id ? pet : item))
      : [...state.pets, { ...pet, id: pet.id || `${pet.name.toLowerCase()}-${Date.now()}` }],
  })),
  deletePet: (id) => set((state) => ({
    pets: state.pets.filter((pet) => pet.id !== id),
    activePetId: state.activePetId === id
      ? (state.pets.find((pet) => pet.id !== id)?.id || '')
      : state.activePetId,
  })),

  /* ── Theme ── */
  theme: defaults,
  presetName: 'Midnight Forest',
  setThemeColor: (key, value) => set((state) => ({ theme: { ...state.theme, [key]: value }, presetName: 'Custom Theme' })),
  setThemePreset: (preset) => set({ theme: preset.colors, presetName: preset.name }),

  /* ── Voice ── */
  voice: 'Nova',
  speed: 1,
  setVoice: (voice) => set({ voice }),
  setSpeed: (speed) => set({ speed }),

  /* ── Translation history ── */
  translationHistory: [],
  lastTranslation: null,
  addTranslation: (entry) => set((state) => ({
    lastTranslation: entry,
    translationHistory: [entry, ...state.translationHistory].slice(0, 100),
  })),
}), {
  name: 'animalvox-frontend',
  storage: createJSONStorage(() => localStorage),
  partialize: (state) => ({
    isAuthenticated: state.isAuthenticated,
    authUser: state.authUser,
    hasCompletedOnboarding: state.hasCompletedOnboarding,
    autoSpeciesDetection: state.autoSpeciesDetection,
    pets: state.pets,
    activePetId: state.activePetId,
    theme: state.theme,
    presetName: state.presetName,
    voice: state.voice,
    speed: state.speed,
    translationHistory: state.translationHistory,
    lastTranslation: state.lastTranslation,
  }),
}))
