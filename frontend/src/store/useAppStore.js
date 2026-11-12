import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import {
  setApiAuthToken,
  syncUserStateApi,
  savePetApi,
  deletePetApi,
  saveTranslationApi,
  logoutApi,
} from '../lib/api'

export const themePresets = [
  { name: 'Midnight Forest', colors: { bg: '#070D0A', surface: '#101E14', accent: '#00C9A7', accent2: '#4CAF6B', text: '#E8F0E9' } },
  { name: 'Ocean Deep', colors: { bg: '#071014', surface: '#10222A', accent: '#35C6D8', accent2: '#4D9EAA', text: '#E4F5F4' } },
  { name: 'Neon Savanna', colors: { bg: '#100D08', surface: '#241B10', accent: '#F5A623', accent2: '#B9D34B', text: '#F6F0E4' } },
  { name: 'Arctic', colors: { bg: '#090D12', surface: '#141D28', accent: '#76BDF2', accent2: '#6DD5B0', text: '#ECF4FA' } },
]

const defaults = {
  bg: '#070D0A', surface: '#101E14', accent: '#00C9A7', accent2: '#4CAF6B', text: '#E8F0E9',
}

let syncInFlight = null
let syncQueued = false
let syncDebounceTimer = null

export function getUserStorageKey(user) {
  if (!user) return ''
  if (user.id !== undefined && user.id !== null && String(user.id).trim() !== '') {
    return `user_id_${user.id}`
  }
  if (user.email) {
    return `user_email_${String(user.email).trim().toLowerCase()}`
  }
  return ''
}

function createEmptyUserBucket() {
  return {
    pets: [],
    activePetId: '',
    translationHistory: [],
    lastTranslation: null,
    quickSession: null,
    recordSession: null,
    detectSession: null,
    hasCompletedOnboarding: false,
    mustCompletePetProfile: false,
    autoSpeciesDetection: true,
    theme: defaults,
    presetName: 'Midnight Forest',
    voice: 'Nova',
    speed: 1,
    accountProfile: {},
  }
}

function extractBucketFromState(state) {
  return {
    pets: Array.isArray(state.pets) ? state.pets : [],
    activePetId: state.activePetId || '',
    translationHistory: Array.isArray(state.translationHistory) ? state.translationHistory : [],
    lastTranslation: state.lastTranslation || null,
    quickSession: state.quickSession || null,
    recordSession: state.recordSession || null,
    detectSession: state.detectSession || null,
    hasCompletedOnboarding: Boolean(state.hasCompletedOnboarding),
    mustCompletePetProfile: Boolean(state.mustCompletePetProfile),
    autoSpeciesDetection: state.autoSpeciesDetection !== undefined ? Boolean(state.autoSpeciesDetection) : true,
    theme: state.theme || defaults,
    presetName: state.presetName || 'Midnight Forest',
    voice: state.voice || 'Nova',
    speed: typeof state.speed === 'number' ? state.speed : 1,
    accountProfile: state.accountProfile || {},
  }
}

function applyUserPatch(state, patch) {
  const nextState = { ...state, ...patch }
  const userKey = getUserStorageKey(nextState.authUser)
  if (!userKey) return patch

  const currentBucket = state.userBuckets?.[userKey] || extractBucketFromState(state)
  const updatedBucket = {
    ...currentBucket,
    ...extractBucketFromState(nextState),
  }
  return {
    ...patch,
    userBuckets: {
      ...(state.userBuckets || {}),
      [userKey]: updatedBucket,
    },
  }
}

export const useAppStore = create(persist((set, get) => ({
  /* ── Per-User Isolated Buckets ── */
  userBuckets: {},

  /* ── Auth ── */
  isAuthenticated: false,
  authUser: null,
  authToken: null,
  mustCompletePetProfile: false,
  setMustCompletePetProfile: (mustCompletePetProfile) =>
    set((state) => applyUserPatch(state, { mustCompletePetProfile })),

  login: (user, token, isNewUser = false) => {
    setApiAuthToken(token || null)
    const userKey = getUserStorageKey(user)
    const state = get()
    const bucket = (userKey && state.userBuckets?.[userKey])
      ? state.userBuckets[userKey]
      : createEmptyUserBucket()

    const hasPets = Array.isArray(bucket.pets) && bucket.pets.length > 0
    const isNew = Boolean((isNewUser || user?.is_new_user) && !hasPets)

    const nextBucket = {
      ...createEmptyUserBucket(),
      ...bucket,
      mustCompletePetProfile: isNew,
      hasCompletedOnboarding: hasPets ? true : Boolean(bucket.hasCompletedOnboarding),
    }

    set({
      isAuthenticated: true,
      authUser: user,
      authToken: token || null,
      ...nextBucket,
      userBuckets: userKey
        ? { ...(state.userBuckets || {}), [userKey]: nextBucket }
        : (state.userBuckets || {}),
    })

    // Immediately synchronize with NeonDB for this user
    setTimeout(() => {
      get().syncWithServer()
    }, 10)
  },

  logout: () => {
    const state = get()
    const userKey = getUserStorageKey(state.authUser)
    const updatedBuckets = userKey
      ? { ...(state.userBuckets || {}), [userKey]: extractBucketFromState(state) }
      : (state.userBuckets || {})

    setApiAuthToken(null)
    logoutApi()

    set({
      isAuthenticated: false,
      authUser: null,
      authToken: null,
      ...createEmptyUserBucket(),
      userBuckets: updatedBuckets,
    })
  },

  /* ── Server Sync (NeonDB Per-User Persistence) ── */
  syncWithServer: async () => {
    if (syncDebounceTimer) {
      clearTimeout(syncDebounceTimer)
      syncDebounceTimer = null
    }
    if (syncInFlight) {
      syncQueued = true
      return syncInFlight
    }

    syncInFlight = (async () => {
      try {
        const state = get()
        if (!state.isAuthenticated || !state.authUser) return
        const userKey = getUserStorageKey(state.authUser)
        if (!userKey) return

        if (state.authToken) {
          setApiAuthToken(state.authToken)
        }

        const payload = {
          pets: state.pets || [],
          activePetId: state.activePetId || '',
          translationHistory: state.translationHistory || [],
          theme: state.theme || defaults,
          presetName: state.presetName || 'Midnight Forest',
          voice: state.voice || 'Nova',
          speed: state.speed || 1,
          autoSpeciesDetection: state.autoSpeciesDetection,
          hasCompletedOnboarding: state.hasCompletedOnboarding || (state.pets && state.pets.length > 0),
          accountProfile: state.accountProfile || {},
          activeSessions: {
            quickSession: state.quickSession || null,
            recordSession: state.recordSession || null,
            detectSession: state.detectSession || null,
            lastTranslation: state.lastTranslation || null,
          },
        }

        const serverData = await syncUserStateApi(payload)
        const currentAfter = get()
        if (!currentAfter.isAuthenticated || getUserStorageKey(currentAfter.authUser) !== userKey) {
          return
        }

        const serverPets = Array.isArray(serverData?.pets) ? serverData.pets : currentAfter.pets
        const serverHistory = Array.isArray(serverData?.translationHistory)
          ? serverData.translationHistory
          : currentAfter.translationHistory

        // Merge local + server translation history preserving unique IDs
        const seenIds = new Set()
        const mergedHistory = []
        for (const item of [...(currentAfter.translationHistory || []), ...serverHistory]) {
          if (!item?.id || seenIds.has(item.id)) continue
          seenIds.add(item.id)
          mergedHistory.push(item)
        }
        mergedHistory.sort((a, b) => new Date(b.createdAt || 0) - new Date(a.createdAt || 0))

        const serverSettings = serverData?.settings || {}
        const serverSessions = serverSettings.activeSessions || {}

        const resolvedActivePetId =
          (currentAfter.activePetId && serverPets.some((p) => p.id === currentAfter.activePetId))
            ? currentAfter.activePetId
            : (serverSettings.activePetId && serverPets.some((p) => p.id === serverSettings.activePetId))
              ? serverSettings.activePetId
              : (serverPets[0]?.id || '')

        const hasPetsNow = serverPets.length > 0

        set((s) =>
          applyUserPatch(s, {
            pets: serverPets,
            activePetId: resolvedActivePetId,
            translationHistory: mergedHistory.slice(0, 100),
            lastTranslation: s.lastTranslation || serverSessions.lastTranslation || mergedHistory[0] || null,
            quickSession: s.quickSession || serverSessions.quickSession || null,
            recordSession: s.recordSession || serverSessions.recordSession || null,
            detectSession: s.detectSession || serverSessions.detectSession || null,
            hasCompletedOnboarding: hasPetsNow ? true : Boolean(s.hasCompletedOnboarding || serverSettings.hasCompletedOnboarding),
            mustCompletePetProfile: hasPetsNow ? false : s.mustCompletePetProfile,
            accountProfile: Object.keys(s.accountProfile || {}).length > 0
              ? s.accountProfile
              : (serverSettings.accountProfile || {}),
          })
        )
      } catch {
        // Offline or unauthenticated token; local per-user bucket still persists
      } finally {
        syncInFlight = null
        if (syncQueued) {
          syncQueued = false
          syncDebounceTimer = setTimeout(() => get().syncWithServer(), 400)
        }
      }
    })()

    return syncInFlight
  },

  /* ── Onboarding ── */
  hasCompletedOnboarding: false,
  setOnboarded: () => set((state) => applyUserPatch(state, { hasCompletedOnboarding: true })),

  /* ── Auto Species Detection ── */
  autoSpeciesDetection: true,
  toggleAutoSpeciesDetection: () =>
    set((state) => applyUserPatch(state, { autoSpeciesDetection: !state.autoSpeciesDetection })),

  /* ── Screens ── */
  activeScreen: 'record',
  setActiveScreen: (activeScreen) => set({ activeScreen }),

  /* ── Pets (strictly scoped per user) ── */
  pets: [],
  activePetId: '',
  setActivePet: (activePetId) => {
    set((state) => applyUserPatch(state, { activePetId }))
    setTimeout(() => get().syncWithServer(), 50)
  },
  getActivePet: () => get().pets.find((pet) => pet.id === get().activePetId) || get().pets[0],
  savePet: (pet) => {
    const finalPet = {
      ...pet,
      id: pet.id || `${pet.name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
    }
    set((state) => {
      const nextPets = state.pets.some((item) => item.id === finalPet.id)
        ? state.pets.map((item) => (item.id === finalPet.id ? finalPet : item))
        : [...state.pets, finalPet]
      const nextActivePetId = state.activePetId || finalPet.id
      return applyUserPatch(state, {
        pets: nextPets,
        activePetId: nextActivePetId,
        hasCompletedOnboarding: true,
        mustCompletePetProfile: false,
      })
    })
    savePetApi(finalPet).catch(() => {})
    setTimeout(() => get().syncWithServer(), 100)
  },
  deletePet: (id) => {
    set((state) => {
      const nextPets = state.pets.filter((pet) => pet.id !== id)
      const nextActivePetId = state.activePetId === id
        ? (nextPets[0]?.id || '')
        : state.activePetId
      return applyUserPatch(state, {
        pets: nextPets,
        activePetId: nextActivePetId,
      })
    })
    deletePetApi(id).catch(() => {})
  },

  /* ── Theme ── */
  theme: defaults,
  presetName: 'Midnight Forest',
  setThemeColor: (key, value) =>
    set((state) => applyUserPatch(state, { theme: { ...state.theme, [key]: value }, presetName: 'Custom Theme' })),
  setThemePreset: (preset) =>
    set((state) => applyUserPatch(state, { theme: preset.colors, presetName: preset.name })),

  /* ── Voice ── */
  voice: 'Nova',
  speed: 1,
  setVoice: (voice) => set((state) => applyUserPatch(state, { voice })),
  setSpeed: (speed) => set((state) => applyUserPatch(state, { speed })),

  /* ── Account Profile (per user) ── */
  accountProfile: {},
  setAccountProfile: (accountProfile) => {
    set((state) => applyUserPatch(state, { accountProfile }))
    setTimeout(() => get().syncWithServer(), 50)
  },

  /* ── Translation history (per user) ── */
  translationHistory: [],
  lastTranslation: null,
  addTranslation: (entry) => {
    set((state) =>
      applyUserPatch(state, {
        lastTranslation: entry,
        translationHistory: [entry, ...state.translationHistory].slice(0, 100),
      })
    )
    saveTranslationApi(entry).catch(() => {})
  },

  /* ── Screen Active Translation Sessions (persisted across tab switches & refreshes per user) ── */
  quickSession: null,
  setQuickSession: (quickSession) => {
    set((state) => applyUserPatch(state, { quickSession }))
    setTimeout(() => get().syncWithServer(), 100)
  },
  clearQuickSession: () => {
    set((state) => applyUserPatch(state, { quickSession: null }))
    setTimeout(() => get().syncWithServer(), 100)
  },

  recordSession: null,
  setRecordSession: (recordSession) => {
    set((state) => applyUserPatch(state, { recordSession }))
    setTimeout(() => get().syncWithServer(), 100)
  },
  clearRecordSession: () => {
    set((state) => applyUserPatch(state, { recordSession: null }))
    setTimeout(() => get().syncWithServer(), 100)
  },

  detectSession: null,
  setDetectSession: (detectSession) => {
    set((state) => applyUserPatch(state, { detectSession }))
    setTimeout(() => get().syncWithServer(), 100)
  },
  clearDetectSession: () => {
    set((state) => applyUserPatch(state, { detectSession: null }))
    setTimeout(() => get().syncWithServer(), 100)
  },
}), {
  name: 'animalvox-frontend',
  storage: createJSONStorage(() => localStorage),
  partialize: (state) => ({
    userBuckets: state.userBuckets,
    isAuthenticated: state.isAuthenticated,
    authUser: state.authUser,
    authToken: state.authToken,
    mustCompletePetProfile: state.mustCompletePetProfile,
    hasCompletedOnboarding: state.hasCompletedOnboarding,
    autoSpeciesDetection: state.autoSpeciesDetection,
    pets: state.pets,
    activePetId: state.activePetId,
    theme: state.theme,
    presetName: state.presetName,
    voice: state.voice,
    speed: state.speed,
    accountProfile: state.accountProfile,
    translationHistory: state.translationHistory,
    lastTranslation: state.lastTranslation,
    quickSession: state.quickSession,
    recordSession: state.recordSession,
    detectSession: state.detectSession,
  }),
  onRehydrateStorage: () => (state) => {
    if (!state) return
    if (state.authToken) {
      setApiAuthToken(state.authToken)
    }

    if (state.isAuthenticated && state.authUser) {
      const userKey = getUserStorageKey(state.authUser)
      if (userKey) {
        const existingBucket = state.userBuckets?.[userKey]
        if (!existingBucket) {
          // Migrate current logged-in user's top-level state into their dedicated user bucket
          const migrated = extractBucketFromState(state)
          useAppStore.setState({
            ...migrated,
            userBuckets: {
              ...(state.userBuckets || {}),
              [userKey]: migrated,
            },
          })
        } else {
          // Ensure top-level state matches this user's isolated bucket (merging if top-level had newer items)
          const mergedPets = existingBucket.pets?.length ? existingBucket.pets : (state.pets || [])
          const mergedHistory = existingBucket.translationHistory?.length
            ? existingBucket.translationHistory
            : (state.translationHistory || [])
          const restored = {
            ...createEmptyUserBucket(),
            ...existingBucket,
            pets: mergedPets,
            activePetId: existingBucket.activePetId || state.activePetId || mergedPets[0]?.id || '',
            translationHistory: mergedHistory,
            lastTranslation: existingBucket.lastTranslation || state.lastTranslation || null,
            quickSession: existingBucket.quickSession || state.quickSession || null,
            recordSession: existingBucket.recordSession || state.recordSession || null,
            detectSession: existingBucket.detectSession || state.detectSession || null,
          }
          useAppStore.setState({
            ...restored,
            userBuckets: {
              ...(state.userBuckets || {}),
              [userKey]: restored,
            },
          })
        }
      }
    } else {
      // Not authenticated: ensure no previous user's data is exposed at top level
      useAppStore.setState({
        ...createEmptyUserBucket(),
      })
    }
  },
}))
