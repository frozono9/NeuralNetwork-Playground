import { create } from 'zustand';

/**
 * Code Store - Manages the Monaco editor state and synchronization
 */
export const useCodeStore = create((set, get) => ({
    // Code content
    code: '',

    // Editor state
    cursorPosition: { line: 1, column: 1 },

    // Sync state
    syncLock: false, // Prevents infinite sync loops
    sourceOfTruth: 'graph', // 'graph' | 'code' - which side initiated the last change
    pendingSync: false,

    // Validation
    syntaxErrors: [],
    isValid: true,

    // Actions
    setCode: (code, source = 'code') => set({
        code,
        sourceOfTruth: source,
        pendingSync: source === 'code' // If code changed, need to sync to graph
    }),

    setCursorPosition: (position) => set({ cursorPosition: position }),

    setSyncLock: (locked) => set({ syncLock: locked }),

    setPendingSync: (pending) => set({ pendingSync: pending }),

    setSyntaxErrors: (errors) => set({
        syntaxErrors: errors,
        isValid: errors.length === 0
    }),

    // Clear pending sync after successful sync
    clearPendingSync: () => set({ pendingSync: false }),

    // Update code from graph (during graph → code sync)
    updateFromGraph: (code) => set({
        code,
        sourceOfTruth: 'graph',
        pendingSync: false
    }),

    // Update code from user editing (triggers code → graph sync)
    updateFromUser: (code) => set({
        code,
        sourceOfTruth: 'code',
        pendingSync: true
    })
}));
