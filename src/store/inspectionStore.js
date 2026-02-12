import { create } from 'zustand';

/**
 * Inspection Store - Manages tensor inspection and debugging data
 */
export const useInspectionStore = create((set, get) => ({
    // Selected node/edge for inspection
    selectedNodeId: null,
    selectedEdgeId: null,

    // Inspection data for selected node
    inspectionData: null, // { activations, gradients, weights, statistics }

    // Feature maps (for Conv layers)
    featureMaps: null,

    // Attention weights (for Attention layers)
    attentionWeights: null,

    // Dead neurons
    deadNeurons: [],

    // Gradient flow data (layer-wise)
    gradientFlow: {},

    // Actions
    setSelectedNode: (nodeId) => set({
        selectedNodeId: nodeId,
        selectedEdgeId: null
    }),

    setSelectedEdge: (edgeId) => set({
        selectedEdgeId: edgeId,
        selectedNodeId: null
    }),

    clearSelection: () => set({
        selectedNodeId: null,
        selectedEdgeId: null,
        inspectionData: null
    }),

    setInspectionData: (data) => set({ inspectionData: data }),

    setFeatureMaps: (maps) => set({ featureMaps: maps }),

    setAttentionWeights: (weights) => set({ attentionWeights: weights }),

    setDeadNeurons: (neurons) => set({ deadNeurons: neurons }),

    setGradientFlow: (flow) => set({ gradientFlow: flow }),

    // Update inspection data for a specific node
    updateNodeInspection: (nodeId, data) => {
        const state = get();
        if (state.selectedNodeId === nodeId) {
            set({ inspectionData: data });
        }
    },

    // Clear all inspection data
    clearInspectionData: () => set({
        inspectionData: null,
        featureMaps: null,
        attentionWeights: null,
        deadNeurons: [],
        gradientFlow: {}
    })
}));
