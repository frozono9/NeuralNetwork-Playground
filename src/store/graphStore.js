import { create } from 'zustand';

/**
 * Graph Store - Manages the visual node graph state
 */
export const useGraphStore = create((set, get) => ({
    // Nodes and edges
    nodes: [],
    edges: [],

    // Selection
    selectedNodes: [],
    selectedEdges: [],

    // Actions
    setNodes: (nodes) => set({ nodes }),
    setEdges: (edges) => set({ edges }),

    addNode: (node) => set((state) => ({
        nodes: [...state.nodes, node]
    })),

    updateNode: (id, data) => set((state) => ({
        nodes: state.nodes.map((node) =>
            node.id === id ? { ...node, data: { ...node.data, ...data } } : node
        )
    })),

    removeNode: (id) => set((state) => ({
        nodes: state.nodes.filter((node) => node.id !== id),
        edges: state.edges.filter((edge) => edge.source !== id && edge.target !== id)
    })),

    addEdge: (edge) => set((state) => ({
        edges: [...state.edges, edge]
    })),

    removeEdge: (id) => set((state) => ({
        edges: state.edges.filter((edge) => edge.id !== id)
    })),

    setSelectedNodes: (nodeIds) => set({ selectedNodes: nodeIds }),
    setSelectedEdges: (edgeIds) => set({ selectedEdges: edgeIds }),

    // Clear all
    clearGraph: () => set({
        nodes: [],
        edges: [],
        selectedNodes: [],
        selectedEdges: []
    }),

    // Get node by id
    getNode: (id) => {
        const state = get();
        return state.nodes.find((node) => node.id === id);
    },

    // Get connected nodes
    getConnectedNodes: (nodeId) => {
        const state = get();
        const connectedEdges = state.edges.filter(
            (edge) => edge.source === nodeId || edge.target === nodeId
        );
        const connectedNodeIds = new Set();
        connectedEdges.forEach((edge) => {
            connectedNodeIds.add(edge.source);
            connectedNodeIds.add(edge.target);
        });
        connectedNodeIds.delete(nodeId);
        return Array.from(connectedNodeIds);
    }
}));
