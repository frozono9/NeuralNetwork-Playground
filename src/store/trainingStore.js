import { create } from 'zustand';

/**
 * Training Store - Manages training state and metrics
 */
export const useTrainingStore = create((set, get) => ({
    // Training state
    isTraining: false,
    isPaused: false,
    isCompiled: false,

    // Model info
    modelInfo: null, // { total_params, trainable_params, flops, memory_mb }

    // Current metrics
    currentEpoch: 0,
    currentBatch: 0,
    totalEpochs: 0,
    totalBatches: 0,

    currentLoss: null,
    currentAccuracy: null,
    currentLearningRate: null,

    // History
    lossHistory: [],
    accuracyHistory: [],

    // Hyperparameters
    hyperparameters: {
        learningRate: 0.001,
        batchSize: 64,
        epochs: 10,
        optimizer: 'adam',
        lossFunction: 'cross_entropy'
    },

    // Dataset
    dataset: 'mnist',

    // Actions
    setTraining: (isTraining) => set({ isTraining }),
    setPaused: (isPaused) => set({ isPaused }),
    setCompiled: (isCompiled) => set({ isCompiled }),

    setModelInfo: (info) => set({ modelInfo: info }),

    setHyperparameters: (params) => set((state) => ({
        hyperparameters: { ...state.hyperparameters, ...params }
    })),

    setDataset: (dataset) => set({ dataset }),

    updateMetrics: (metrics) => set((state) => {
        const newState = {
            currentEpoch: metrics.epoch ?? state.currentEpoch,
            currentBatch: metrics.batch ?? state.currentBatch,
            totalEpochs: metrics.total_epochs ?? state.totalEpochs,
            totalBatches: metrics.total_batches ?? state.totalBatches,
            currentLoss: metrics.loss ?? state.currentLoss,
            currentAccuracy: metrics.accuracy ?? state.currentAccuracy,
            currentLearningRate: metrics.learning_rate ?? state.currentLearningRate
        };

        // Add to history if loss/accuracy updated
        if (metrics.loss !== undefined) {
            newState.lossHistory = [...state.lossHistory, {
                epoch: metrics.epoch,
                batch: metrics.batch,
                value: metrics.loss
            }];
        }

        if (metrics.accuracy !== undefined) {
            newState.accuracyHistory = [...state.accuracyHistory, {
                epoch: metrics.epoch,
                batch: metrics.batch,
                value: metrics.accuracy
            }];
        }

        return newState;
    }),

    resetMetrics: () => set({
        currentEpoch: 0,
        currentBatch: 0,
        currentLoss: null,
        currentAccuracy: null,
        lossHistory: [],
        accuracyHistory: []
    }),

    // Start training
    startTraining: () => set({
        isTraining: true,
        isPaused: false
    }),

    // Pause training
    pauseTraining: () => set({
        isPaused: true
    }),

    // Resume training
    resumeTraining: () => set({
        isPaused: false
    }),

    // Stop training
    stopTraining: () => set({
        isTraining: false,
        isPaused: false
    })
}));
