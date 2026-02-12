import { io } from 'socket.io-client';
import { useTrainingStore } from '../store/trainingStore';
import { useInspectionStore } from '../store/inspectionStore';
import { useGraphStore } from '../store/graphStore';

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    connect() {
        if (this.socket && this.connected) {
            console.log('WebSocket already connected');
            return;
        }

        this.socket = io('ws://localhost:8000', {
            transports: ['websocket'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: this.maxReconnectAttempts
        });

        this.setupEventHandlers();
    }

    setupEventHandlers() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.reconnectAttempts = 0;
        });

        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            this.connected = false;
        });

        this.socket.on('connect_error', (error) => {
            console.error('WebSocket connection error:', error);
            this.reconnectAttempts++;
        });

        // Training events
        this.socket.on('training_started', (data) => {
            console.log('Training started:', data);
            useTrainingStore.getState().startTraining();
        });

        this.socket.on('training_paused', () => {
            console.log('Training paused');
            useTrainingStore.getState().pauseTraining();
        });

        this.socket.on('training_stopped', () => {
            console.log('Training stopped');
            useTrainingStore.getState().stopTraining();
        });

        // Metrics updates
        this.socket.on('metrics_update', (data) => {
            useTrainingStore.getState().updateMetrics(data);
        });

        // Tensor inspection data
        this.socket.on('inspection_data', (data) => {
            const { node_id, activations, gradients, weights, statistics } = data;
            useInspectionStore.getState().updateNodeInspection(node_id, {
                activations,
                gradients,
                weights,
                statistics
            });
        });

        // Feature maps (for Conv layers)
        this.socket.on('feature_maps', (data) => {
            useInspectionStore.getState().setFeatureMaps(data);
        });

        // Attention weights
        this.socket.on('attention_weights', (data) => {
            useInspectionStore.getState().setAttentionWeights(data);
        });

        // Gradient flow data
        this.socket.on('gradient_flow', (data) => {
            useInspectionStore.getState().setGradientFlow(data);

            // Update node visualizations
            const graphStore = useGraphStore.getState();
            Object.entries(data).forEach(([nodeId, flowData]) => {
                graphStore.updateNode(nodeId, {
                    gradientMagnitude: flowData.magnitude,
                    gradientSign: flowData.sign
                });
            });
        });

        // Dead neurons detection
        this.socket.on('dead_neurons', (data) => {
            useInspectionStore.getState().setDeadNeurons(data);
        });

        // Model compilation result
        this.socket.on('model_compiled', (data) => {
            console.log('Model compiled:', data);
            useTrainingStore.getState().setCompiled(true);
            useTrainingStore.getState().setModelInfo(data.model_info);
        });

        // Errors
        this.socket.on('error', (data) => {
            console.error('Server error:', data);
        });
    }

    // Send messages to server
    emit(event, data) {
        if (!this.connected) {
            console.warn('WebSocket not connected, cannot emit:', event);
            return;
        }
        this.socket.emit(event, data);
    }

    // Compile model
    compileModel(graphData) {
        this.emit('compile_model', graphData);
    }

    // Training controls
    startTraining(hyperparameters) {
        this.emit('start_training', hyperparameters);
    }

    pauseTraining() {
        this.emit('pause_training');
    }

    resumeTraining() {
        this.emit('resume_training');
    }

    stopTraining() {
        this.emit('stop_training');
    }

    stepBatch() {
        this.emit('step_batch');
    }

    // Request inspection data for a specific node
    requestInspection(nodeId) {
        this.emit('request_inspection', { node_id: nodeId });
    }

    // Parse code to graph
    parseCode(code) {
        this.emit('parse_code', { code });
    }

    // Disconnect
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.connected = false;
        }
    }
}

// Singleton instance
export const wsManager = new WebSocketManager();

// Auto-connect on import
if (typeof window !== 'undefined') {
    wsManager.connect();
}

export default wsManager;
