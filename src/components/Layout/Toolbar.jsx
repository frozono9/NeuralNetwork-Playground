import React from 'react';
import { useTrainingStore } from '../../store/trainingStore';
import { useGraphStore } from '../../store/graphStore';
import wsManager from '../../services/websocket';
import { syncCoordinator } from '../../services/sync/SyncCoordinator';
import './styles/Toolbar.css';

export default function Toolbar() {
    const { isCompiled, isTraining } = useTrainingStore();
    const { nodes, edges } = useGraphStore();

    const handleCompile = () => {
        // Sync graph to code first
        syncCoordinator.syncGraphToCode();

        // Send graph to backend for compilation
        wsManager.compileModel({ nodes, edges });
    };

    const handleTrain = () => {
        const { hyperparameters, dataset } = useTrainingStore.getState();
        wsManager.startTraining({ ...hyperparameters, dataset });
    };

    const handlePause = () => {
        wsManager.pauseTraining();
    };

    const handleStop = () => {
        wsManager.stopTraining();
    };

    const handleExport = () => {
        const { code } = useCodeStore.getState();
        const blob = new Blob([code], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'model.py';
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="toolbar">
            <div className="toolbar-section">
                <div className="toolbar-logo">
                    <span className="gradient-text">Visual DL IDE</span>
                </div>
            </div>

            <div className="toolbar-section">
                <button
                    className="toolbar-btn primary"
                    onClick={handleCompile}
                    disabled={nodes.length === 0}
                >
                    <span>⚙️</span>
                    Compile
                </button>

                {!isTraining ? (
                    <button
                        className="toolbar-btn success"
                        onClick={handleTrain}
                        disabled={!isCompiled}
                    >
                        <span>▶️</span>
                        Train
                    </button>
                ) : (
                    <>
                        <button
                            className="toolbar-btn warning"
                            onClick={handlePause}
                        >
                            <span>⏸️</span>
                            Pause
                        </button>
                        <button
                            className="toolbar-btn error"
                            onClick={handleStop}
                        >
                            <span>⏹️</span>
                            Stop
                        </button>
                    </>
                )}

                <button
                    className="toolbar-btn"
                    onClick={handleExport}
                >
                    <span>💾</span>
                    Export
                </button>
            </div>

            <div className="toolbar-section">
                <div className="toolbar-status">
                    {isCompiled && <span className="status-badge success">Compiled</span>}
                    {isTraining && <span className="status-badge training animate-pulse">Training</span>}
                </div>
            </div>
        </div>
    );
}
