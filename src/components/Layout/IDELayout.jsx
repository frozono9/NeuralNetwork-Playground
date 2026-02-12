import React from 'react';
import './styles/IDELayout.css';
import GraphEditor from '../GraphEditor/GraphCanvas';
import CodeEditor from '../CodeEditor/PyTorchEditor';
import Toolbar from './Toolbar';
import StatusBar from './StatusBar';
import InspectorPanel from '../Inspector/InspectorPanel';
import TrainingPanel from '../Training/TrainingPanel';

/**
 * Main IDE Layout
 * Three-pane layout: Graph | Code | Inspector
 */
export default function IDELayout() {
    const [graphWidth, setGraphWidth] = React.useState(40); // percentage
    const [codeWidth, setCodeWidth] = React.useState(35); // percentage
    const [showInspector, setShowInspector] = React.useState(true);
    const [showTraining, setShowTraining] = React.useState(true);

    return (
        <div className="ide-layout">
            <Toolbar />

            <div className="ide-content">
                {/* Graph Editor Pane */}
                <div
                    className="ide-pane graph-pane"
                    style={{ width: `${graphWidth}%` }}
                >
                    <div className="pane-header">
                        <h3>Visual Graph</h3>
                    </div>
                    <div className="pane-content">
                        <GraphEditor />
                    </div>
                </div>

                {/* Resizer */}
                <div className="pane-resizer" />

                {/* Code Editor Pane */}
                <div
                    className="ide-pane code-pane"
                    style={{ width: `${codeWidth}%` }}
                >
                    <div className="pane-header">
                        <h3>PyTorch Code</h3>
                    </div>
                    <div className="pane-content">
                        <CodeEditor />
                    </div>
                </div>

                {/* Resizer */}
                {showInspector && <div className="pane-resizer" />}

                {/* Inspector/Training Pane */}
                {showInspector && (
                    <div
                        className="ide-pane inspector-pane"
                        style={{ width: `${100 - graphWidth - codeWidth}%` }}
                    >
                        <div className="pane-tabs">
                            <button
                                className={showTraining ? 'active' : ''}
                                onClick={() => setShowTraining(true)}
                            >
                                Training
                            </button>
                            <button
                                className={!showTraining ? 'active' : ''}
                                onClick={() => setShowTraining(false)}
                            >
                                Inspector
                            </button>
                        </div>
                        <div className="pane-content">
                            {showTraining ? <TrainingPanel /> : <InspectorPanel />}
                        </div>
                    </div>
                )}
            </div>

            <StatusBar />
        </div>
    );
}
