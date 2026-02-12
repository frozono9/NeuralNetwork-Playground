import React from 'react';
import { useTrainingStore } from '../../store/trainingStore';
import { useGraphStore } from '../../store/graphStore';
import wsManager from '../../services/websocket';
import './styles/StatusBar.css';

export default function StatusBar() {
    const { modelInfo, currentEpoch, totalEpochs, currentLoss, currentAccuracy } = useTrainingStore();
    const { nodes } = useGraphStore();
    const [connectionStatus, setConnectionStatus] = React.useState('disconnected');

    React.useEffect(() => {
        const checkConnection = () => {
            setConnectionStatus(wsManager.connected ? 'connected' : 'disconnected');
        };

        const interval = setInterval(checkConnection, 1000);
        checkConnection();

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="status-bar">
            <div className="status-section">
                <span className={`connection-indicator ${connectionStatus}`} />
                <span className="status-text">
                    {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
                </span>
            </div>

            <div className="status-section">
                <span className="status-text">Nodes: {nodes.length}</span>
            </div>

            {modelInfo && (
                <>
                    <div className="status-section">
                        <span className="status-text">
                            Params: {modelInfo.total_params?.toLocaleString()}
                        </span>
                    </div>
                    <div className="status-section">
                        <span className="status-text">
                            FLOPs: {modelInfo.flops?.toLocaleString()}
                        </span>
                    </div>
                    <div className="status-section">
                        <span className="status-text">
                            Memory: {modelInfo.memory_mb?.toFixed(2)} MB
                        </span>
                    </div>
                </>
            )}

            {currentEpoch > 0 && (
                <>
                    <div className="status-section">
                        <span className="status-text">
                            Epoch: {currentEpoch}/{totalEpochs}
                        </span>
                    </div>
                    {currentLoss !== null && (
                        <div className="status-section">
                            <span className="status-text">
                                Loss: {currentLoss.toFixed(4)}
                            </span>
                        </div>
                    )}
                    {currentAccuracy !== null && (
                        <div className="status-section">
                            <span className="status-text">
                                Acc: {(currentAccuracy * 100).toFixed(2)}%
                            </span>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
