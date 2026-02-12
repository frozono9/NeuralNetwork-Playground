import React from 'react';
import { useTrainingStore } from '../../store/trainingStore';

export default function TrainingPanel() {
    const {
        currentEpoch,
        totalEpochs,
        currentLoss,
        currentAccuracy,
        lossHistory
    } = useTrainingStore();

    return (
        <div style={{ padding: '24px' }}>
            <h3>Training</h3>

            {currentEpoch > 0 && (
                <div>
                    <p>Epoch: {currentEpoch}/{totalEpochs}</p>
                    {currentLoss !== null && <p>Loss: {currentLoss.toFixed(4)}</p>}
                    {currentAccuracy !== null && <p>Accuracy: {(currentAccuracy * 100).toFixed(2)}%</p>}

                    <div style={{ marginTop: '20px' }}>
                        <h4>Loss History</h4>
                        <div style={{ height: '200px', background: 'var(--color-bg-secondary)', borderRadius: '8px', padding: '12px' }}>
                            {lossHistory.length > 0 ? (
                                <p>{lossHistory.length} data points</p>
                            ) : (
                                <p>No data yet</p>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {currentEpoch === 0 && (
                <p style={{ color: 'var(--color-text-secondary)' }}>
                    Click "Train" to start training
                </p>
            )}
        </div>
    );
}
