import React from 'react';
import { useInspectionStore } from '../../store/inspectionStore';

export default function InspectorPanel() {
    const { selectedNodeId, inspectionData } = useInspectionStore();

    if (!selectedNodeId) {
        return (
            <div style={{ padding: '24px', color: 'var(--color-text-secondary)' }}>
                <p>Select a node to inspect its tensors</p>
            </div>
        );
    }

    return (
        <div style={{ padding: '24px' }}>
            <h3>Inspector</h3>
            <p>Node: {selectedNodeId}</p>
            {inspectionData && (
                <div>
                    <h4>Statistics</h4>
                    <pre>{JSON.stringify(inspectionData.statistics, null, 2)}</pre>
                </div>
            )}
        </div>
    );
}
