import React from 'react';
import ReactFlow, { Background, Controls, MiniMap } from 'reactflow';
import 'reactflow/dist/style.css';
import { useGraphStore } from '../../store/graphStore';
import { syncCoordinator } from '../../services/sync/SyncCoordinator';

export default function GraphCanvas() {
    const { nodes, edges, setNodes, setEdges } = useGraphStore();

    const onNodesChange = React.useCallback((changes) => {
        setNodes(changes);
        // Trigger sync to code
        syncCoordinator.onGraphChange();
    }, [setNodes]);

    const onEdgesChange = React.useCallback((changes) => {
        setEdges(changes);
        // Trigger sync to code
        syncCoordinator.onGraphChange();
    }, [setEdges]);

    return (
        <div style={{ width: '100%', height: '100%' }}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                fitView
            >
                <Background />
                <Controls />
                <MiniMap />
            </ReactFlow>
        </div>
    );
}
