import { useCodeStore } from '../../store/codeStore';
import { useGraphStore } from '../../store/graphStore';
import { graphToCode } from './GraphToCode';
import { codeToGraph } from './CodeToGraph';

/**
 * Sync Coordinator
 * Manages bidirectional synchronization between graph and code
 */

class SyncCoordinator {
    constructor() {
        this.syncInProgress = false;
        this.debounceTimer = null;
        this.debounceDelay = 500; // ms
    }

    /**
     * Sync graph to code
     */
    async syncGraphToCode() {
        const codeStore = useCodeStore.getState();
        const graphStore = useGraphStore.getState();

        // Prevent sync if already in progress or if code is source of truth
        if (this.syncInProgress || codeStore.syncLock) {
            return;
        }

        if (codeStore.sourceOfTruth === 'code') {
            return;
        }

        try {
            this.syncInProgress = true;
            codeStore.setSyncLock(true);

            // Generate code from graph
            const code = graphToCode(graphStore.nodes, graphStore.edges);

            // Update code store
            codeStore.updateFromGraph(code);

        } catch (error) {
            console.error('Error syncing graph to code:', error);
        } finally {
            this.syncInProgress = false;
            codeStore.setSyncLock(false);
        }
    }

    /**
     * Sync code to graph
     */
    async syncCodeToGraph(code) {
        const codeStore = useCodeStore.getState();
        const graphStore = useGraphStore.getState();

        // Prevent sync if already in progress or if graph is source of truth
        if (this.syncInProgress || codeStore.syncLock) {
            return;
        }

        if (codeStore.sourceOfTruth === 'graph') {
            return;
        }

        try {
            this.syncInProgress = true;
            codeStore.setSyncLock(true);

            // Parse code to graph
            const { nodes, edges } = await codeToGraph(code);

            // Update graph store
            graphStore.setNodes(nodes);
            graphStore.setEdges(edges);

            // Clear pending sync
            codeStore.clearPendingSync();

        } catch (error) {
            console.error('Error syncing code to graph:', error);
            codeStore.setSyntaxErrors([{
                message: error.message,
                line: 0
            }]);
        } finally {
            this.syncInProgress = false;
            codeStore.setSyncLock(false);
        }
    }

    /**
     * Debounced sync code to graph
     */
    debouncedSyncCodeToGraph(code) {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
            this.syncCodeToGraph(code);
        }, this.debounceDelay);
    }

    /**
     * Handle graph change
     */
    onGraphChange() {
        const codeStore = useCodeStore.getState();

        // Only sync if graph is source of truth
        if (codeStore.sourceOfTruth === 'graph') {
            this.syncGraphToCode();
        }
    }

    /**
     * Handle code change
     */
    onCodeChange(code) {
        const codeStore = useCodeStore.getState();

        // Only sync if code is source of truth
        if (codeStore.sourceOfTruth === 'code') {
            this.debouncedSyncCodeToGraph(code);
        }
    }
}

// Singleton instance
export const syncCoordinator = new SyncCoordinator();

export default syncCoordinator;
