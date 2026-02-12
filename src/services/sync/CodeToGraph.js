import wsManager from '../websocket';

/**
 * Code to Graph Parser
 * Parses PyTorch code and reconstructs the graph
 */

export class CodeToGraph {
    constructor(code) {
        this.code = code;
    }

    /**
     * Parse code and return graph structure
     * This sends code to backend for AST parsing
     */
    async parse() {
        return new Promise((resolve, reject) => {
            // Send code to backend for parsing
            wsManager.socket.emit('parse_code', { code: this.code });

            // Listen for response
            const timeout = setTimeout(() => {
                reject(new Error('Code parsing timeout'));
            }, 5000);

            wsManager.socket.once('code_parsed', (data) => {
                clearTimeout(timeout);

                if (data.error) {
                    reject(new Error(data.error));
                } else {
                    resolve({
                        nodes: data.nodes,
                        edges: data.edges
                    });
                }
            });
        });
    }
}

/**
 * Convert code to graph
 */
export async function codeToGraph(code) {
    const parser = new CodeToGraph(code);
    return await parser.parse();
}
