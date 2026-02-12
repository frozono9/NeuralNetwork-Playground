/**
 * Graph to Code Generator
 * Converts React Flow graph to PyTorch code
 */

export class GraphToCode {
    constructor(nodes, edges) {
        this.nodes = nodes;
        this.edges = edges;
    }

    /**
     * Generate complete PyTorch model code
     */
    generate() {
        const sortedNodes = this.topologicalSort();
        const imports = this.generateImports();
        const classDefinition = this.generateClass(sortedNodes);

        return `${imports}\n\n${classDefinition}`;
    }

    /**
     * Topological sort to determine layer order
     */
    topologicalSort() {
        const adjacencyList = new Map();
        const inDegree = new Map();

        // Initialize
        this.nodes.forEach(node => {
            adjacencyList.set(node.id, []);
            inDegree.set(node.id, 0);
        });

        // Build graph
        this.edges.forEach(edge => {
            adjacencyList.get(edge.source).push(edge.target);
            inDegree.set(edge.target, inDegree.get(edge.target) + 1);
        });

        // Kahn's algorithm
        const queue = [];
        const sorted = [];

        inDegree.forEach((degree, nodeId) => {
            if (degree === 0) {
                queue.push(nodeId);
            }
        });

        while (queue.length > 0) {
            const nodeId = queue.shift();
            const node = this.nodes.find(n => n.id === nodeId);
            sorted.push(node);

            adjacencyList.get(nodeId).forEach(neighborId => {
                inDegree.set(neighborId, inDegree.get(neighborId) - 1);
                if (inDegree.get(neighborId) === 0) {
                    queue.push(neighborId);
                }
            });
        }

        return sorted;
    }

    /**
     * Generate import statements
     */
    generateImports() {
        return `import torch
import torch.nn as nn
import torch.nn.functional as F`;
    }

    /**
     * Generate class definition
     */
    generateClass(sortedNodes) {
        const initMethod = this.generateInit(sortedNodes);
        const forwardMethod = this.generateForward(sortedNodes);

        return `class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
${initMethod}
    
${forwardMethod}`;
    }

    /**
     * Generate __init__ method
     */
    generateInit(sortedNodes) {
        const layers = sortedNodes
            .filter(node => node.data.type !== 'input' && node.data.type !== 'output')
            .map(node => this.generateLayerInit(node))
            .filter(Boolean);

        return layers.map(layer => `        ${layer}`).join('\n');
    }

    /**
     * Generate layer initialization code
     */
    generateLayerInit(node) {
        const { type, params } = node.data;
        const layerName = this.getLayerName(node.id);

        switch (type) {
            case 'linear':
                return `self.${layerName} = nn.Linear(${params.in_features}, ${params.out_features})`;

            case 'conv2d':
                return `self.${layerName} = nn.Conv2d(${params.in_channels}, ${params.out_channels}, kernel_size=${params.kernel_size})`;

            case 'batchnorm2d':
                return `self.${layerName} = nn.BatchNorm2d(${params.num_features})`;

            case 'dropout':
                return `self.${layerName} = nn.Dropout(p=${params.p})`;

            case 'maxpool2d':
                return `self.${layerName} = nn.MaxPool2d(kernel_size=${params.kernel_size})`;

            // Activation functions don't need initialization
            case 'relu':
            case 'sigmoid':
            case 'tanh':
                return null;

            default:
                return `# Unknown layer type: ${type}`;
        }
    }

    /**
     * Generate forward method
     */
    generateForward(sortedNodes) {
        const forwardLines = ['    def forward(self, x):'];

        // Track tensor names
        const tensorNames = new Map();
        tensorNames.set(sortedNodes[0].id, 'x');

        sortedNodes.forEach((node, idx) => {
            if (node.data.type === 'input') return;
            if (node.data.type === 'output') return;

            const inputTensor = this.getInputTensor(node.id, tensorNames);
            const outputTensor = this.getOutputTensor(node.id, idx);
            const operation = this.generateForwardOperation(node, inputTensor);

            forwardLines.push(`        ${outputTensor} = ${operation}`);
            tensorNames.set(node.id, outputTensor);
        });

        // Return final tensor
        const finalNode = sortedNodes[sortedNodes.length - 1];
        const finalTensor = tensorNames.get(finalNode.id) || 'x';
        forwardLines.push(`        return ${finalTensor}`);

        return forwardLines.join('\n');
    }

    /**
     * Get input tensor for a node
     */
    getInputTensor(nodeId, tensorNames) {
        const incomingEdge = this.edges.find(edge => edge.target === nodeId);
        if (!incomingEdge) return 'x';
        return tensorNames.get(incomingEdge.source) || 'x';
    }

    /**
     * Get output tensor name
     */
    getOutputTensor(nodeId, idx) {
        return `x${idx}`;
    }

    /**
     * Generate forward operation for a node
     */
    generateForwardOperation(node, inputTensor) {
        const { type } = node.data;
        const layerName = this.getLayerName(node.id);

        switch (type) {
            case 'linear':
            case 'conv2d':
            case 'batchnorm2d':
            case 'dropout':
            case 'maxpool2d':
                return `self.${layerName}(${inputTensor})`;

            case 'relu':
                return `F.relu(${inputTensor})`;

            case 'sigmoid':
                return `torch.sigmoid(${inputTensor})`;

            case 'tanh':
                return `torch.tanh(${inputTensor})`;

            default:
                return `${inputTensor}  # Unknown operation: ${type}`;
        }
    }

    /**
     * Get layer name from node ID
     */
    getLayerName(nodeId) {
        return `layer_${nodeId.replace(/-/g, '_')}`;
    }
}

/**
 * Convert graph to code
 */
export function graphToCode(nodes, edges) {
    const generator = new GraphToCode(nodes, edges);
    return generator.generate();
}
