import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { io } from 'socket.io-client';
import ReactFlow, {
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    Panel,
    Handle,
    Position,
    ReactFlowProvider,
    useReactFlow
} from 'reactflow';
import 'reactflow/dist/style.css';
import './styles/index.css';

// ============================================================================
// CODE GENERATOR
// ============================================================================

function generateCode(config) {
    const {
        dataset,
        inputSize,
        hiddenLayers,
        outputSize,
        lossFunction,
        optimizer,
        learningRate,
        epochs
    } = config;

    let layerDefs = [];
    let forwardOps = [];

    const firstLayerSize = hiddenLayers.length > 0 ? hiddenLayers[0].size : outputSize;
    layerDefs.push(`        self.fc1 = nn.Linear(${inputSize}, ${firstLayerSize})`);

    for (let i = 0; i < hiddenLayers.length; i++) {
        const layer = hiddenLayers[i];
        const nextSize = i < hiddenLayers.length - 1 ? hiddenLayers[i + 1].size : outputSize;
        layerDefs.push(`        self.fc${i + 2} = nn.Linear(${layer.size}, ${nextSize})`);
    }

    const hasDropout = hiddenLayers.some(l => l.dropout > 0);
    if (hasDropout) {
        layerDefs.push(`        self.dropout = nn.Dropout(0.5)`);
    }

    forwardOps.push(`        x = x.view(-1, ${inputSize})`);

    for (let i = 0; i < hiddenLayers.length; i++) {
        const layer = hiddenLayers[i];
        const activation = layer.activation.toLowerCase();

        if (activation === 'relu') {
            forwardOps.push(`        x = torch.relu(self.fc${i + 1}(x))`);
        } else if (activation === 'sigmoid') {
            forwardOps.push(`        x = torch.sigmoid(self.fc${i + 1}(x))`);
        } else if (activation === 'tanh') {
            forwardOps.push(`        x = torch.tanh(self.fc${i + 1}(x))`);
        } else {
            forwardOps.push(`        x = self.fc${i + 1}(x)`);
        }

        if (layer.dropout > 0) {
            forwardOps.push(`        x = self.dropout(x)`);
        }
    }

    const lastLayerNum = hiddenLayers.length + 1;
    forwardOps.push(`        x = self.fc${lastLayerNum}(x)`);
    forwardOps.push(`        return x`);

    return `import torch
import torch.nn as nn
import torch.optim as optim

# Model Definition
class Model(nn.Module):
    def __init__(self):
        super().__init__()
${layerDefs.join('\n')}
    
    def forward(self, x):
${forwardOps.join('\n')}

# Hyperparameters
LR = ${learningRate}
EPOCHS = ${epochs}
OPTIMIZER = "${optimizer}"
LOSS_FUNC = "${lossFunction}"

# Initialization
model = Model()
optimizer = optim.${optimizer}(model.parameters(), lr=LR)
criterion = nn.${lossFunction}()

print(f"Initialized {OPTIMIZER} optimizer with LR={LR}")`;
}

// ============================================================================
// INDIVIDUAL NEURON NODE COMPONENT
// ============================================================================

function IndividualNeuronNode({ data, selected }) {
    const activationColors = {
        'ReLU': '#00ff41', // Matrix Green
        'Sigmoid': '#00d4ff',
        'Tanh': '#ff0055',
        'LeakyReLU': '#00ff41',
        'None': '#1a1a1a'
    };

    const isOutputNode = data.layerType === 'output';
    const isInputNode = data.layerType === 'input';
    const baseColor = isOutputNode ? '#ffcf00' : (activationColors[data.activation] || '#1a1a1a');
    const activationValue = data.activationValue || 0;
    
    // CIFAR-10 class names for output neurons
    const cifar10Names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    const isCifar = data.dataset === 'cifar10';
    
    // Industrial styling
    const size = 32;
    
    return (
        <div
            onClick={(e) => {
                console.log('IndividualNeuronNode clicked (DOM event)', data.neuronIndex);
                // Optionally stop propagation if we want to bypass ReactFlow's handler, 
                // but usually better to let ReactFlow handle it if possible.
            }}
            style={{
            width: `${size}px`,
            height: `${size}px`,
            background: selected ? '#00ff41' : (activationValue > 0.05 ? baseColor : '#fff'),
            border: selected ? `4px solid #ff0055` : `2px solid #000`,
            borderRadius: isInputNode ? '4px' : '0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.1s ease-out',
            boxShadow: selected ? '0 0 20px #ff0055' : (activationValue > 0.5 ? `0 0 15px ${baseColor}` : 'none'),
            transform: selected ? 'scale(1.3)' : `scale(${1 + activationValue * 0.15})`,
            position: 'relative',
            cursor: 'pointer',
            zIndex: selected ? 1000 : 1
        }}
            title={`${data.layerName} - Index ${data.neuronIndex}\nActivation: ${activationValue.toFixed(4)}`}
        >
            <div style={{ 
                fontSize: isOutputNode && isCifar ? '8px' : '11px', 
                color: activationValue > 0.5 ? '#fff' : '#000', 
                fontWeight: 'bold',
                fontFamily: 'monospace',
                pointerEvents: 'none',
                textAlign: 'center',
                lineHeight: '1.1'
            }}>
                {isOutputNode ? (isCifar ? cifar10Names[data.neuronIndex] || data.neuronIndex : data.neuronIndex) : ''}
                {isInputNode && activationValue > 0.1 ? '●' : ''}
            </div>

            <Handle
                type="target"
                position={Position.Left}
                style={{ background: '#000', borderRadius: 0, width: '4px', height: '4px', left: '-3px' }}
                isConnectable={false}
            />

            <Handle
                type="source"
                position={Position.Right}
                style={{ background: '#000', borderRadius: 0, width: '4px', height: '4px', right: '-3px' }}
                isConnectable={false}
            />
        </div>
    );
}

const nodeTypes = {
    neuron: IndividualNeuronNode
};

function getDatasetSpec(dataset) {
    const name = (dataset || 'mnist').toLowerCase();
    if (name === 'cifar10') {
        return { name: 'cifar10', label: 'CIFAR-10', width: 32, height: 32, channels: 3, inputMode: 'image' };
    }
    return { name: 'mnist', label: 'MNIST', width: 28, height: 28, channels: 1, inputMode: 'draw' };
}

// ============================================================================
// DRAWING CANVAS COMPONENT
// ============================================================================

function DrawingCanvas({ onPredict, onClear, isEnabled, dataset }) {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const isDrawingRef = useRef(false);
    const lastCellRef = useRef(null); // {x, y} in grid coords
    const spec = getDatasetSpec(dataset);
    const planeSize = spec.width * spec.height;
    const totalSize = planeSize * spec.channels;
    const pixelsRef = useRef(new Float32Array(totalSize));
    const gridRef = useRef(null);
    const rafRef = useRef(null);
    const [isPredicting, setIsPredicting] = useState(false);

    const renderCanvas = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        if (gridRef.current) {
            ctx.drawImage(gridRef.current, 0, 0);
        } else {
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        const cell = canvas.width / spec.width;
        const pixels = pixelsRef.current;

        ctx.fillStyle = '#000';
        for (let y = 0; y < spec.height; y++) {
            for (let x = 0; x < spec.width; x++) {
                const baseIdx = y * spec.width + x;
                if (spec.channels === 1) {
                    const v = pixels[baseIdx] || 0;
                    if (v > 0) {
                        // Use a bit of overlap to avoid background bleed
                        ctx.fillRect(x * cell, y * cell, cell + 0.3, cell + 0.3);
                    }
                } else {
                    // RGB preview for CIFAR.
                    const r = pixels[0 * planeSize + baseIdx] || 0;
                    const g = pixels[1 * planeSize + baseIdx] || 0;
                    const b = pixels[2 * planeSize + baseIdx] || 0;
                    const rr = Math.max(0, Math.min(255, Math.round(r * 255)));
                    const gg = Math.max(0, Math.min(255, Math.round(g * 255)));
                    const bb = Math.max(0, Math.min(255, Math.round(b * 255)));
                    if (rr + gg + bb > 0) {
                        ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
                        ctx.fillRect(x * cell, y * cell, cell + 0.3, cell + 0.3);
                    }
                }
            }
        }
    }, [planeSize, spec.channels, spec.height, spec.width]);

    const scheduleRender = useCallback(() => {
        if (rafRef.current) return;
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            renderCanvas();
        });
    }, [renderCanvas]);

    useEffect(() => {
        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, []);

    // Reset pixel buffer when dataset changes.
    useEffect(() => {
        pixelsRef.current = new Float32Array(totalSize);
        isDrawingRef.current = false;
        setIsDrawing(false);
        lastCellRef.current = null;
        gridRef.current = null;
        renderCanvas();
    }, [renderCanvas, totalSize]);

    // Native mouse listeners (most reliable across browsers).
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const onDown = (e) => {
            e.preventDefault();
            isDrawingRef.current = true;
            setIsDrawing(true);
            lastCellRef.current = null;
            handleDraw(e);
        };

        const onMove = (e) => {
            if (!isDrawingRef.current) return;

            // If we missed mouseup (e.g., window lost focus), stop drawing.
            if (typeof e.buttons === 'number' && e.buttons === 0) {
                isDrawingRef.current = false;
                setIsDrawing(false);
                lastCellRef.current = null;
                return;
            }

            e.preventDefault();
            handleDraw(e);
        };

        const onUp = (e) => {
            e.preventDefault?.();
            isDrawingRef.current = false;
            setIsDrawing(false);
            lastCellRef.current = null;
        };

        const onBlur = () => {
            isDrawingRef.current = false;
            setIsDrawing(false);
            lastCellRef.current = null;
        };

        canvas.addEventListener('mousedown', onDown);
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        window.addEventListener('blur', onBlur);

        return () => {
            canvas.removeEventListener('mousedown', onDown);
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
            window.removeEventListener('blur', onBlur);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const loadImageToPixels = useCallback(async (file) => {
        if (!file) return;
        if (spec.channels !== 3) return;

        const url = URL.createObjectURL(file);
        try {
            const img = new Image();
            img.decoding = 'async';
            img.src = url;
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
            });

            const off = document.createElement('canvas');
            off.width = spec.width;
            off.height = spec.height;
            const ctx = off.getContext('2d', { willReadFrequently: true });
            if (!ctx) return;

            // Fit image into square without distortion: center-crop then scale.
            const s = Math.min(img.naturalWidth, img.naturalHeight);
            const sx = Math.floor((img.naturalWidth - s) / 2);
            const sy = Math.floor((img.naturalHeight - s) / 2);

            ctx.clearRect(0, 0, off.width, off.height);
            ctx.drawImage(img, sx, sy, s, s, 0, 0, off.width, off.height);

            const imageData = ctx.getImageData(0, 0, off.width, off.height);
            const buf = new Float32Array(totalSize);
            const plane = planeSize;

            // Channel-first [C][H][W] floats in [0,1].
            for (let i = 0; i < plane; i++) {
                const p = i * 4;
                buf[0 * plane + i] = (imageData.data[p + 0] || 0) / 255;
                buf[1 * plane + i] = (imageData.data[p + 1] || 0) / 255;
                buf[2 * plane + i] = (imageData.data[p + 2] || 0) / 255;
            }

            pixelsRef.current = buf;
            isDrawingRef.current = false;
            setIsDrawing(false);
            lastCellRef.current = null;
            scheduleRender();
        } catch (e) {
            console.error('Failed to load image:', e);
            alert('Failed to load image. Try a different file.');
        } finally {
            URL.revokeObjectURL(url);
        }
    }, [planeSize, scheduleRender, spec.channels, spec.height, spec.width, totalSize]);

    // Pre-render the grid once to an offscreen canvas for smoother drawing.
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const off = document.createElement('canvas');
        off.width = canvas.width;
        off.height = canvas.height;
        const ctx = off.getContext('2d');
        if (!ctx) return;

        const cell = off.width / spec.width;
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, off.width, off.height);

        ctx.strokeStyle = '#f0f0f0';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= spec.width; i++) {
            ctx.beginPath();
            ctx.moveTo(i * cell, 0);
            ctx.lineTo(i * cell, off.height);
            ctx.stroke();
        }

        for (let i = 0; i <= spec.height; i++) {
            ctx.beginPath();
            ctx.moveTo(0, i * cell);
            ctx.lineTo(off.width, i * cell);
            ctx.stroke();
        }

        gridRef.current = off;
        renderCanvas();
    }, [renderCanvas, scheduleRender, spec.height, spec.width]);

    const stopDrawing = useCallback(() => {
        isDrawingRef.current = false;
        setIsDrawing(false);
    }, []);

    const handleDraw = (e) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / (rect.width / spec.width));
        const y = Math.floor((e.clientY - rect.top) / (rect.height / spec.height));

        if (x >= 0 && x < spec.width && y >= 0 && y < spec.height) {
            const pixels = pixelsRef.current;

            const stamp = (cx, cy) => {
                // Draw with brush (3x3)
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const nx = cx + dx;
                        const ny = cy + dy;
                        if (nx >= 0 && nx < spec.width && ny >= 0 && ny < spec.height) {
                            const nidx = ny * spec.width + nx;
                            const distance = Math.sqrt(dx * dx + dy * dy);
                            const strength = Math.max(0, 1 - distance / 2);
                            if (strength <= 0) continue;

                            // Set (not add) so marks are dark immediately.
                            if (spec.channels === 1) {
                                pixels[nidx] = Math.max(pixels[nidx], strength);
                            } else {
                                // CIFAR: channel-first layout [C][H][W]. Draw grayscale into all channels.
                                for (let c = 0; c < spec.channels; c++) {
                                    const idx = c * planeSize + nidx;
                                    pixels[idx] = Math.max(pixels[idx], strength);
                                }
                            }
                        }
                    }
                }
            };

            const last = lastCellRef.current;
            if (last && typeof last.x === 'number' && typeof last.y === 'number') {
                const dx = x - last.x;
                const dy = y - last.y;
                const steps = Math.max(Math.abs(dx), Math.abs(dy));
                if (steps === 0) {
                    stamp(x, y);
                } else {
                    for (let i = 1; i <= steps; i++) {
                        const ix = Math.round(last.x + (dx * i) / steps);
                        const iy = Math.round(last.y + (dy * i) / steps);
                        stamp(ix, iy);
                    }
                }
            } else {
                stamp(x, y);
            }

            lastCellRef.current = { x, y };

            // Paint once per frame.
            scheduleRender();
        }
    };

    const handleClear = () => {
        pixelsRef.current.fill(0);
        isDrawingRef.current = false;
        setIsDrawing(false);
        lastCellRef.current = null;
        renderCanvas();
        if (onClear) onClear();
    };

    const handlePredict = async () => {
        if (!isEnabled) {
            alert('Please compile and train the model first');
            return;
        }

        setIsPredicting(true);

        try {
            const response = await fetch('http://localhost:8000/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels: Array.from(pixelsRef.current) })
            });

            const result = await response.json();

            if (result.error) {
                alert('Prediction error: ' + result.error);
            } else {
                onPredict(result, Array.from(pixelsRef.current));
            }
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Failed to predict. Make sure the backend is running.');
        } finally {
            setIsPredicting(false);
        }
    };

    return (
        <div style={{ padding: '24px', borderBottom: '1px solid #000' }}>
            <h3 style={{ marginTop: 0, fontSize: '11px', marginBottom: '20px', color: '#000', textTransform: 'uppercase', letterSpacing: '0.15em', fontWeight: 'bold' }}>
                INPUT_CANVAS // RAW_DATA
            </h3>

            {spec.inputMode === 'image' && (
                <div style={{ marginBottom: '14px' }}>
                    <div style={{ fontSize: '10px', color: '#666', marginBottom: '8px', fontWeight: 'bold', letterSpacing: '0.08em' }}>
                        IMAGE_INPUT (auto crop + resize to {spec.width}x{spec.height})
                    </div>
                    <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => loadImageToPixels(e.target.files?.[0])}
                        style={{ width: '100%', fontFamily: 'monospace', fontSize: '11px' }}
                    />
                </div>
            )}

            <div style={{
                position: 'relative',
                width: '240px',
                height: '240px',
                border: '2px solid #000',
                margin: '0 auto 20px auto',
                background: '#fff'
            }}>
                <canvas
                    ref={canvasRef}
                    width={240}
                    height={240}
                    style={{ cursor: 'crosshair', display: 'block', width: '100%', height: '100%', touchAction: 'none' }}
                />
            </div>

            <div style={{ display: 'flex', gap: '8px', justifyContent: 'stretch' }}>
                <button
                    onClick={handleClear}
                    style={{
                        flex: 1,
                        padding: '10px',
                        background: '#fff',
                        border: '1px solid #000',
                        color: '#000',
                        cursor: 'pointer',
                        fontSize: '10px',
                        fontWeight: 'bold',
                        letterSpacing: '0.1em'
                    }}>
                    CLEAR_DATA
                </button>
                <button
                    onClick={handlePredict}
                    disabled={isPredicting || !isEnabled}
                    style={{
                        flex: 2,
                        padding: '10px',
                        background: '#000',
                        border: '1px solid #000',
                        color: '#fff',
                        cursor: 'pointer',
                        fontSize: '10px',
                        fontWeight: 'bold',
                        letterSpacing: '0.1em',
                        opacity: (isPredicting || !isEnabled) ? 0.3 : 1
                    }}>
                    {isPredicting ? 'PROCESSING...' : 'RUN_INFERENCE'}
                </button>
            </div>
        </div>
    );
}

// ============================================================================
// NETWORK BUILDER COMPONENT
// ============================================================================

function NetworkBuilder({ onConfigChange, onGenerateCode, onApplyModel }) {
    const [dataset, setDataset] = useState('mnist');
    const [inputSize, setInputSize] = useState(784);
    const [numHiddenLayers, setNumHiddenLayers] = useState(2);
    const [hiddenLayers, setHiddenLayers] = useState([
        { size: 16, activation: 'ReLU', dropout: 0 },
        { size: 16, activation: 'ReLU', dropout: 0 }
    ]);
    const [outputSize, setOutputSize] = useState(10);
    const [lossFunction, setLossFunction] = useState('CrossEntropyLoss');
    const [optimizer, setOptimizer] = useState('Adam');
    const [learningRate, setLearningRate] = useState(0.001);
    const [epochs, setEpochs] = useState(10);

    useEffect(() => {
        if (dataset === 'mnist') {
            setInputSize(784);
            setOutputSize(10);
        } else if (dataset === 'cifar10') {
            setInputSize(3072);
            setOutputSize(10);
        }
    }, [dataset]);

    useEffect(() => {
        const newLayers = [];
        for (let i = 0; i < numHiddenLayers; i++) {
            if (i < hiddenLayers.length) {
                newLayers.push(hiddenLayers[i]);
            } else {
                newLayers.push({ size: 16, activation: 'ReLU', dropout: 0 });
            }
        }
        setHiddenLayers(newLayers);
    }, [numHiddenLayers]);

    useEffect(() => {
        const config = {
            dataset,
            inputSize,
            hiddenLayers,
            outputSize,
            lossFunction,
            optimizer,
            learningRate,
            epochs
        };
        onConfigChange(config);
    }, [dataset, inputSize, hiddenLayers, outputSize, lossFunction, optimizer, learningRate, epochs]);

    const updateLayer = (index, field, value) => {
        const newLayers = [...hiddenLayers];
        newLayers[index] = { ...newLayers[index], [field]: value };
        setHiddenLayers(newLayers);
    };

    const labelStyle = { display: 'block', marginBottom: '8px', color: '#666', fontWeight: 'bold', fontSize: '10px', textTransform: 'uppercase' };
    const inputStyle = {
        width: '100%',
        padding: '8px',
        background: '#fff',
        border: '1px solid #ddd',
        borderRadius: '0',
        color: '#1a1a1a',
        fontSize: '12px',
        fontFamily: 'monospace',
        marginBottom: '16px'
    };

    return (
        <div style={{ padding: '20px' }}>
            <h3 style={{ marginTop: 0, fontSize: '13px', marginBottom: '24px', color: '#000', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                Network Architecture
            </h3>

            <div style={{ marginBottom: '24px' }}>
                <label style={labelStyle}>Dataset</label>
                <select value={dataset} onChange={(e) => setDataset(e.target.value)} style={inputStyle}>
                    <option value="mnist">MNIST (28x28 grayscale)</option>
                    <option value="cifar10">CIFAR-10 (32x32 RGB)</option>
                </select>

                <label style={labelStyle}>Hidden Layers</label>
                <input
                    type="number"
                    min="1"
                    max="10"
                    value={numHiddenLayers}
                    onChange={(e) => {
                        const next = parseInt(e.target.value, 10);
                        if (Number.isNaN(next)) return;
                        setNumHiddenLayers(Math.max(1, Math.min(10, next)));
                    }}
                    style={inputStyle}
                />
            </div>

            <div style={{ marginBottom: '24px' }}>
                <label style={labelStyle}>Layer Configuration</label>
                {hiddenLayers.map((layer, idx) => (
                    <div key={idx} style={{
                        marginBottom: '16px',
                        padding: '16px',
                        background: '#f8f8f8',
                        border: '1px solid #ddd'
                    }}>
                        <div style={{ fontSize: '10px', color: '#000', marginBottom: '12px', fontWeight: 'bold' }}>LAYER {idx + 1}</div>
                        
                        <label style={{ ...labelStyle, marginBottom: '4px' }}>Neurons</label>
                        <input
                            type="number" min="1" max="64" value={layer.size}
                            onChange={(e) => updateLayer(idx, 'size', parseInt(e.target.value))}
                            style={{ ...inputStyle, marginBottom: '12px' }}
                        />

                        <label style={{ ...labelStyle, marginBottom: '4px' }}>Activation</label>
                        <select
                            value={layer.activation}
                            onChange={(e) => updateLayer(idx, 'activation', e.target.value)}
                            style={{ ...inputStyle, marginBottom: 0 }}
                        >
                            <option value="ReLU">ReLU</option>
                            <option value="Sigmoid">Sigmoid</option>
                            <option value="Tanh">Tanh</option>
                            <option value="LeakyReLU">LeakyReLU</option>
                            <option value="None">None</option>
                        </select>
                    </div>
                ))}
            </div>

            <div style={{ marginBottom: '24px' }}>
                <label style={labelStyle}>Hyperparameters</label>
                <div style={{ padding: '16px', background: '#f8f8f8', border: '1px solid #ddd' }}>
                    <label style={{ ...labelStyle, marginBottom: '4px' }}>Optimizer</label>
                    <select
                        value={optimizer}
                        onChange={(e) => setOptimizer(e.target.value)}
                        style={{ ...inputStyle, marginBottom: '12px' }}
                    >
                        <option value="Adam">Adam</option>
                        <option value="SGD">SGD</option>
                        <option value="RMSprop">RMSprop</option>
                    </select>

                    <label style={{ ...labelStyle, marginBottom: '4px' }}>Loss Function</label>
                    <select
                        value={lossFunction}
                        onChange={(e) => setLossFunction(e.target.value)}
                        style={{ ...inputStyle, marginBottom: '12px' }}
                    >
                        <option value="CrossEntropyLoss">CrossEntropyLoss</option>
                        <option value="MSELoss">MSELoss</option>
                    </select>

                    <label style={{ ...labelStyle, marginBottom: '4px' }}>Learning Rate</label>
                    <input
                        type="number" step="0.0001" min="0.0001" max="0.1" value={learningRate}
                        onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                        style={{ ...inputStyle, marginBottom: '12px' }}
                    />

                    <label style={{ ...labelStyle, marginBottom: '4px' }}>Epochs</label>
                    <input
                        type="number" min="1" max="100" value={epochs}
                        onChange={(e) => setEpochs(parseInt(e.target.value))}
                        style={{ ...inputStyle, marginBottom: 0 }}
                    />
                </div>
            </div>

        </div>
    );
}

// ============================================================================
// NEURON-LEVEL GRAPH VISUALIZATION
// ============================================================================

function AutoFit({ nodes }) {
    const { fitView } = useReactFlow();
    useEffect(() => {
        // Debounce or slightly delay fitView to allow ReactFlow to finish layout calculation
        const timer = setTimeout(() => {
            fitView({ duration: 400, padding: 0.1 });
        }, 50);
        return () => clearTimeout(timer);
    }, [nodes.length, fitView]);
    return null;
}

function NeuronLevelGraph({ config, selectedNode, onNodeClick, neuronActivations, animatingLayer, isTraining }) {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    useEffect(() => {
        // Deep compare or check for significant change to avoid resets.
        if (isTraining) return;

        console.log('NeuronLevelGraph main structure useEffect triggered, config:', !!config);
        if (!config) {
            setNodes([]);
            setEdges([]);
            return;
        }

        const maxInputNeurons = 20;
        const inputNeurons = Math.min(config.inputSize || 784, maxInputNeurons);

        const allLayers = [
            { size: inputNeurons, name: 'Input', activation: 'None', type: 'input' },
            ...(config.hiddenLayers || []).map((l, i) => ({ ...l, name: `Layer ${i + 1}`, type: 'hidden' })),
            { size: config.outputSize || 10, name: 'Output', activation: 'None', type: 'output' }
        ];

        const newNodes = [];
        const newEdges = [];
        const layerSpacing = 350;
        const neuronSpacing = 54;
        const startX = 100;

        allLayers.forEach((layer, layerIdx) => {
            const x = startX + layerIdx * layerSpacing;
            const layerHeight = (layer.size - 1) * neuronSpacing;
            const startY = 400 - layerHeight / 2;

            for (let neuronIdx = 0; neuronIdx < layer.size; neuronIdx++) {
                const y = startY + neuronIdx * neuronSpacing;
                const nodeId = `L${layerIdx}N${neuronIdx}`;
                const activationValue = 0;

                newNodes.push({
                    id: nodeId,
                    type: 'neuron',
                    position: { x, y },
                    selectable: true,
                    data: {
                        layerIndex: layerIdx,
                        neuronIndex: neuronIdx,
                        layerName: layer.name,
                        activation: layer.activation,
                        layerType: layer.type,
                        activationValue: activationValue,
                        dataset: config?.dataset
                    },
                    selected: selectedNode?.id === nodeId
                });

                if (layerIdx < allLayers.length - 1) {
                    const nextLayer = allLayers[layerIdx + 1];

                    for (let nextNeuronIdx = 0; nextNeuronIdx < nextLayer.size; nextNeuronIdx++) {
                        const targetId = `L${layerIdx + 1}N${nextNeuronIdx}`;
                        const edgeId = `${nodeId}-${targetId}`;

                        newEdges.push({
                            id: edgeId,
                            source: nodeId,
                            target: targetId,
                            type: 'straight',
                            animated: false,
                            style: {
                                stroke: '#bbb',
                                strokeWidth: 1,
                                opacity: 0.3,
                            }
                        });
                    }
                }
            }

            // Layer Labels
            newNodes.push({
                id: `layer-label-${layerIdx}`,
                type: 'default',
                selectable: false,
                draggable: false,
                position: { x: x - 50, y: startY - 60 },
                data: { label: layer.name.toUpperCase() },
                style: {
                    background: '#000',
                    color: '#fff',
                    border: 'none',
                    fontSize: '10px',
                    fontWeight: 'bold',
                    width: 140,
                    textAlign: 'center',
                    fontFamily: 'monospace',
                    padding: '4px'
                }
            });
        });

        setNodes(newNodes);
        setEdges(newEdges);
    }, [config, isTraining]); // Removed onNodeClick from here to prevent structure reset on handler change

    // Update activations + selection without rebuilding the full graph.
    useEffect(() => {
        setNodes((prev) => prev.map((n) => {
            if (n.type !== 'neuron') return n;

            const layerIdx = n?.data?.layerIndex ?? 0;
            const neuronIdx = n?.data?.neuronIndex ?? 0;
            const activationValue = neuronActivations?.[layerIdx]?.[neuronIdx] || 0;
            const selected = selectedNode?.id === n.id;

            // Keep the rest of node.data stable (notably onSelect).
            return {
                ...n,
                selected,
                data: {
                    ...n.data,
                    activationValue,
                },
            };
        }));
        // animatingLayer no longer drives per-edge animation to keep perf stable.
    }, [neuronActivations, selectedNode?.id, animatingLayer, setNodes]);

    return (
        <div style={{ width: '100%', height: '100%', background: '#fff' }}>
            <ReactFlowProvider>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onPaneClick={(event) => {
                        console.log('ReactFlow PANE clicked (background)');
                    }}
                    onNodeClick={(event, node) => {
                        console.log('ReactFlow onNodeClick FIRED');
                        if (node.type === 'neuron') {
                            onNodeClick(node);
                        }
                    }}
                    nodeTypes={nodeTypes}
                    nodesDraggable={false}
                    nodesConnectable={false}
                    elementsSelectable={true}
                    selectNodesOnDrag={false}
                    panOnDrag={[1, 2]}
                    panOnScroll={true}
                    zoomOnScroll={true}
                    zoomOnDoubleClick={false}
                    fitView
                    minZoom={0.01}
                    maxZoom={2}
                    defaultViewport={{ x: 0, y: 0, zoom: 0.6 }}
                >
                    <Background color="#f0f0f0" gap={30} />
                    <Controls showInteractive={false} style={{ boxShadow: 'none', border: '1px solid #000' }} />
                    <Panel position="top-right" style={{
                        background: '#fff',
                        padding: '8px 12px',
                        border: '2px solid #000',
                        fontSize: '11px',
                        fontWeight: 'bold',
                        fontFamily: 'monospace'
                    }}>
                        CORE_OS // V_NODES: {nodes.filter(n => n.type === 'neuron').length}
                    </Panel>
                    <AutoFit nodes={nodes} />
                </ReactFlow>
            </ReactFlowProvider>
        </div>
    );
}

// ============================================================================
// PREDICTION DISPLAY COMPONENT
// ============================================================================

function PredictionDisplay({ prediction, dataset }) {
    const ds = (dataset || 'mnist').toLowerCase();
    const cifar10Names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    const labelFor = (cls) => {
        if (ds === 'cifar10') return cifar10Names[cls] ?? `class_${cls}`;
        return String(cls);
    };

    if (!prediction) {
        return (
            <div style={{
                padding: '24px',
                color: '#999',
                fontSize: '11px',
                textAlign: 'center',
                fontFamily: 'monospace'
            }}>
                Awaiting input...
            </div>
        );
    }

    return (
        <div style={{
            padding: '20px',
            fontSize: '11px',
            color: '#1a1a1a',
            fontFamily: 'monospace'
        }}>
            <h4 style={{ margin: '0 0 16px 0', fontSize: '13px', color: '#000', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                Prediction
            </h4>

            <div style={{
                textAlign: 'center',
                marginBottom: '20px',
                padding: '20px',
                background: '#f8f8f8',
                border: '1px solid #000'
            }}>
                <div style={{ fontSize: '34px', fontWeight: 'bold', color: '#000', marginBottom: '8px', textTransform: 'uppercase' }}>
                    {labelFor(prediction.prediction)}
                </div>
                <div style={{ fontSize: '12px', color: '#666' }}>
                    {(prediction.confidence * 100).toFixed(1)}% CONFIDENCE
                </div>
            </div>

            <div>
                <div style={{ color: '#000', fontSize: '10px', marginBottom: '8px', fontWeight: 'bold', textTransform: 'uppercase' }}>
                    Top Rankings
                </div>
                {Array.isArray(prediction.top3) && prediction.top3.map((item, idx) => (
                    <div key={idx} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '8px',
                        background: idx === 0 ? '#fff' : 'transparent',
                        marginBottom: '4px',
                        border: idx === 0 ? '1px solid #000' : '1px solid #eee'
                    }}>
                        <span style={{ fontWeight: idx === 0 ? 'bold' : 'normal' }}>{ds === 'cifar10' ? 'CLASS' : 'DIGIT'} {labelFor(item.class)}</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <div style={{
                                width: '80px',
                                height: '4px',
                                background: '#eee',
                                overflow: 'hidden'
                            }}>
                                <div style={{
                                    width: `${item.confidence * 100}%`,
                                    height: '100%',
                                    background: '#000',
                                    transition: 'width 0.3s'
                                }} />
                            </div>
                            <span style={{ fontSize: '10px', color: '#666', minWidth: '40px', textAlign: 'right' }}>
                                {(item.confidence * 100).toFixed(1)}%
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ============================================================================
// TRAINING SAMPLE PREVIEW
// ============================================================================

function TrainingSamplePreview({ sample }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;
        
        // Use a very light gray background instead of pure white so empty images are visible
        ctx.fillStyle = '#f9f9f9';
        ctx.fillRect(0, 0, w, h);

        if (!sample || !Array.isArray(sample.pixels) || sample.pixels.length === 0) {
            return;
        }

        const len = sample.pixels.length;

        // MNIST (784)
        if (len === 784) {
            const cell = w / 28;
            ctx.fillStyle = '#000';
            for (let y = 0; y < 28; y++) {
                for (let x = 0; x < 28; x++) {
                    const v = sample.pixels[y * 28 + x];
                    if (v > 0.1) { // Lower threshold slightly for visibility
                        ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
                    }
                }
            }
            return;
        }

        // CIFAR-10 (3072) or any 3-channel image
        if (len === 3072 || (len > 0 && len % 3 === 0)) {
            const size = Math.sqrt(len / 3);
            if (Number.isInteger(size)) {
                const plane = size * size;
                const cell = w / size;
                for (let y = 0; y < size; y++) {
                    for (let x = 0; x < size; x++) {
                        const idx = y * size + x;
                        const r = sample.pixels[idx] || 0;
                        const g = sample.pixels[plane + idx] || 0;
                        const b = sample.pixels[2 * plane + idx] || 0;
                        
                        const rr = Math.floor(r * 255.99);
                        const gg = Math.floor(g * 255.99);
                        const bb = Math.floor(b * 255.99);
                        
                        ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
                        // Use a tiny overlap to avoid "white grid" artifacts from subpixel rendering
                        ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
                    }
                }
            }
        }
    }, [sample]);

    if (!sample) return null;

    const cifar10Names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    const labelFor = (cls) => {
        // Detect if it's likely CIFAR by pixel count (3072)
        if (sample.pixels?.length === 3072) return cifar10Names[cls] || cls;
        return cls;
    };

    return (
        <div style={{ padding: '16px 24px', borderBottom: '1px solid #eee' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '10px' }}>
                <div style={{ fontSize: '11px', fontWeight: 'bold', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                    Train Sample
                </div>
                <div style={{ fontSize: '10px', color: '#666' }}>
                    y={labelFor(sample.target)} pred={labelFor(sample.prediction)}
                </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'center' }}>
                <canvas
                    ref={canvasRef}
                    width={140}
                    height={140}
                    style={{ border: '1px solid #000', imageRendering: 'pixelated' }}
                />
            </div>
        </div>
    );
}

// ============================================================================
// NEURON INSPECTOR PANEL
// ============================================================================

function NeuronInspector({ node, isCompiled }) {
    console.log('NeuronInspector render - node:', node?.id, 'data:', node?.data);
    
    if (!node || node.type !== 'neuron') {
        return (
            <div style={{
                padding: '24px',
                color: '#999',
                fontSize: '11px',
                textAlign: 'center',
                fontFamily: 'monospace',
                background: '#f8f8f8',
                border: '2px solid #e0e0e0'
            }}>
                Click a neuron to inspect
            </div>
        );
    }

    const { data } = node;
    const cifar10Names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    const isCifar = data.dataset === 'cifar10';
    const displayLabel = data.layerType === 'output' && isCifar 
        ? (cifar10Names[data.neuronIndex] || data.neuronIndex)
        : data.neuronIndex;

    return (
        <div style={{
            padding: '20px',
            fontSize: '11px',
            color: '#1a1a1a',
            fontFamily: 'monospace'
        }}>
            <h4 style={{ margin: '0 0 16px 0', fontSize: '13px', color: '#000', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                {data.layerType === 'output' && isCifar ? `${displayLabel.toUpperCase()}` : `NEURON [${displayLabel}]`}
            </h4>

            <div style={{ marginBottom: '12px' }}>
                <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>Layer</div>
                <div style={{ fontSize: '12px' }}>{data.layerName} ({data.layerType})</div>
            </div>

            {data.activation !== 'None' && (
                <div style={{ marginBottom: '12px' }}>
                    <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>Function</div>
                    <div style={{ fontSize: '12px' }}>{data.activation}</div>
                </div>
            )}

            <div style={{ marginBottom: '12px' }}>
                <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>Activation Value</div>
                <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                    {(data.activationValue ?? 0).toFixed(6)}
                </div>
                <div style={{
                    marginTop: '8px',
                    height: '4px',
                    background: '#eee',
                    overflow: 'hidden'
                }}>
                    <div style={{
                        width: `${(data.activationValue ?? 0) * 100}%`,
                        height: '100%',
                        background: '#000',
                        transition: 'width 0.3s'
                    }} />
                </div>
            </div>

            {data.layerStats && (
                <div style={{ marginTop: '18px', borderTop: '1px solid #eee', paddingTop: '14px' }}>
                    <div style={{ color: '#999', fontSize: '9px', marginBottom: '6px', textTransform: 'uppercase' }}>Layer Telemetry</div>

                    {data.layerStats.meta && (
                        <div style={{ marginBottom: '10px', fontSize: '10px', color: '#666' }}>
                            epoch {data.layerStats.meta.epoch}, batch {data.layerStats.meta.batch}
                        </div>
                    )}

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                        <div>
                            <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>W mean / std</div>
                            <div style={{ fontSize: '12px', fontWeight: 'bold' }}>
                                {Number(data.layerStats.weight?.mean ?? 0).toFixed(6)} / {Number(data.layerStats.weight?.std ?? 0).toFixed(6)}
                            </div>
                        </div>
                        <div>
                            <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>B mean / std</div>
                            <div style={{ fontSize: '12px', fontWeight: 'bold' }}>
                                {Number(data.layerStats.bias?.mean ?? 0).toFixed(6)} / {Number(data.layerStats.bias?.std ?? 0).toFixed(6)}
                            </div>
                        </div>
                        <div>
                            <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>∥∇W∥</div>
                            <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{Number(data.layerStats.grad?.weightNorm ?? 0).toFixed(6)}</div>
                        </div>
                        <div>
                            <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>∥∇b∥</div>
                            <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{Number(data.layerStats.grad?.biasNorm ?? 0).toFixed(6)}</div>
                        </div>
                    </div>
                </div>
            )}

            {!data.inspection && data.layerType === 'input' && (
                <div style={{ marginTop: '18px', borderTop: '1px solid #eee', paddingTop: '14px' }}>
                    <div style={{ color: '#999', fontSize: '10px', fontStyle: 'italic' }}>
                        Input neurons do not have weights or biases.
                    </div>
                </div>
            )}

            {!data.inspection && data.layerType !== 'input' && (
                <div style={{ marginTop: '18px', borderTop: '1px solid #eee', paddingTop: '14px' }}>
                    <div style={{ color: '#999', fontSize: '10px', fontStyle: 'italic' }}>
                        {isCompiled 
                            ? 'Fetching neuron parameters...' 
                            : 'Compile the model to view weights and biases'}
                    </div>
                </div>
            )}

            {data.inspection && (
                <div style={{ marginTop: '18px', borderTop: '1px solid #eee', paddingTop: '14px' }}>
                    <div style={{ color: '#999', fontSize: '9px', marginBottom: '6px', textTransform: 'uppercase' }}>Weights / Bias</div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', marginBottom: '10px' }}>
                        <div style={{ flex: 1 }}>
                            <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>Bias</div>
                            <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{Number(data.inspection.bias).toFixed(6)}</div>
                        </div>
                        {data.inspection.deltaBias !== undefined && data.inspection.deltaBias !== null && (
                            <div style={{ flex: 1 }}>
                                <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>Δ Bias (last tick)</div>
                                <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{Number(data.inspection.deltaBias).toFixed(6)}</div>
                            </div>
                        )}
                    </div>

                    {Array.isArray(data.inspection.weightRow) && (
                        <div>
                            <div style={{ color: '#999', fontSize: '9px', marginBottom: '6px', textTransform: 'uppercase' }}>Weight Row</div>
                            <pre style={{
                                margin: 0,
                                maxHeight: '220px',
                                overflow: 'auto',
                                background: '#f8f8f8',
                                border: '1px solid #eee',
                                padding: '10px',
                                fontSize: '10px'
                            }}>{JSON.stringify(data.inspection.weightRow.slice(0, 64), null, 0)}{data.inspection.weightRow.length > 64 ? '\n…' : ''}</pre>

                            {data.inspection.deltaWeightRowNorm !== undefined && data.inspection.deltaWeightRowNorm !== null && (
                                <div style={{ marginTop: '10px' }}>
                                    <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px', textTransform: 'uppercase' }}>Δ Weight Row Norm (last tick)</div>
                                    <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{Number(data.inspection.deltaWeightRowNorm).toFixed(6)}</div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

// ============================================================================
// MAIN APP
// ============================================================================

function App() {
    const [config, setConfig] = useState(() => ({
        dataset: 'mnist',
        inputSize: 784,
        hiddenLayers: [
            { size: 16, activation: 'ReLU', dropout: 0 },
            { size: 16, activation: 'ReLU', dropout: 0 }
        ],
        outputSize: 10,
        lossFunction: 'CrossEntropyLoss',
        optimizer: 'Adam',
        learningRate: 0.001,
        epochs: 10
    }));
    const [code, setCode] = useState('');
    const [selectedNode, setSelectedNode] = useState(null);
    const [neuronInspection, setNeuronInspection] = useState(null);
    const [lastParamUpdate, setLastParamUpdate] = useState(null);
    const [trainingSignals, setTrainingSignals] = useState({});
    const [trainingSample, setTrainingSample] = useState(null);
    const [neuronActivations, setNeuronActivations] = useState({});
    const [animatingLayer, setAnimatingLayer] = useState(null);
    const [prediction, setPrediction] = useState(null);

    const [socket, setSocket] = useState(null);
    const [status, setStatus] = useState('Connecting...');
    const [isCompiled, setIsCompiled] = useState(false);
    const [isTraining, setIsTraining] = useState(false);
    const [isModelTrained, setIsModelTrained] = useState(false);
    const [metrics, setMetrics] = useState({
        loss: null,
        accuracy: null,
        epoch: 0,
        totalEpochs: 10,
        batch: 0
    });

    const [learningRate, setLearningRate] = useState(0.001);
    const [batchSize, setBatchSize] = useState(64);
    const [epochs, setEpochs] = useState(10);

    const socketRef = useRef(null);
    const compileTimerRef = useRef(null);
    const lastRequestedCompileIdRef = useRef(null);
    const lastRequestedSigRef = useRef(null);
    const compileSeqRef = useRef(1);
    const compiledSigRef = useRef(null);
    const [isCompilePending, setIsCompilePending] = useState(false);

    const computeCompileSig = useCallback((cfg) => {
        if (!cfg) return null;
        const compileSigObj = {
            dataset: cfg.dataset,
            inputSize: cfg.inputSize,
            outputSize: cfg.outputSize,
            hiddenLayers: (cfg.hiddenLayers || []).map((l) => ({
                size: l.size,
                activation: l.activation,
                dropout: l.dropout
            }))
        };
        return JSON.stringify(compileSigObj);
    }, []);

    // Debug: Track selectedNode changes
    useEffect(() => {
        console.log('🔴 selectedNode STATE CHANGED:', selectedNode?.id, selectedNode);
    }, [selectedNode]);

    // Debug: Track neuronInspection changes
    useEffect(() => {
        console.log('🔵 neuronInspection STATE CHANGED:', neuronInspection);
    }, [neuronInspection]);

    const mapBackendActivationsToVisual = useCallback((backendActivations, inputPixels) => {
        if (!backendActivations || typeof backendActivations !== 'object') return {};

        const backendLayersCnt = Object.keys(backendActivations).length;
        const newActivations = {};

        // Layer 0: Input
        if (Array.isArray(inputPixels) && inputPixels.length > 0) {
            newActivations[0] = {};
            const maxInputNeurons = 20;
            for (let i = 0; i < maxInputNeurons; i++) {
                const startIdx = Math.floor(i * (inputPixels.length / maxInputNeurons));
                const endIdx = Math.floor((i + 1) * (inputPixels.length / maxInputNeurons));
                const chunk = inputPixels.slice(startIdx, endIdx);
                const avg = chunk.reduce((a, b) => a + b, 0) / chunk.length;
                newActivations[0][i] = avg;
            }
        }

        for (let i = 0; i < backendLayersCnt; i++) {
            const key = `layer_${i}`;
            const targetVisualLayer = i + 1;
            const vals = backendActivations[key];
            if (Array.isArray(vals) && vals.length > 0) {
                newActivations[targetVisualLayer] = {};
                let max = 0.001;
                for (let k = 0; k < vals.length; k++) {
                    const a = Math.abs(vals[k]);
                    if (a > max) max = a;
                }
                for (let j = 0; j < vals.length; j++) {
                    newActivations[targetVisualLayer][j] = Math.max(0, vals[j] / max);
                }
            }
        }

        return newActivations;
    }, []);

    // WebSocket connection
    useEffect(() => {
        const ws = io('http://localhost:8000', {
            transports: ['polling', 'websocket'],
            upgrade: true,
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 5
        });

        ws.on('connect', () => {
            setStatus('Connected');
        });

        ws.on('disconnect', () => {
            setStatus('Disconnected');
        });

        ws.on('model_compiled', (data) => {
            const compileId = data?.compileId;
            if (compileId && lastRequestedCompileIdRef.current && compileId !== lastRequestedCompileIdRef.current) {
                // Ignore out-of-order compiles.
                return;
            }

            // Mark the last requested architecture as compiled.
            compiledSigRef.current = lastRequestedSigRef.current;
            setIsCompilePending(false);
            setIsCompiled(true);
            setIsModelTrained(false); // Reset training flag for new model
            console.log('✓ Model compiled successfully:', data.model_info);
        });

        ws.on('training_started', () => {
            setIsTraining(true);
            setIsModelTrained(true);
        });

        ws.on('training_stopped', () => {
            setIsTraining(false);
        });

        ws.on('error', (data) => {
            console.error('Backend Error:', data.message);
            setIsCompilePending(false); // Ensure we don't get stuck in pending state
            alert('Error: ' + data.message);
        });

        ws.on('metrics_update', (data) => {
            setMetrics({
                loss: data.loss,
                accuracy: data.accuracy,
                epoch: data.epoch,
                totalEpochs: data.total_epochs,
                batch: data.batch
            });
        });

        ws.on('param_update', (data) => {
            setLastParamUpdate(data);

            // Visualize training by mapping per-neuron bias magnitudes onto the node grid.
            // Backend layerIndex 0 corresponds to visual layer 1 (first hidden/output).
            if (data && Array.isArray(data.layers)) {
                const nextSignals = { 0: {} };

                for (const layer of data.layers) {
                    const bias = layer?.perNeuron?.bias;
                    if (!Array.isArray(bias) || bias.length === 0) continue;

                    const absVals = bias.map((v) => Math.abs(v));
                    const max = Math.max(...absVals, 1e-6);
                    const visualLayerIndex = (layer.layerIndex ?? 0) + 1;

                    nextSignals[visualLayerIndex] = {};
                    for (let i = 0; i < absVals.length; i++) {
                        nextSignals[visualLayerIndex][i] = absVals[i] / max;
                    }
                }

                setTrainingSignals(nextSignals);
            }
        });

        ws.on('training_sample', (data) => {
            setTrainingSample(data);

            // Prefer true activations for training visualization.
            const visualActs = mapBackendActivationsToVisual(data?.activations, data?.pixels);
            if (visualActs && Object.keys(visualActs).length > 0) {
                setTrainingSignals(visualActs);
            }
        });

        ws.on('neuron_inspection', (data) => {
            console.log('Received neuron inspection:', data);
            setNeuronInspection({
                layerIndex: data.layerIndex,
                neuronIndex: data.neuronIndex,
                bias: data.bias,
                weightRow: data.weightRow,
                inFeatures: data.inFeatures,
                outFeatures: data.outFeatures
            });
        });

        ws.on('inspected_neuron_update', (data) => {
            // Streamed training telemetry for the currently inspected neuron.
            setNeuronInspection((prev) => {
                if (!prev) return prev;
                if (prev.layerIndex !== data.layerIndex || prev.neuronIndex !== data.neuronIndex) return prev;
                return {
                    ...prev,
                    bias: data.bias,
                    weightRow: data.weightRow,
                    deltaWeightRowNorm: data.deltaWeightRowNorm,
                    deltaBias: data.deltaBias,
                    biasGrad: data.biasGrad,
                    weightRowGradMeanAbs: data.weightRowGradMeanAbs
                };
            });
        });

        socketRef.current = ws;
        setSocket(ws);

        return () => ws.disconnect();
    }, []);

    const handleConfigChange = useCallback((newConfig) => {
        if (isTraining) return; // Prevent config updates while training

        setConfig(newConfig);
        if (newConfig.learningRate !== undefined) setLearningRate(newConfig.learningRate);
        if (newConfig.epochs !== undefined) setEpochs(newConfig.epochs);

        // Auto-generate the code view as config changes.
        try {
            setCode(generateCode(newConfig));
        } catch (e) {
            // Keep UI resilient if partial config is temporarily invalid while typing.
            console.warn('generateCode failed:', e);
        }

        const nextSig = computeCompileSig(newConfig);
        if (compiledSigRef.current && nextSig && nextSig !== compiledSigRef.current) {
            setIsCompiled(false);
        }
    }, [isTraining, computeCompileSig]);

    // Auto-compile (debounced) when architecture/dataset changes.
    useEffect(() => {
        if (!config) return;
        if (isTraining) return;
        if (!socketRef.current || !socketRef.current.connected) return;

        const sig = computeCompileSig(config);
        if (!sig) return;
        
        // Skip if already compiled.
        if (sig === compiledSigRef.current && isCompiled) return;

        // Basic validation: ignore if any hidden layer has 0 size (user is typing).
        const hasInvalidLayer = (config.hiddenLayers || []).some(l => !l.size || l.size <= 0);
        if (hasInvalidLayer) return;

        if (compileTimerRef.current) clearTimeout(compileTimerRef.current);
        compileTimerRef.current = setTimeout(() => {
            const compileId = String(compileSeqRef.current++);
            lastRequestedCompileIdRef.current = compileId;
            lastRequestedSigRef.current = sig;
            setIsCompilePending(true);
            socketRef.current.emit('compile_model', { config, compileId });
        }, 350);

        return () => {
            if (compileTimerRef.current) clearTimeout(compileTimerRef.current);
        };
    }, [config?.dataset, config?.inputSize, config?.outputSize, config?.hiddenLayers, isTraining, computeCompileSig, isCompiled]);

    const handleTrain = () => {
        console.log('handleTrain called. Current state:', { isCompiled, isCompilePending, socket: !!socketRef.current });
        if (!socketRef.current || !socketRef.current.connected) {
            alert('Not connected to backend');
            return;
        }

        if (!isCompiled || isCompilePending) {
            alert('Model is still compiling. Please wait a moment.');
            return;
        }

        socketRef.current.emit('start_training', {
            learningRate,
            batchSize,
            epochs,
            dataset: config?.dataset || 'mnist',
            optimizer: config?.optimizer || 'Adam',
            lossFunction: config?.lossFunction || 'CrossEntropyLoss'
        });
    };

    const handleStop = () => {
        if (socketRef.current && socketRef.current.connected) {
            socketRef.current.emit('stop_training', {});
        }
    };

    const handleNodeClick = useCallback((node) => {
        console.log('handleNodeClick CALLED!!!');
        setSelectedNode(node);

        // Map visual layer index (0=input) to backend linear layer index (0=first linear)
        const layerIndex = (node?.data?.layerIndex ?? 0) - 1;
        const neuronIndex = node?.data?.neuronIndex ?? 0;

        if (layerIndex >= 0 && socketRef.current && socketRef.current.connected) {
            socketRef.current.emit('inspect_neuron', {
                layerIndex,
                neuronIndex
            });
        } else {
            setNeuronInspection(null);
        }
    }, [socket]); // socket state is enough to know if we can emit

    const handlePredict = async (result, inputPixels) => {
        if (!result) return;
        setPrediction(result);

        // Animate activation flow
        const activations = result.activations;
        if (!activations || typeof activations !== 'object') {
            console.warn('No activations in prediction result');
            return;
        }
        const newActivations = mapBackendActivationsToVisual(activations, inputPixels);
        const totalVisualLayers = Object.keys(newActivations).length;

        // Sequence animation
        setNeuronActivations({}); // Reset first
        for (let l = 0; l < totalVisualLayers; l++) {
            setAnimatingLayer(l);
            setNeuronActivations(prev => ({
                ...prev,
                [l]: newActivations[l]
            }));
            await new Promise(r => setTimeout(r, 100));
        }

        setAnimatingLayer(null);
        setNeuronActivations(newActivations);
    };

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100vh',
            fontFamily: 'monospace',
            background: '#ffffff',
            color: '#1a1a1a'
        }}>
            {/* Header: Simple Industrial Style */}
            <div style={{
                padding: '16px 24px',
                borderBottom: '2px solid #000',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                background: '#fff'
            }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '16px' }}>
                    <h1 style={{ margin: 0, fontSize: '16px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        Neural_Core IDE [v1.0.4]
                    </h1>
                    <div style={{ fontSize: '10px', color: '#666' }}>
                        STATUS: {status.toUpperCase()} // ENV: PRODUCTION
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '12px' }}>
                    {!isTraining ? (
                        <button
                            onClick={handleTrain}
                            disabled={isCompilePending || status !== 'Connected' || computeCompileSig(config) !== compiledSigRef.current}
                            style={{
                                padding: '8px 16px',
                                background: '#000',
                                border: '1px solid #000',
                                color: '#fff',
                                cursor: 'pointer',
                                fontSize: '11px',
                                fontWeight: 'bold',
                                opacity: (isCompilePending || status !== 'Connected' || computeCompileSig(config) !== compiledSigRef.current) ? 0.4 : 1
                            }}>
                            START_TRAINING
                        </button>
                    ) : (
                        <button
                            onClick={handleStop}
                            style={{
                                padding: '8px 16px',
                                background: '#ff0000',
                                border: '1px solid #ff0000',
                                color: '#fff',
                                cursor: 'pointer',
                                fontSize: '11px',
                                fontWeight: 'bold'
                            }}>
                            STOP_TRAINING
                        </button>
                    )}
                </div>
            </div>

            {/* Main Content Area */}
            <div style={{
                display: 'flex',
                flex: 1,
                overflow: 'hidden'
            }}>
                {/* Column 1: ARCHITECTURE & CODE */}
                <div style={{
                    width: '320px',
                    borderRight: '1px solid #000',
                    display: 'flex',
                    flexDirection: 'column',
                    background: '#fff',
                    overflowY: 'auto'
                }}>
                    <NetworkBuilder onConfigChange={handleConfigChange} />
                    
                    <div style={{ borderTop: '1px solid #000', padding: '20px' }}>
                        <h3 style={{ margin: '0 0 12px 0', fontSize: '11px', fontWeight: 'bold', color: '#666' }}>PYTORCH_DEFINITION</h3>
                        <textarea
                            value={code}
                            onChange={(e) => setCode(e.target.value)}
                            readOnly
                            style={{
                                width: '100%',
                                height: '200px',
                                background: '#f8f8f8',
                                border: '1px solid #ddd',
                                padding: '12px',
                                color: '#444',
                                fontSize: '10px',
                                lineHeight: '1.4',
                                resize: 'none',
                                outline: 'none',
                                fontFamily: 'monospace'
                            }}
                        />
                    </div>
                </div>

                {/* Column 2: VISUALIZATION (Main) */}
                <div style={{
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    background: '#fcfcfc'
                }}>
                    <div style={{ padding: '8px 16px', borderBottom: '1px solid #eee', background: '#fff', fontSize: '10px', color: '#999' }}>
                        GRAPH_VIEWPORT :: REALTIME_ACTIVATION_MESH
                    </div>
                    <div style={{ flex: 1 }}>
                        <NeuronLevelGraph
                            config={config}
                            selectedNode={selectedNode}
                            onNodeClick={handleNodeClick}
                            neuronActivations={isTraining ? (trainingSignals || {}) : (neuronActivations || {})}
                            animatingLayer={animatingLayer}
                            isTraining={isTraining}
                        />
                    </div>
                </div>

                {/* Column 3: INFERENCE & INSPECTOR */}
                <div style={{
                    width: '350px',
                    borderLeft: '1px solid #000',
                    display: 'flex',
                    flexDirection: 'column',
                    background: '#fff',
                    overflowY: 'auto'
                }}>
                    <DrawingCanvas
                        onPredict={handlePredict}
                        onClear={() => setPrediction(null)}
                        isEnabled={computeCompileSig(config) === compiledSigRef.current && !isTraining && !isCompilePending && isModelTrained}
                        dataset={config?.dataset || 'mnist'}
                    />

                    {isTraining && trainingSample && (
                        <TrainingSamplePreview sample={trainingSample} />
                    )}
                    
                    <PredictionDisplay prediction={prediction} dataset={config?.dataset || 'mnist'} />
                    
                    <div style={{ borderTop: '1px solid #eee', marginTop: 'auto' }}>
                        <NeuronInspector
                            node={selectedNode ? {
                                ...selectedNode,
                                data: {
                                    ...selectedNode.data,
                                    inspection: neuronInspection,
                                    layerStats: (() => {
                                        const backendLayerIndex = (selectedNode?.data?.layerIndex ?? 0) - 1;
                                        const layer = lastParamUpdate?.layers?.find((l) => l.layerIndex === backendLayerIndex);
                                        if (!layer) return null;
                                        return {
                                            ...layer,
                                            meta: { epoch: lastParamUpdate.epoch, batch: lastParamUpdate.batch }
                                        };
                                    })()
                                }
                            } : null}
                            isCompiled={isCompiled}
                        />
                    </div>
                </div>
            </div>

            {/* Footer: Telemetry */}
            <div style={{
                padding: '12px 24px',
                borderTop: '2px solid #000',
                background: '#fff',
                display: 'flex',
                gap: '32px',
                fontSize: '11px',
                color: '#000',
                fontWeight: 'bold'
            }}>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ color: '#999' }}>EPOCH:</span>
                    <span>{metrics.epoch.toString().padStart(2, '0')} / {metrics.totalEpochs.toString().padStart(2, '0')}</span>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ color: '#999' }}>LOSS:</span>
                    <span>{metrics.loss !== null ? metrics.loss.toFixed(6) : 'N/A'}</span>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ color: '#999' }}>ACCURACY:</span>
                    <span>{metrics.accuracy !== null ? (metrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}</span>
                </div>
                
                {isTraining && (
                    <div style={{ marginLeft: 'auto', color: '#ff0000', animation: 'pulse 1s infinite' }}>
                        ● TRAINING_IN_PROGRESS...
                    </div>
                )}
                
                {!isTraining && prediction && (
                    <div style={{ marginLeft: 'auto', background: '#000', color: '#fff', padding: '0 8px' }}>
                        LAST_PRED: {prediction.prediction} [CONF: {(prediction.confidence * 100).toFixed(1)}%]
                    </div>
                )}
            </div>
            
            <style>{`
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.3; }
                    100% { opacity: 1; }
                }
                ::-webkit-scrollbar {
                    width: 6px;
                }
                ::-webkit-scrollbar-track {
                    background: #f1f1f1;
                }
                ::-webkit-scrollbar-thumb {
                    background: #ccc;
                }
                ::-webkit-scrollbar-thumb:hover {
                    background: #000;
                }
            `}</style>
        </div>
    );
}

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);
