# Neural_Core IDE [v1.0.4]

**A visual development environment for real-time neural network design, training, and tensor inspection.**

Neural_Core is an interactive playground to help bridge high-level architectural design and low-level PyTorch implementation. It allows you to build neural networks visually, watch them train in real-time, and inspect individual neurons to understand how weights, biases, and gradients evolve during the learning process.

![Neural_Core Interface](screenshot.png)

[![Neural Network Playground Demo](https://img.youtube.com/vi/_JZBxz40_H8/0.jpg)](https://youtu._JZBxz40_H8)
*Watch the Demo: [https://youtu.be/_JZBxz40_H8](https://youtu.be/_JZBxz40_H8)*

---

## What it does

### 1. Visual Architecture Design
- **Interactive Graph**: Build networks by selecting datasets (MNIST/CIFAR-10) and defining hidden layer depths (up to 10 layers).
- **Auto-Reframing**: The viewport automatically adjusts as you scale your architecture to keep the entire network in focus.
- **Bidirectional Sync**: Visual changes instantly update the generated PyTorch code preview.

### 2. Real-Time Training Visualization
- **Activation Mesh**: Watch neurons light up in real-time as the model processes training samples.
- **Dynamic Telemetry**: Visual feedback on bias magnitudes and activation values directly on the nodes.
- **Metrics Tracking**: Live tracking of Loss and Accuracy across epochs.

### 3. Deep Tensor Inspection
- **Neuron-Level Debugging**: Click any neuron to open the **Inspector Panel**.
- **Live Parameter Stream**: View real-time weights, biases, and gradient distributions for the specific neuron you are inspecting.
- **Interference Testing**: Draw on the integrated canvas to test model predictions instantly and observe the internal firing patterns.

---

## Features

### Architecture & Code
- **Graph → Code**: Visual configuration generates production-ready PyTorch code.
- **Industrial Styling**: A clean, high-contrast monospace UI designed for technical clarity.

### Training & Inference
- **Socket.IO Backend**: Low-latency communication with a FastAPI/PyTorch backend.
- **Drawing Canvas**: Real-time digit/image classification testing.

---

## Future Steps

- [ ] **CNN Support**: Adding Convolutional (Conv2d) and Pooling layers to the visual builder.
- [ ] **Custom Dataset Upload**: Allow users to drag-and-drop their own CSV/Image datasets for training.
- [ ] **Weight Export**: Download trained state-dicts directly from the IDE.
- [ ] **Cloud Training**: Integration with cloud GPUs for more intensive training sessions.
- [ ] **Expanded Ops**: Support for LayerNorm, Residual connections, and Attention heads.

---

## Setup & Installation

### Backend (Python)
1. Navigate to `backend/`
2. Create environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Run server: `python main.py`

### Frontend (React)
1. Install dependencies: `npm install`
2. Launch dev server: `npm run dev`

---

## Technology Stack

- **Frontend**: React, React Flow (Visualization), Socket.IO-client, Vite
- **Backend**: Python 3.14, FastAPI (Web Server), PyTorch (Deep Learning), Socket.IO (Real-time Stream)

- Real-time shape inference and validation

### Monaco Code Editor
- VSCode-like Python editing experience
- Syntax highlighting and autocomplete
- Real-time synchronization with graph

### Neural Network Debugger
- **Live Training Visualization**:
  - Edge thickness = gradient magnitude
  - Edge color = gradient sign/direction
  - Node pulse animations during updates
- **Tensor Introspection**:
  - Activation histograms
  - Gradient distributions
  - Weight statistics
  - Feature maps (CNNs)
  - Dead neuron detection

### Model Analysis
- Parameter count
- FLOPs calculation
- Memory estimation
- Real-time metrics (loss, accuracy)

---

## Architecture

**Frontend**: React + React Flow + Monaco Editor + WebGL  
**Backend**: Python + FastAPI + PyTorch + WebSocket  
**Communication**: WebSocket streaming (statistics + sampled slices)

---

## Setup

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- (Optional) CUDA for GPU training

### Installation

**1. Install Frontend Dependencies**
```bash
npm install
```

**2. Install Backend Dependencies**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Application

**Terminal 1 - Backend Server:**
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

Backend will run on `http://localhost:8000`

**Terminal 2 - Frontend Dev Server:**
```bash
npm run dev
```

Frontend will run on `http://localhost:5173`

Open your browser to `http://localhost:5173`

---

## Usage

### Building a Model

1. **Visual Approach**:
   - Drag layer nodes from the palette
   - Connect them to define architecture
   - Watch PyTorch code generate automatically

2. **Code Approach**:
   - Write PyTorch code in the Monaco editor
   - Watch the visual graph update automatically

### Training

1. Click **"Compile"** to validate your model
2. Configure hyperparameters (learning rate, batch size, epochs)
3. Select dataset (MNIST, CIFAR-10)
4. Click **"Train"** to start training
5. Watch live visualizations:
   - Gradient flow through edges
   - Loss/accuracy charts
   - Real-time metrics

### Debugging

1. Click any node during training to inspect:
   - Activation distributions
   - Gradient statistics
   - Weight norms
2. Pause training to examine specific batches
3. Use step-by-step execution for detailed analysis

---

## Project Structure

```
.
├── src/                      # Frontend source
│   ├── components/
│   │   ├── Layout/          # IDE layout components
│   │   ├── GraphEditor/     # Visual graph editor
│   │   ├── CodeEditor/      # Monaco editor integration
│   │   ├── Inspector/       # Tensor inspection panel
│   │   └── Training/        # Training controls
│   ├── services/
│   │   ├── sync/            # Bidirectional sync engine
│   │   └── websocket.js     # WebSocket manager
│   ├── store/               # State management (Zustand)
│   └── styles/              # CSS design system
│
├── backend/
│   ├── compiler/            # Graph → PyTorch compiler
│   ├── parser/              # Code → Graph parser (AST)
│   ├── training/            # Training engine with hooks
│   └── main.py              # FastAPI server
│
└── README.md
```

---

## Technology Stack

### Frontend
- **React** - UI framework
- **React Flow** - Node-based graph editor
- **Monaco Editor** - VSCode-like code editor
- **Zustand** - State management
- **D3.js** - Data visualization
- **Socket.IO** - WebSocket client

### Backend
- **PyTorch** - Deep learning framework
- **FastAPI** - Web framework
- **Python AST** - Code parsing
- **THOP** - FLOPs calculation

---

## Development

### Frontend Development
```bash
npm run dev    # Start dev server
npm run build  # Build for production
npm test       # Run tests
```

### Backend Development
```bash
python main.py              # Start server
pytest tests/               # Run tests
```

---

## Roadmap

- [x] Bidirectional graph ↔ code sync
- [x] Visual model builder
- [x] Monaco editor integration
- [x] Training engine with hooks
- [x] Real-time gradient visualization
- [ ] Loss landscape visualization (3D)
- [ ] Attention heatmaps
- [ ] Expandable composite blocks (Transformer, ResNet)
- [ ] Custom dataset upload
- [ ] Model export (ONNX, TorchScript)
- [ ] Collaborative editing

---

## License

MIT

---

## Contributing

This is a portfolio project demonstrating:
- Deep learning systems knowledge
- Compiler-like engineering (AST parsing, code generation)
- Real-time streaming architecture
- Professional IDE design

Contributions welcome!
