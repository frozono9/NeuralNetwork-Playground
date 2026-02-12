# Visual Deep Learning IDE

**A professional development environment where visual graph and PyTorch code are bidirectionally synchronized.**

Build neural networks visually → auto-generate PyTorch code → edit code → graph updates → train with full internal visualization → debug tensors interactively.

**Think: VSCode × React Flow × PyTorch Debugger**

---

## Features

### 🔥 Bidirectional Synchronization
- **Graph → Code**: Drag layers in visual editor → PyTorch code generates automatically
- **Code → Graph**: Edit Python code → visual graph updates in real-time
- AST-based parsing for accurate code-to-graph reconstruction

### 🎨 Visual Model Builder
- Drag-and-drop layer nodes (Linear, Conv2d, BatchNorm, ReLU, Dropout, etc.)
- Connect layers visually to define model architecture
- Real-time shape inference and validation

### 💻 Monaco Code Editor
- VSCode-like Python editing experience
- Syntax highlighting and autocomplete
- Real-time synchronization with graph

### 🔬 Neural Network Debugger
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

### ⚙️ Model Analysis
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
