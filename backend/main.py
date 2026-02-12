"""
Visual Deep Learning IDE - Backend Server  
FastAPI + Socket.IO + PyTorch - SIMPLIFIED VERSION
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
import asyncio
import uvicorn
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create Socket.IO server with proper configuration
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)

# Create FastAPI app
app = FastAPI(title="Visual Deep Learning IDE")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wrap with Socket.IO
socket_app = socketio.ASGIApp(
    sio,
    app,
    socketio_path='socket.io'
)

# Global state
current_model = None
is_training = False
inspected_neuron = None  # {"layerIndex": int, "neuronIndex": int}
current_config = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _dataset_key(name: str) -> str:
    return (name or 'mnist').lower().replace('-', '').replace('_', '')


def _get_dataset_spec(name: str):
    """Central dataset registry.

    Return dict with:
      - key: canonical name
      - expected_pixels: int
      - make_train_dataset(DATA_DIR): torchvision dataset
      - normalize_input(tensor): tensor
      - unnormalize_for_viz(tensor): tensor in ~[0,1]
    """
    key = _dataset_key(name)

    if key == 'cifar10':
        def make_train_dataset(root, datasets, transforms):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            return datasets.CIFAR10(root, train=True, download=True, transform=transform)

        def normalize_input(x: torch.Tensor) -> torch.Tensor:
            return (x - 0.5) / 0.5

        def unnormalize_for_viz(x: torch.Tensor) -> torch.Tensor:
            return (x * 0.5 + 0.5).clamp(0, 1)

        return {
            'key': 'cifar10',
            'expected_pixels': 3072,
            'make_train_dataset': make_train_dataset,
            'normalize_input': normalize_input,
            'unnormalize_for_viz': unnormalize_for_viz,
        }

    # Default: MNIST
    def make_train_dataset(root, datasets, transforms):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return datasets.MNIST(root, train=True, download=True, transform=transform)

    def normalize_input(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.1307) / 0.3081

    def unnormalize_for_viz(x: torch.Tensor) -> torch.Tensor:
        return (x * 0.3081 + 0.1307).clamp(0, 1)

    return {
        'key': 'mnist',
        'expected_pixels': 784,
        'make_train_dataset': make_train_dataset,
        'normalize_input': normalize_input,
        'unnormalize_for_viz': unnormalize_for_viz,
    }


def _activation_from_name(name: str):
    name = (name or '').lower().replace('_', '').replace(' ', '')
    if name == 'relu':
        return torch.relu
    if name == 'sigmoid':
        return torch.sigmoid
    if name == 'tanh':
        return torch.tanh
    if name == 'leakyrelu':
        return lambda x: F.leaky_relu(x, negative_slope=0.01)
    if name == 'none' or name == 'linear':
        return None
    return None


class ConfigurableMLP(nn.Module):
    def __init__(self, input_size: int, hidden_layers: list, output_size: int):
        super().__init__()

        sizes = [int(input_size)]
        for layer in hidden_layers or []:
            sizes.append(int(layer.get('size', 0)))
        sizes.append(int(output_size))

        self.linears = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ])

        self.activations = []
        self.dropouts = nn.ModuleList()

        # One activation/dropout per hidden layer (not for output layer)
        for layer in hidden_layers or []:
            self.activations.append(_activation_from_name(layer.get('activation')))
            p = float(layer.get('dropout', 0) or 0)
            self.dropouts.append(nn.Dropout(p=p) if p > 0 else nn.Identity())

    def forward(self, x, capture_activations: bool = False):
        # Flatten any image-like input.
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        layer_acts = {}

        for i, linear in enumerate(self.linears):
            x = linear(x)

            # For all but last linear, apply configured activation/dropout.
            if i < len(self.linears) - 1:
                act_fn = self.activations[i] if i < len(self.activations) else None
                if act_fn is not None:
                    x = act_fn(x)
                if i < len(self.dropouts):
                    x = self.dropouts[i](x)

            if capture_activations:
                layer_acts[f'layer_{i}'] = x.detach().cpu().numpy().flatten().tolist()

        return (x, layer_acts) if capture_activations else x


def _get_linear_layers(model: nn.Module):
    if isinstance(model, ConfigurableMLP):
        return list(model.linears)
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def _layer_summaries(model: nn.Module):
    summaries = []
    linears = _get_linear_layers(model)
    for idx, layer in enumerate(linears):
        w = layer.weight
        b = layer.bias

        w_grad = layer.weight.grad
        b_grad = layer.bias.grad if layer.bias is not None else None

        # Per-neuron signals (for visualizing nodes while training)
        with torch.no_grad():
            weight_row_norms = w.detach().norm(dim=1).cpu().numpy().tolist()
            bias_values = b.detach().cpu().numpy().tolist() if b is not None else [0.0] * int(w.shape[0])

        summaries.append({
            'layerIndex': idx,
            'outFeatures': int(w.shape[0]),
            'inFeatures': int(w.shape[1]),
            'weight': {
                'mean': float(w.detach().mean().item()),
                'std': float(w.detach().std().item()),
            },
            'bias': {
                'mean': float(b.detach().mean().item()) if b is not None else 0.0,
                'std': float(b.detach().std().item()) if b is not None else 0.0,
            },
            'grad': {
                'weightNorm': float(w_grad.detach().norm().item()) if w_grad is not None else 0.0,
                'biasNorm': float(b_grad.detach().norm().item()) if b_grad is not None else 0.0,
            },
            'perNeuron': {
                'bias': bias_values,
                'weightRowNorm': weight_row_norms
            }
        })
    return summaries


@app.get("/")
async def root():
    return {"message": "Visual Deep Learning IDE Backend", "status": "running"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "socketio": "enabled"}


@sio.event
async def connect(sid, environ):
    print(f"✓ Client connected: {sid}")
    await sio.emit('connected', {'message': 'Connected to backend'}, room=sid)


@sio.event
async def disconnect(sid):
    print(f"✗ Client disconnected: {sid}")


@sio.event
async def compile_model(sid, data):
    """Compile model"""
    global current_model, current_config
    print(f"Compile request from {sid}")
    
    try:
        cfg = (data or {}).get('config') or {}
        compile_id = (data or {}).get('compileId')
        current_config = cfg

        input_size = int(cfg.get('inputSize', 784))
        output_size = int(cfg.get('outputSize', 10))
        hidden_layers = cfg.get('hiddenLayers', []) or []

        # Minimal validation to prevent silent "all zeros" visualization
        for i, layer in enumerate(hidden_layers):
            if int(layer.get('size', 0)) <= 0:
                raise ValueError(f"Hidden layer {i} must have a positive size")

        current_model = ConfigurableMLP(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in current_model.parameters())
        num_linear = len(_get_linear_layers(current_model))
        
        await sio.emit('model_compiled', {
            'success': True,
            'compileId': compile_id,
            'model_info': {
                'total_params': int(total_params),
                'layers': int(num_linear)
            }
        }, room=sid)
        
        print(f"✓ Model compiled: {total_params} parameters")
        
    except Exception as e:
        print(f"✗ Compile error: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)


@sio.event
async def start_training(sid, data):
    """Start training"""
    global is_training
    
    print(f"Training request from {sid}: {data}")
    
    if current_model is None:
        await sio.emit('error', {'message': 'Please compile model first'}, room=sid)
        return
    
    is_training = True
    await sio.emit('training_started', {}, room=sid)
    
    # Start training task
    asyncio.create_task(run_training(sid, data))


async def run_training(sid, hyperparams):
    """Run training loop"""
    global is_training
    
    try:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        import torch.optim as optim
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_model.to(device)
        
        lr = hyperparams.get('learningRate', 0.001)
        batch_size = hyperparams.get('batchSize', 64)
        epochs = hyperparams.get('epochs', 10)
        dataset_name = hyperparams.get('dataset', 'mnist')
        ds = _get_dataset_spec(dataset_name)
        
        print(f"Training: LR={lr}, Batch={batch_size}, Epochs={epochs}, Dataset={ds['key']}")
        
        # Load dataset (backend-local data directory; registry-driven)
        dataset = ds['make_train_dataset'](DATA_DIR, datasets, transforms)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Get optimizer from config
        optimizer_name = hyperparams.get('optimizer', 'Adam')
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(current_model.parameters(), lr=lr)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(current_model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(current_model.parameters(), lr=lr)
            
        criterion_name = hyperparams.get('lossFunction', 'CrossEntropyLoss')
        criterion_name = (criterion_name or 'CrossEntropyLoss')
        use_mse = (criterion_name == 'MSELoss')
        criterion = nn.MSELoss() if use_mse else nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            if not is_training:
                break
            
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if not is_training:
                    break
                
                data, target = data.to(device), target.to(device)

                # Optional: snapshot inspected neuron params for delta reporting.
                inspected_before = None
                if inspected_neuron and current_model is not None:
                    try:
                        linears = _get_linear_layers(current_model)
                        li = int(inspected_neuron.get('layerIndex'))
                        ni = int(inspected_neuron.get('neuronIndex'))
                        if 0 <= li < len(linears):
                            layer = linears[li]
                            if 0 <= ni < layer.weight.shape[0]:
                                inspected_before = {
                                    'layerIndex': li,
                                    'neuronIndex': ni,
                                    'weightRow': layer.weight.detach()[ni].clone(),
                                    'bias': layer.bias.detach()[ni].clone() if layer.bias is not None else torch.tensor(0.0, device=layer.weight.device)
                                }
                    except Exception:
                        inspected_before = None
                
                optimizer.zero_grad()
                output = current_model(data)

                # Make MSELoss usable for classification: compare softmax probs vs one-hot targets.
                if use_mse:
                    probs = torch.softmax(output, dim=1)
                    target_onehot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1.0)
                    loss = criterion(probs, target_onehot)
                else:
                    loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                accuracy = correct / total
                
                epoch_loss += loss.item()
                
                # Send update every 10 batches
                if batch_idx % 10 == 0:
                    await sio.emit('metrics_update', {
                        'epoch': epoch + 1,
                        'batch': batch_idx,
                        'total_epochs': epochs,
                        'total_batches': len(train_loader),
                        'loss': float(loss.item()),
                        'accuracy': float(accuracy),
                        'learning_rate': lr
                    }, room=sid)

                    # Layer-wise parameter/gradient summaries (layer by layer)
                    await sio.emit('param_update', {
                        'epoch': epoch + 1,
                        'batch': batch_idx,
                        'layers': _layer_summaries(current_model)
                    }, room=sid)

                    # Send a real training sample + its activations so the UI can "light up" neurons.
                    try:
                        sample_x = data[:1]
                        sample_y = target[:1]

                        with torch.no_grad():
                            out, acts = current_model(sample_x, capture_activations=True)

                        sample_pixels = None
                        try:
                            # detach and cpu everything before conversion
                            viz_sample = sample_x[0].detach().cpu()
                            img_t = ds['unnormalize_for_viz'](viz_sample)
                            
                            # Ensure we have floats. Flatten to [C*H*W]
                            sample_pixels = img_t.numpy().flatten().tolist()
                            
                            if batch_idx == 0:
                                print(f"✓ Training sample processed: {len(sample_pixels)} values")
                        except Exception as e:
                            print(f"✗ Sample viz error: {e}")
                            sample_pixels = None

                        await sio.emit('training_sample', {
                            'epoch': epoch + 1,
                            'batch': batch_idx,
                            'target': int(sample_y[0].item()),
                            'prediction': int(out.argmax(dim=1)[0].item()),
                            'pixels': sample_pixels,
                            'activations': acts
                        }, room=sid)
                    except Exception as e:
                        print(f"training_sample emit error: {e}")

                    # Selected neuron telemetry (neuron by neuron, for the inspected one)
                    if inspected_neuron:
                        try:
                            linears = _get_linear_layers(current_model)
                            li = int(inspected_neuron.get('layerIndex'))
                            ni = int(inspected_neuron.get('neuronIndex'))
                            if 0 <= li < len(linears):
                                layer = linears[li]
                                if 0 <= ni < layer.weight.shape[0]:
                                    w_row = layer.weight.detach()[ni]
                                    b_val = layer.bias.detach()[ni] if layer.bias is not None else torch.tensor(0.0, device=w_row.device)

                                    w_row_grad = layer.weight.grad[ni] if layer.weight.grad is not None else None
                                    b_grad = layer.bias.grad[ni] if (layer.bias is not None and layer.bias.grad is not None) else None

                                    delta_w_norm = None
                                    delta_b = None
                                    if inspected_before and inspected_before.get('layerIndex') == li and inspected_before.get('neuronIndex') == ni:
                                        delta_w_norm = float((w_row - inspected_before['weightRow']).norm().item())
                                        delta_b = float((b_val - inspected_before['bias']).item())

                                    await sio.emit('inspected_neuron_update', {
                                        'epoch': epoch + 1,
                                        'batch': batch_idx,
                                        'layerIndex': li,
                                        'neuronIndex': ni,
                                        'bias': float(b_val.item()),
                                        'biasGrad': float(b_grad.item()) if b_grad is not None else 0.0,
                                        'weightRow': w_row.cpu().numpy().tolist(),
                                        'weightRowGradMeanAbs': float(w_row_grad.detach().abs().mean().item()) if w_row_grad is not None else 0.0,
                                        'deltaWeightRowNorm': delta_w_norm,
                                        'deltaBias': delta_b
                                    }, room=sid)
                        except Exception as e:
                            # Don't kill training for inspection issues.
                            print(f"Inspection telemetry error: {e}")
                    
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
        
        is_training = False
        await sio.emit('training_stopped', {}, room=sid)
        print("Training complete")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        is_training = False
        await sio.emit('error', {'message': str(e)}, room=sid)


@sio.event
async def stop_training(sid, data):
    """Stop training"""
    global is_training
    is_training = False
    await sio.emit('training_stopped', {}, room=sid)
    print("Training stopped by user")


@sio.event
async def inspect_neuron(sid, data):
    """Set inspected neuron (for training telemetry) and return its current params."""
    global inspected_neuron

    print(f"✓ Inspect neuron request from {sid}: {data}")

    if current_model is None:
        print("✗ Model not compiled")
        await sio.emit('error', {'message': 'Model not compiled'}, room=sid)
        return

    try:
        payload = data or {}
        layer_index = int(payload.get('layerIndex'))
        neuron_index = int(payload.get('neuronIndex'))

        print(f"  Inspecting layer {layer_index}, neuron {neuron_index}")

        inspected_neuron = {"layerIndex": layer_index, "neuronIndex": neuron_index}

        linears = _get_linear_layers(current_model)
        if not (0 <= layer_index < len(linears)):
            raise ValueError(f"layerIndex out of range (0..{len(linears)-1})")

        layer = linears[layer_index]
        if not (0 <= neuron_index < layer.weight.shape[0]):
            raise ValueError(f"neuronIndex out of range (0..{layer.weight.shape[0]-1})")

        w_row = layer.weight.detach()[neuron_index].cpu().numpy().tolist()
        b_val = float(layer.bias.detach()[neuron_index].item()) if layer.bias is not None else 0.0

        response = {
            'layerIndex': layer_index,
            'neuronIndex': neuron_index,
            'bias': b_val,
            'weightRow': w_row,
            'inFeatures': int(layer.weight.shape[1]),
            'outFeatures': int(layer.weight.shape[0])
        }

        print(f"  ✓ Emitting neuron_inspection: layer={layer_index}, neuron={neuron_index}, bias={b_val:.6f}")
        await sio.emit('neuron_inspection', response, room=sid)

    except Exception as e:
        print(f"✗ Inspect neuron error: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {'message': str(e)}, room=sid)


@app.post("/api/predict")
async def predict(data: dict):
    """Predict on user-drawn input and return activations"""
    global current_model
    
    if current_model is None:
        return {"error": "Model not compiled"}
    
    try:
        # Get pixels from request
        pixels = data.get('pixels', [])
        if not isinstance(pixels, list):
            return {"error": "pixels must be a list"}

        dataset_name = (current_config or {}).get('dataset', 'mnist') if current_config else 'mnist'
        ds = _get_dataset_spec(dataset_name)

        expected = int(ds['expected_pixels'])
        if len(pixels) != expected:
            return {"error": f"Expected {expected} pixels for {ds['key']}, got {len(pixels)}"}

        if ds['key'] == 'cifar10':
            input_tensor = torch.tensor(pixels, dtype=torch.float32).reshape(1, 3, 32, 32)
        else:
            input_tensor = torch.tensor(pixels, dtype=torch.float32).reshape(1, 1, 28, 28)

        input_tensor = ds['normalize_input'](input_tensor)
        
        # Run forward pass and capture layer activations
        with torch.no_grad():
            output, activations = current_model(input_tensor, capture_activations=True)
        
        # Get prediction
        probabilities = torch.softmax(output, dim=1).squeeze()
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[predicted_class].item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        top3 = [
            {"class": int(idx), "confidence": float(prob)}
            for idx, prob in zip(top3_indices, top3_probs)
        ]
        
        print(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
        
        return {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "top3": top3,
            "activations": activations,
            "output_scores": output.squeeze().tolist()
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    print("=" * 60)
    print("Visual Deep Learning IDE - Backend Server")
    print("=" * 60)
    print("Server starting on http://0.0.0.0:8000")
    print("Socket.IO endpoint: ws://0.0.0.0:8000/socket.io/")
    print("=" * 60)
    uvicorn.run(socket_app, host="0.0.0.0", port=8000, log_level="info")
