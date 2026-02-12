"""
Training Engine
Handles model training with hooks for introspection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import asyncio
from typing import Dict, Any
import numpy as np


class TrainingEngine:
    def __init__(self, model: nn.Module, sio, sid):
        self.model = model
        self.sio = sio
        self.sid = sid
        self.is_training = False
        self.is_paused = False
        self.hooks = {}
        self.activations = {}
        self.gradients = {}
    
    async def train(self, hyperparameters: Dict[str, Any]):
        """Main training loop"""
        self.is_training = True
        self.is_paused = False
        
        print(f"Training with hyperparameters: {hyperparameters}")
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Hyperparameters
        lr = hyperparameters.get('learningRate', 0.001)
        batch_size = hyperparameters.get('batchSize', 64)
        epochs = hyperparameters.get('epochs', 10)
        dataset_name = hyperparameters.get('dataset', 'mnist')
        
        print(f"Device: {device}, LR: {lr}, Batch: {batch_size}, Epochs: {epochs}, Dataset: {dataset_name}")
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Dataset
        try:
            train_loader = self._get_dataloader(dataset_name, batch_size, train=True)
            print(f"Loaded {dataset_name} dataset")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            await self.sio.emit('error', {'message': f'Error loading dataset: {e}'}, room=self.sid)
            return
        
        # Register hooks
        self._register_hooks()
        
        # Training loop
        for epoch in range(epochs):
            if not self.is_training:
                break
            
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Check if paused
                while self.is_paused and self.is_training:
                    await asyncio.sleep(0.1)
                
                if not self.is_training:
                    break
                
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                accuracy = correct / total
                
                epoch_loss += loss.item()
                
                # Send metrics update every 10 batches
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
                    
                    await self.sio.emit('metrics_update', {
                        'epoch': epoch + 1,
                        'batch': batch_idx,
                        'total_epochs': epochs,
                        'total_batches': len(train_loader),
                        'loss': float(loss.item()),
                        'accuracy': float(accuracy),
                        'learning_rate': lr
                    }, room=self.sid)
                    
                    # Send gradient flow data
                    await self._send_gradient_flow()
            
            # End of epoch summary
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} complete - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        self.is_training = False
        self._remove_hooks()
        
        await self.sio.emit('training_stopped', {}, room=self.sid)
        print("Training complete")
    
    def pause(self):
        """Pause training"""
        self.is_paused = True
    
    def resume(self):
        """Resume training"""
        self.is_paused = False
    
    def stop(self):
        """Stop training"""
        self.is_training = False
        self.is_paused = False
    
    async def step_batch(self):
        """Execute one batch (for debugging)"""
        # TODO: Implement single batch stepping
        pass
    
    def get_inspection_data(self, node_id: str) -> Dict:
        """Get inspection data for a specific node"""
        return {
            'node_id': node_id,
            'activations': self.activations.get(node_id, {}),
            'gradients': self.gradients.get(node_id, {}),
            'statistics': self._calculate_statistics(node_id)
        }
    
    def _register_hooks(self):
        """Register forward and backward hooks on all layers"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                # Forward hook
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._forward_hook(n, out)
                )
                self.hooks[f"{name}_forward"] = hook
                
                # Backward hook
                hook = module.register_full_backward_hook(
                    lambda m, grad_in, grad_out, n=name: self._backward_hook(n, grad_out)
                )
                self.hooks[f"{name}_backward"] = hook
    
    def _forward_hook(self, name: str, output: torch.Tensor):
        """Capture activations"""
        if isinstance(output, torch.Tensor):
            self.activations[name] = output.detach().cpu().numpy()
    
    def _backward_hook(self, name: str, grad_output):
        """Capture gradients"""
        if grad_output and isinstance(grad_output[0], torch.Tensor):
            self.gradients[name] = grad_output[0].detach().cpu().numpy()
    
    def _remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
    
    async def _send_gradient_flow(self):
        """Send gradient flow data to frontend"""
        gradient_flow = {}
        
        for name, grad in self.gradients.items():
            if grad is not None:
                magnitude = float(np.abs(grad).mean())
                sign = float(np.sign(grad.mean()))
                
                gradient_flow[name] = {
                    'magnitude': magnitude,
                    'sign': sign
                }
        
        await self.sio.emit('gradient_flow', gradient_flow, room=self.sid)
    
    def _calculate_statistics(self, node_id: str) -> Dict:
        """Calculate statistics for a node"""
        stats = {}
        
        if node_id in self.activations:
            act = self.activations[node_id]
            stats['activation_mean'] = float(np.mean(act))
            stats['activation_std'] = float(np.std(act))
            stats['activation_min'] = float(np.min(act))
            stats['activation_max'] = float(np.max(act))
        
        if node_id in self.gradients:
            grad = self.gradients[node_id]
            stats['gradient_mean'] = float(np.mean(grad))
            stats['gradient_std'] = float(np.std(grad))
            stats['gradient_norm'] = float(np.linalg.norm(grad))
        
        return stats
    
    def _get_dataloader(self, dataset_name: str, batch_size: int, train: bool = True) -> DataLoader:
        """Get dataset dataloader"""
        if dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
        
        elif dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = datasets.CIFAR10('./data', train=train, download=True, transform=transform)
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=train)
