"""
Graph Compiler
Converts JSON graph to PyTorch nn.Module
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from thop import profile


class GraphCompiler:
    def __init__(self):
        self.model = None
        self.layer_registry = {
            'linear': nn.Linear,
            'conv2d': nn.Conv2d,
            'batchnorm2d': nn.BatchNorm2d,
            'dropout': nn.Dropout,
            'maxpool2d': nn.MaxPool2d,
        }
    
    def compile(self, nodes: List[Dict], edges: List[Dict]) -> Tuple[nn.Module, Dict]:
        """
        Compile graph to PyTorch model
        Returns: (model, model_info)
        """
        # Topological sort
        sorted_nodes = self._topological_sort(nodes, edges)
        
        # Build model
        model = self._build_model(sorted_nodes, edges)
        
        # Calculate model info
        model_info = self._calculate_model_info(model)
        
        self.model = model
        return model, model_info
    
    def _topological_sort(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Topological sort using Kahn's algorithm"""
        # Build adjacency list and in-degree map
        adjacency = {node['id']: [] for node in nodes}
        in_degree = {node['id']: 0 for node in nodes}
        
        for edge in edges:
            adjacency[edge['source']].append(edge['target'])
            in_degree[edge['target']] += 1
        
        # Find nodes with no incoming edges
        queue = [node for node in nodes if in_degree[node['id']] == 0]
        sorted_nodes = []
        
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            
            for neighbor_id in adjacency[node['id']]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    neighbor = next(n for n in nodes if n['id'] == neighbor_id)
                    queue.append(neighbor)
        
        return sorted_nodes
    
    def _build_model(self, sorted_nodes: List[Dict], edges: List[Dict]) -> nn.Module:
        """Build PyTorch model from sorted nodes"""
        
        class DynamicModel(nn.Module):
            def __init__(self, nodes, edges, layer_registry):
                super().__init__()
                self.nodes = nodes
                self.edges = edges
                self.layers = nn.ModuleDict()
                
                # Create layers
                for node in nodes:
                    if node['data']['type'] in ['input', 'output']:
                        continue
                    
                    layer_type = node['data']['type']
                    params = node['data'].get('params', {})
                    
                    if layer_type in layer_registry:
                        layer_class = layer_registry[layer_type]
                        layer = self._create_layer(layer_class, params)
                        self.layers[node['id']] = layer
            
            def _create_layer(self, layer_class, params):
                """Create layer instance from class and params"""
                if layer_class == nn.Linear:
                    return layer_class(params['in_features'], params['out_features'])
                elif layer_class == nn.Conv2d:
                    return layer_class(
                        params['in_channels'],
                        params['out_channels'],
                        kernel_size=params.get('kernel_size', 3)
                    )
                elif layer_class == nn.BatchNorm2d:
                    return layer_class(params['num_features'])
                elif layer_class == nn.Dropout:
                    return layer_class(p=params.get('p', 0.5))
                elif layer_class == nn.MaxPool2d:
                    return layer_class(kernel_size=params.get('kernel_size', 2))
                else:
                    return layer_class()
            
            def forward(self, x):
                # Track tensors for each node
                tensors = {}
                
                # Find input node
                input_node = next(n for n in self.nodes if n['data']['type'] == 'input')
                tensors[input_node['id']] = x
                
                # Execute layers in order
                for node in self.nodes:
                    if node['data']['type'] in ['input', 'output']:
                        continue
                    
                    # Get input tensor
                    input_edge = next((e for e in self.edges if e['target'] == node['id']), None)
                    if input_edge:
                        input_tensor = tensors[input_edge['source']]
                    else:
                        input_tensor = x
                    
                    # Apply layer or activation
                    layer_type = node['data']['type']
                    
                    if layer_type in self.layers:
                        output = self.layers[node['id']](input_tensor)
                    elif layer_type == 'relu':
                        output = torch.relu(input_tensor)
                    elif layer_type == 'sigmoid':
                        output = torch.sigmoid(input_tensor)
                    elif layer_type == 'tanh':
                        output = torch.tanh(input_tensor)
                    else:
                        output = input_tensor
                    
                    tensors[node['id']] = output
                
                # Return output from last node
                output_node = next((n for n in self.nodes if n['data']['type'] == 'output'), None)
                if output_node:
                    output_edge = next((e for e in self.edges if e['target'] == output_node['id']), None)
                    if output_edge:
                        return tensors[output_edge['source']]
                
                # Return last tensor
                return list(tensors.values())[-1] if tensors else x
        
        model = DynamicModel(sorted_nodes, edges, self.layer_registry)
        return model
    
    def _calculate_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Calculate model statistics"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Try to calculate FLOPs (requires sample input)
        try:
            sample_input = torch.randn(1, 3, 32, 32)  # Default CIFAR-10 size
            flops, _ = profile(model, inputs=(sample_input,), verbose=False)
        except:
            flops = 0
        
        # Estimate memory
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        memory_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'flops': int(flops),
            'memory_mb': float(memory_mb)
        }
