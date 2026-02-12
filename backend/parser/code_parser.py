"""
Code Parser
Parses Python code to extract graph structure
"""

import ast
from typing import Dict, List, Tuple


class CodeParser:
    def __init__(self):
        self.nodes = []
        self.edges = []
    
    def parse(self, code: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse Python code and extract graph structure
        Returns: (nodes, edges)
        """
        self.nodes = []
        self.edges = []
        
        try:
            tree = ast.parse(code)
            self._visit_tree(tree)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {e}")
        
        return self.nodes, self.edges
    
    def _visit_tree(self, tree: ast.AST):
        """Visit AST tree and extract model structure"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if any(base.id == 'Module' for base in node.bases if isinstance(base, ast.Name)):
                    self._parse_model_class(node)
    
    def _parse_model_class(self, class_node: ast.ClassDef):
        """Parse nn.Module class definition"""
        # Find __init__ method
        init_method = None
        forward_method = None
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == '__init__':
                    init_method = item
                elif item.name == 'forward':
                    forward_method = item
        
        if init_method:
            self._parse_init_method(init_method)
        
        if forward_method:
            self._parse_forward_method(forward_method)
    
    def _parse_init_method(self, init_node: ast.FunctionDef):
        """Parse __init__ method to extract layers"""
        for stmt in init_node.body:
            if isinstance(stmt, ast.Assign):
                # Look for self.layer_name = nn.Layer(...)
                if isinstance(stmt.targets[0], ast.Attribute):
                    if isinstance(stmt.targets[0].value, ast.Name) and stmt.targets[0].value.id == 'self':
                        layer_name = stmt.targets[0].attr
                        
                        if isinstance(stmt.value, ast.Call):
                            layer_info = self._parse_layer_call(stmt.value)
                            if layer_info:
                                node_id = f"node_{len(self.nodes)}"
                                self.nodes.append({
                                    'id': node_id,
                                    'type': 'layer',
                                    'data': {
                                        'type': layer_info['type'],
                                        'params': layer_info['params'],
                                        'label': layer_name
                                    },
                                    'position': {'x': 100 + len(self.nodes) * 200, 'y': 100}
                                })
    
    def _parse_layer_call(self, call_node: ast.Call) -> Dict:
        """Parse layer instantiation call"""
        if isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == 'nn':
                layer_type = call_node.func.attr.lower()
                params = {}
                
                # Extract positional arguments
                if layer_type == 'linear' and len(call_node.args) >= 2:
                    params['in_features'] = self._get_constant_value(call_node.args[0])
                    params['out_features'] = self._get_constant_value(call_node.args[1])
                
                elif layer_type == 'conv2d' and len(call_node.args) >= 2:
                    params['in_channels'] = self._get_constant_value(call_node.args[0])
                    params['out_channels'] = self._get_constant_value(call_node.args[1])
                    
                    # Extract keyword arguments
                    for keyword in call_node.keywords:
                        if keyword.arg == 'kernel_size':
                            params['kernel_size'] = self._get_constant_value(keyword.value)
                
                return {
                    'type': layer_type,
                    'params': params
                }
        
        return None
    
    def _parse_forward_method(self, forward_node: ast.FunctionDef):
        """Parse forward method to extract connections"""
        # This is a simplified version
        # In a full implementation, we would track tensor flow through the forward pass
        # For now, we'll create sequential connections
        
        if len(self.nodes) > 1:
            for i in range(len(self.nodes) - 1):
                self.edges.append({
                    'id': f"edge_{i}",
                    'source': self.nodes[i]['id'],
                    'target': self.nodes[i + 1]['id']
                })
    
    def _get_constant_value(self, node: ast.AST):
        """Extract constant value from AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        return None
