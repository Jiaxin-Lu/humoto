import json
import torch
import torch.nn as nn
import numpy as np
import trimesh
from typing import Dict, List, Optional, Tuple, Union
import sys
import os
HUMAN_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

_kaolin = None

def _get_kaolin():
    """Lazy import kaolin with caching"""
    global _kaolin
    if _kaolin is None:
        try:
            import kaolin
            _kaolin = kaolin
        except ImportError:
            _kaolin = False
            raise ImportError("Kaolin package is not available")
    elif _kaolin is False:
        raise ImportError("Kaolin package is not available")
    return _kaolin

class HumanModelDifferentiable(nn.Module):
    def __init__(self, character_data_path: str = os.path.join(HUMAN_MODEL_DIR, 'human_model_data_yup.json'), device: str = "cpu"):
        """Initialize differentiable mesh deformer with enhanced joint functionality"""
        super().__init__()
        self.device = device
        
        # Load character data
        with open(character_data_path, 'r') as f:
            data = json.load(f)
        
        # Convert vertices to tensor (Nx3)
        vertices = torch.tensor(data['vertices'], dtype=torch.float32, device=device)
        self.register_buffer('vertices', vertices)
        
        # Process faces
        self.faces = data['faces']
        self.triangulated_faces = []
        for face in self.faces:
            if len(face) == 3:
                self.triangulated_faces.append(face)
            elif len(face) == 4:
                self.triangulated_faces.append([face[0], face[1], face[2]])
                self.triangulated_faces.append([face[0], face[2], face[3]])
            else:
                for i in range(1, len(face) - 1):
                    self.triangulated_faces.append([face[0], face[i], face[i + 1]])
        self.triangulated_faces = np.array(self.triangulated_faces)
        self.register_buffer("triangulated_faces_torch", torch.tensor(self.triangulated_faces, dtype=torch.long, device=device))
        
        # Convert to homogeneous coordinates (Nx4)
        vertices_homogeneous = torch.ones((len(vertices), 4), dtype=torch.float32, device=device)
        vertices_homogeneous[:, :3] = vertices
        self.register_buffer('vertices_homogeneous', vertices_homogeneous)
        
        # Process bone data with enhanced joint information
        self.bone_chains = {}  # Cache bone chains
        self.bone_rest_matrices = {}
        self.bone_rest_inverse = {}
        self.bone_names = list(data['bones'].keys())
        
        # Cache parent-child relationships for IK
        self.child_bones = {bone_name: [] for bone_name in self.bone_names}
        self.parent_bones = {}
        
        for bone_name, bone_data in data['bones'].items():
            # Convert matrices to tensors
            rest_matrix = torch.tensor(bone_data['matrix'], dtype=torch.float32, device=device)
            rest_inverse = torch.inverse(rest_matrix)
            
            self.bone_rest_matrices[bone_name] = rest_matrix
            self.bone_rest_inverse[bone_name] = rest_inverse
            
            # Extract and store joint position from rest matrix
            joint_pos = rest_matrix[:3, 3]
            self.register_buffer(f'joint_rest_pos_{bone_name}', joint_pos)
            
            # Store parent-child relationships
            parent = bone_data['parent']
            self.parent_bones[bone_name] = parent
            if parent:
                self.child_bones[parent].append(bone_name)
            
            # Cache bone chain
            chain = []
            current_bone = bone_name
            while current_bone:
                chain.append(current_bone)
                current_bone = data['bones'][current_bone]['parent']
            self.bone_chains[bone_name] = list(reversed(chain))
        
        self.parents_idx = []
        for bone_name in self.bone_names:
            if bone_name == 'mixamorig:Hips' or '4' in bone_name or 'End' in bone_name or self.parent_bones.get(bone_name, '') in ['', None]:
                self.parents_idx.append(-1)
            else:
                self.parents_idx.append(self.bone_names.index(self.parent_bones.get(bone_name, '')))
        self.register_buffer('parents_idx_torch', torch.tensor(self.parents_idx, dtype=torch.long, device=device))
        
        # Process skinning weights
        self.bone_weight_indices = {}
        self.bone_weight_values = {}
        
        for vertex_idx, influences in data['bone_influences'].items():
            vertex_idx = int(vertex_idx)
            for influence in influences:
                bone_name = influence['bone']
                weight = influence['weight']
                
                if bone_name not in self.bone_weight_indices:
                    self.bone_weight_indices[bone_name] = []
                    self.bone_weight_values[bone_name] = []
                
                self.bone_weight_indices[bone_name].append(vertex_idx)
                self.bone_weight_values[bone_name].append(weight)
        
        # Convert weight data to tensors
        for bone_name in self.bone_weight_indices:
            indices = torch.tensor(self.bone_weight_indices[bone_name], dtype=torch.long, device=device)
            values = torch.tensor(self.bone_weight_values[bone_name], dtype=torch.float32, device=device)
            self.register_buffer(f'weight_indices_{bone_name}', indices)
            self.register_buffer(f'weight_values_{bone_name}', values)
        
        self.deformed_vertices = None
        self.joint_positions = None

    def compute_bone_transforms(self, pose_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute transformation matrices for all bones
        Args:
            pose_params: Dict of bone_name -> tensor [batch_size, 4, 4] (rotation matrix + translation)
        Returns:
            Dict of bone_name -> tensor [batch_size, 4, 4] (transformation matrices)
        """
        batch_size = next(iter(pose_params.values())).shape[0]
        bone_transforms = {}
        
        for bone_name in self.bone_rest_matrices:
            final_transform = torch.eye(4, dtype=torch.float32, 
                                     device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
            
            for chain_bone in self.bone_chains[bone_name]:
                rest_matrix = self.bone_rest_matrices[chain_bone]
                rest_inverse = self.bone_rest_inverse[chain_bone]
                
                if chain_bone in pose_params:
                    # Provided as full transformation matrices [B, 4, 4]
                    transform = pose_params[chain_bone]
                else:
                    transform = torch.eye(4, dtype=torch.float32, 
                                      device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
                
                local_transform = rest_matrix.unsqueeze(0) @ transform @ rest_inverse.unsqueeze(0)
                final_transform = final_transform @ local_transform
            
            bone_transforms[bone_name] = final_transform
            
        return bone_transforms

    def compute_joint_positions(self, pose_params: Dict[str, torch.Tensor], 
                              bone_transforms: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute the positions of all joints in world space
        Args:
            pose_params: Dict of bone_name -> tensor [batch_size, 4, 4] (rotation matrix + translation)
            bone_transforms: Optional precomputed bone transforms
        Returns:
            Dict of bone_name -> tensor [batch_size, 3] (joint positions)
        """
        if bone_transforms is None:
            bone_transforms = self.compute_bone_transforms(pose_params)
            
        joint_positions = {}
        batch_size = next(iter(pose_params.values())).shape[0]
        
        for bone_name in self.bone_rest_matrices:
            rest_pos = getattr(self, f'joint_rest_pos_{bone_name}')
            rest_pos_homog = torch.cat([rest_pos, torch.ones(1, device=self.device)])
            
            # Transform rest position to world space
            pos_homog = rest_pos_homog.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
            transformed = torch.matmul(bone_transforms[bone_name], pos_homog)
            joint_positions[bone_name] = transformed[:, :3, 0]
        
        return joint_positions

    def get_joint_chain(self, start_bone: str, end_bone: str) -> List[str]:
        """
        Get the chain of joints between two bones for IK
        Args:
            start_bone: Starting bone name
            end_bone: Ending bone name
        Returns:
            List of bone names in the chain
        """
        if start_bone not in self.bone_names or end_bone not in self.bone_names:
            raise ValueError(f"Invalid bone names: {start_bone}, {end_bone}")
            
        # Find the chain from end to root
        end_chain = []
        current = end_bone
        while current:
            end_chain.append(current)
            current = self.parent_bones.get(current)
            
        # Find the chain from start to root
        start_chain = []
        current = start_bone
        while current:
            if current in end_chain:
                # Found common ancestor
                common_idx = end_chain.index(current)
                return start_chain[::-1] + end_chain[:common_idx+1]
            start_chain.append(current)
            current = self.parent_bones.get(current)
            
        return []  # No valid chain found

    def tensor_to_pose_params(self, pose_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert pose tensor to pose parameters

        Args:
            pose_tensor: Tensor [batch_size, num_bones, 4, 4]
        Returns:
            Dict of bone_name -> tensor [batch_size, 4, 4]
        """
        assert pose_tensor.shape[-3] == len(self.bone_names), f"Pose tensor has {pose_tensor.shape[-3]} bones, but model has {len(self.bone_names)} bones"
        pose_params = {self.bone_names[i]: pose_tensor[:, i] for i in range(pose_tensor.shape[-3])}
        return pose_params
    
    def pose_params_to_tensor(self, pose_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert pose parameters to pose tensor
        """
        if len(pose_params) == 0:
            # Default to identity transforms
            return torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(1, len(self.bone_names), 4, 4).clone()
        batch_size = next(iter(pose_params.values())).shape[0]
        default_pose_tensor = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        pose_tensor = [pose_params.get(bone_name, default_pose_tensor) for bone_name in self.bone_names]
        pose_tensor = torch.stack(pose_tensor, dim=1)
        return pose_tensor

    def joint_positions_to_tensor(self, joint_positions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert joint positions to pose tensor
        """
        if len(joint_positions) == 0:
            return torch.zeros(1, len(self.bone_names), 3, device=self.device)
        batch_size = next(iter(joint_positions.values())).shape[0]
        default_joint_positions_tensor = torch.zeros(batch_size, 3, device=self.device)
        joint_positions_tensor = [joint_positions.get(bone_name, default_joint_positions_tensor) for bone_name in self.bone_names]
        joint_positions_tensor = torch.stack(joint_positions_tensor, dim=1)
        return joint_positions_tensor

    def forward(self, pose_params: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass to compute deformed vertices and joint positions
        Args:
            pose_params: Dict of bone_name -> tensor [batch_size, 4, 4] (rotation matrix + translation)
        Returns:
            - Deformed vertices tensor [batch_size, num_vertices, 3]
            - Dict of joint positions {bone_name: tensor [batch_size, 3]}
        """
        if isinstance(pose_params, dict):
            batch_size = next(iter(pose_params.values())).shape[0]
        else:
            batch_size = pose_params.shape[0]
            pose_params = self.tensor_to_pose_params(pose_params)
        num_vertices = len(self.vertices)
        self.deformed_vertices = None
        self.joint_positions = None
        
        # Initialize deformed positions
        deformed_positions = torch.zeros((batch_size, num_vertices, 3), 
                                      dtype=torch.float32, device=self.device)
        vertex_weights = torch.zeros((batch_size, num_vertices), 
                                   dtype=torch.float32, device=self.device)
        
        # Compute bone transforms once
        bone_transforms = self.compute_bone_transforms(pose_params)
        
        # Deform vertices
        for bone_name in self.bone_rest_matrices:
            if not hasattr(self, f'weight_indices_{bone_name}'):
                continue
                
            # Get vertices influenced by this bone
            indices = getattr(self, f'weight_indices_{bone_name}')
            weights = getattr(self, f'weight_values_{bone_name}')
            
            # Transform vertices using precomputed bone transform
            vertices = self.vertices_homogeneous[indices]  # [N, 4]
            vertices_batch = vertices.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 4]
            vertices_batch = vertices_batch.unsqueeze(-1)  # [B, N, 4, 1]
            
            transform_expanded = bone_transforms[bone_name].unsqueeze(1)  # [B, 1, 4, 4]
            transformed = torch.matmul(transform_expanded, vertices_batch)  # [B, N, 4, 1]
            transformed = transformed.squeeze(-1)  # [B, N, 4]
            
            # Handle homogeneous divide
            transformed_3d = transformed[..., :3] / transformed[..., 3:4]  # [B, N, 3]
            
            # Accumulate weighted positions
            weight_matrix = weights.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
            deformed_positions[:, indices] += transformed_3d * weight_matrix
            vertex_weights[:, indices] += weights
        
        # Normalize by total weights
        valid_weights = vertex_weights > 0
        deformed_positions[valid_weights] /= vertex_weights[valid_weights].unsqueeze(-1)
        self.deformed_vertices = deformed_positions
        
        # Compute joint positions
        joint_positions = self.compute_joint_positions(pose_params, bone_transforms)
        self.joint_positions = joint_positions
        
        return deformed_positions, joint_positions

    def to_trimesh(self, batch_idx: int = 0) -> 'trimesh.Trimesh':
        """Convert deformed vertices to a trimesh object"""
        vertices = self.deformed_vertices[batch_idx].detach().cpu().numpy()
        return trimesh.Trimesh(vertices=vertices, faces=self.triangulated_faces)
    
    def compute_point_penetration_batched(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fully batched computation of point penetration using Kaolin's optimized operations.
        
        Note: Kaolin's check_sign doesn't support different faces for each batch.
        
        Args:
            points: Point cloud tensor [batch_size, num_points, 3]
        Returns:
            - penetration: Penetration depths [batch_size, num_points]
            - is_inside: Binary inside/outside mask [batch_size, num_points]
        """
        kaolin = _get_kaolin()
    
        vertices = self.deformed_vertices
        faces = self.triangulated_faces_torch
        face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)
        unsigned_distance, index, dist_type = kaolin.metrics.trianglemesh.point_to_mesh_distance(points, face_vertices)
        is_inside = kaolin.ops.mesh.check_sign(vertices, faces, points)
        penetration = unsigned_distance * is_inside.float()
        return penetration, is_inside, unsigned_distance, index
