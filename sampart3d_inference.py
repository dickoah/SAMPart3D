"""
SAMPart3D Inference Module

This module provides a high-level interface for performing mesh segmentation
using the SAMPart3D model.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Union
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import trimesh
import matplotlib.colors as mcolors

try:
    from cuml.cluster.hdbscan import HDBSCAN
except ImportError:
    from sklearn.cluster import HDBSCAN

import pointops
from sklearn.preprocessing import QuantileTransformer

# Import model and dataset utilities
from pointcept.models.builder import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.sampart3d_util import sample_surface


class SAMPart3DInference:
    """
    High-level inference interface for SAMPart3D mesh segmentation.
    
    This class handles:
    - Model loading and initialization
    - Mesh preprocessing (sampling, normalization, grid sampling)
    - Multi-scale feature extraction
    - Clustering and segmentation
    - Result visualization and mesh coloring
    """
    
    # Default paths (relative to the repo root)
    DEFAULT_CONFIG = "configs/sampart3d/sampart3d-trainmlp-render16views.py"
    DEFAULT_BACKBONE_WEIGHT = None  # User should provide
    DEFAULT_MODEL_WEIGHT = None  # User should provide
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        backbone_weight_path: Optional[str] = None,
        model_weight_path: Optional[str] = None,
        device: str = "cuda",
        sample_num: int = 15000,
        grid_size: float = 0.01,
    ):
        """
        Initialize the SAMPart3D inference engine.
        
        Args:
            config_path: Path to config file. If None, uses default config.
            backbone_weight_path: Path to pretrained backbone weights (PTv3).
            model_weight_path: Path to trained SAMPart3D model weights.
            device: Device to run inference on ("cuda" or "cpu").
            sample_num: Number of points to sample from mesh surface.
            grid_size: Grid size for voxelization.
        """
        self.device = device
        self.sample_num = sample_num
        self.grid_size = grid_size
        
        # Load config
        self.cfg = self._load_config(config_path)
        
        # Build model
        self.model = self._build_model()
        
        # Load weights
        self._load_weights(backbone_weight_path, model_weight_path)
        
        # Set model to eval mode
        self.model.eval()
        
        # Build transform pipeline
        self.transform = self._build_transform()
        
        # Color palette for visualization
        self.colors = self._generate_colors()
        
        print(f"SAMPart3D inference engine initialized on {device}")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file."""
        if config_path is None or config_path == "":
            # Use default config
            repo_root = Path(__file__).parent
            config_path = repo_root / self.DEFAULT_CONFIG
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Parse config file
        cfg = {}
        with open(config_path, 'r') as f:
            content = f.read()
            exec(content, cfg)
        
        # Handle base config
        if "_base_" in cfg:
            base_path = config_path.parent / cfg["_base_"][0]
            with open(base_path, 'r') as f:
                base_content = f.read()
                base_cfg = {}
                exec(base_content, base_cfg)
                # Merge base config with current
                for k, v in base_cfg.items():
                    if k not in cfg:
                        cfg[k] = v
        
        return cfg
    
    def _build_model(self) -> nn.Module:
        """Build the SAMPart3D model."""
        model_cfg = self.cfg.get("model", {})
        
        # Build model using the registry
        model = build_model(model_cfg)
        model = model.to(self.device)
        
        return model
    
    def _load_weights(
        self,
        backbone_weight_path: Optional[str],
        model_weight_path: Optional[str]
    ):
        """Load pretrained weights for backbone and full model."""
        
        # Load backbone weights
        if backbone_weight_path and backbone_weight_path != "" and os.path.isfile(backbone_weight_path):
            print(f"Loading backbone weights from: {backbone_weight_path}")
            checkpoint = torch.load(
                backbone_weight_path,
                map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage,
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                # Remove "module." prefix if present
                if key.startswith("module."):
                    key = key[7:]
                # Add "backbone." prefix
                key = "backbone." + key
                weight[key] = value
            
            load_info = self.model.load_state_dict(weight, strict=False)
            print(f"Backbone loaded. Missing keys: {len(load_info.missing_keys)}")
        
        # Load full model weights
        if model_weight_path and model_weight_path != "" and os.path.isfile(model_weight_path):
            print(f"Loading model weights from: {model_weight_path}")
            checkpoint = torch.load(
                model_weight_path,
                map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage,
            )
            load_info = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(f"Model loaded. Missing keys: {len(load_info.missing_keys)}")
            
            # Load scale statistics for quantile transformer
            if "scale_statistics" in checkpoint["state_dict"]:
                scale_statistics = checkpoint["state_dict"]["scale_statistics"]
                self.model.quantile_transformer = self._get_quantile_func(scale_statistics)
            else:
                # Create default quantile transformer if not in checkpoint
                self.model.quantile_transformer = self._get_default_quantile_func()
        else:
            # Create default quantile transformer
            self.model.quantile_transformer = self._get_default_quantile_func()
    
    def _get_quantile_func(self, scales: torch.Tensor, distribution: str = "normal"):
        """Create quantile transformer function from scale statistics."""
        scales = scales.flatten()
        max_grouping_scale = 2
        scales = scales[(scales > 0) & (scales < max_grouping_scale)]
        scales = scales.detach().cpu().numpy()
        
        quantile_transformer = QuantileTransformer(output_distribution=distribution)
        quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))
        
        def quantile_transformer_func(scales):
            return torch.Tensor(
                quantile_transformer.transform(scales.cpu().numpy())
            ).to(scales.device)
        
        return quantile_transformer_func
    
    def _get_default_quantile_func(self, distribution: str = "normal"):
        """Create a default quantile transformer when no scale statistics available."""
        # Create dummy scales for initialization
        scales = np.linspace(0.01, 1.9, 1000)
        
        quantile_transformer = QuantileTransformer(output_distribution=distribution)
        quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))
        
        def quantile_transformer_func(scales):
            return torch.Tensor(
                quantile_transformer.transform(scales.cpu().numpy())
            ).to(scales.device)
        
        return quantile_transformer_func
    
    def _build_transform(self) -> Compose:
        """Build the data transformation pipeline."""
        transform_cfg = [
            dict(type="NormalizeCoord"),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=self.grid_size,
                keys=("coord", "color", "normal", "origin_coord", "face_index"),
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "inverse", "origin_coord", "face_index"),
                feat_keys=("coord", "normal", "color"),
            ),
        ]
        return Compose(transform_cfg)
    
    def _generate_colors(self) -> List[np.ndarray]:
        """Generate a color palette for visualization."""
        hex_colors = list(mcolors.CSS4_COLORS.values())
        rgb_colors = np.array([
            mcolors.to_rgb(color) for color in hex_colors 
            if color not in ['#000000', '#FFFFFF']
        ])
        
        def relative_luminance(color):
            return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        
        rgb_colors = [
            color for color in rgb_colors 
            if 0.4 < relative_luminance(color) < 0.8
        ]
        np.random.shuffle(rgb_colors)
        
        return rgb_colors
    
    def preprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
    ) -> Tuple[dict, np.ndarray, np.ndarray]:
        """
        Preprocess mesh for inference.
        
        Args:
            mesh: Input trimesh object.
        
        Returns:
            Tuple of (processed_data_dict, pcd_inverse, face_index)
        """
        # Sample points from mesh surface
        coord, face_index, color = sample_surface(
            mesh, count=self.sample_num, sample_color=True
        )
        color = color[..., :3]
        
        # Get face normals
        face_normals = mesh.face_normals
        normal = face_normals[face_index]
        
        # Store original coordinates for reference
        origin_coord = coord.copy()
        
        # Create offset tensor
        offset = torch.tensor(coord.shape[0])
        
        # Create data dict
        obj = dict(
            coord=coord,
            normal=normal,
            color=color,
            offset=offset,
            origin_coord=origin_coord,
            face_index=face_index
        )
        
        # Apply transforms
        obj = self.transform(obj)
        
        # Extract inverse mapping and face index before removing them
        pcd_inverse = obj["inverse"].clone().numpy()
        processed_face_index = obj["face_index"].clone().numpy()
        
        # Remove keys not needed for model
        del obj["origin_coord"]
        del obj["face_index"]
        del obj["inverse"]
        
        return obj, pcd_inverse, processed_face_index
    
    @torch.no_grad()
    def extract_features(
        self,
        data_dict: dict,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Extract instance features at a given scale.
        
        Args:
            data_dict: Preprocessed data dictionary.
            scale: Grouping scale parameter (0.0 to 2.0).
        
        Returns:
            Instance features as numpy array.
        """
        # Move data to device
        input_dict = {"obj": data_dict, "scale": scale}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(self.device)
        
        data_dict["grid_size"] = self.grid_size
        
        # Run model in eval mode
        instance_feat = self.model(input_dict)
        
        return instance_feat.cpu().detach().numpy()
    
    def cluster_features(
        self,
        features: np.ndarray,
        coord: torch.Tensor,
        min_samples: int = 30,
        min_cluster_size: int = 30,
        cluster_selection_epsilon: float = 0.1,
    ) -> np.ndarray:
        """
        Cluster instance features using HDBSCAN.
        
        Args:
            features: Instance features to cluster.
            coord: Point coordinates for nearest neighbor assignment.
            min_samples: HDBSCAN min_samples parameter.
            min_cluster_size: HDBSCAN min_cluster_size parameter.
            cluster_selection_epsilon: HDBSCAN cluster_selection_epsilon.
        
        Returns:
            Cluster labels for each point.
        """
        clusterer = HDBSCAN(
            cluster_selection_epsilon=cluster_selection_epsilon,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            allow_single_cluster=False,
        )
        clusterer.fit(features)
        labels = clusterer.labels_
        
        # Handle noise points (-1 labels)
        invalid_label_mask = labels == -1
        if invalid_label_mask.sum() > 0:
            if invalid_label_mask.sum() == len(invalid_label_mask):
                # All points are noise - assign single cluster
                labels = np.zeros_like(labels)
            else:
                # Assign noise points to nearest valid cluster
                coord = coord.to(self.device).contiguous().float()
                valid_coord = coord[~invalid_label_mask]
                valid_offset = torch.tensor(valid_coord.shape[0]).to(self.device)
                invalid_coord = coord[invalid_label_mask]
                invalid_offset = torch.tensor(invalid_coord.shape[0]).to(self.device)
                
                indices, distances = pointops.knn_query(
                    1, valid_coord, valid_offset, invalid_coord, invalid_offset
                )
                indices = indices[:, 0].cpu().numpy()
                labels[invalid_label_mask] = labels[~invalid_label_mask][indices]
        
        return labels
    
    def mesh_voting(
        self,
        mesh: trimesh.Trimesh,
        labels: np.ndarray,
        face_index: np.ndarray,
    ) -> np.ndarray:
        """
        Transfer point labels to mesh faces via voting.
        
        Args:
            mesh: Input mesh.
            labels: Point labels.
            face_index: Face index for each sampled point.
        
        Returns:
            Face labels for the mesh.
        """
        num_faces = len(mesh.faces)
        num_labels = max(labels) + 1
        
        # Compute votes for each face
        votes = np.zeros((num_faces, num_labels), dtype=np.int32)
        np.add.at(votes, (face_index, labels), 1)
        
        # Find label with most votes for each face
        max_votes_labels = np.argmax(votes, axis=1)
        
        # Set label to -1 for faces with no votes
        max_votes_labels[np.all(votes == 0, axis=1)] = -1
        
        # Handle faces with no corresponding points
        valid_mask = max_votes_labels != -1
        if not valid_mask.all():
            face_centroids = mesh.triangles_center
            coord = torch.tensor(face_centroids).to(self.device).contiguous().float()
            valid_coord = coord[valid_mask]
            valid_offset = torch.tensor(valid_coord.shape[0]).to(self.device)
            invalid_coord = coord[~valid_mask]
            invalid_offset = torch.tensor(invalid_coord.shape[0]).to(self.device)
            
            indices, distances = pointops.knn_query(
                1, valid_coord, valid_offset, invalid_coord, invalid_offset
            )
            indices = indices[:, 0].cpu().numpy()
            max_votes_labels[~valid_mask] = max_votes_labels[valid_mask][indices]
        
        return max_votes_labels
    
    def colorize_mesh(
        self,
        mesh: trimesh.Trimesh,
        face_labels: np.ndarray,
    ) -> trimesh.Trimesh:
        """
        Apply colors to mesh faces based on labels.
        
        Args:
            mesh: Input mesh.
            face_labels: Label for each face.
        
        Returns:
            Colored mesh.
        """
        # Ensure mesh has color visual
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        
        # Assign colors
        for face_idx, label in enumerate(face_labels):
            color = self.colors[label % len(self.colors)]
            color_uint8 = (np.array(color) * 255).astype(np.uint8)
            color_with_alpha = np.append(color_uint8, 255)
            mesh.visual.face_colors[face_idx] = color_with_alpha
        
        return mesh
    
    def segment_mesh(
        self,
        mesh_path: Union[str, Path],
        scale: float = 1.0,
        scales_list: Optional[List[float]] = None,
        return_all_scales: bool = False,
    ) -> Tuple[trimesh.Trimesh, np.ndarray]:
        """
        Segment a mesh file.
        
        Args:
            mesh_path: Path to mesh file (GLB, OBJ, etc.).
            scale: Grouping scale (0.0 = fine, 2.0 = coarse).
            scales_list: Optional list of scales to evaluate.
            return_all_scales: If True, return results for all scales.
        
        Returns:
            Tuple of (segmented_mesh, face_labels).
            If return_all_scales is True, returns dict of results per scale.
        """
        # Load mesh
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        return self.segment_trimesh(
            mesh=mesh,
            scale=scale,
            scales_list=scales_list,
            return_all_scales=return_all_scales,
        )
    
    def segment_trimesh(
        self,
        mesh: trimesh.Trimesh,
        scale: float = 1.0,
        scales_list: Optional[List[float]] = None,
        return_all_scales: bool = False,
    ) -> Tuple[trimesh.Trimesh, np.ndarray]:
        """
        Segment a trimesh object.
        
        Args:
            mesh: Trimesh object to segment.
            scale: Grouping scale (0.0 = fine, 2.0 = coarse).
            scales_list: Optional list of scales to evaluate.
            return_all_scales: If True, return results for all scales.
        
        Returns:
            Tuple of (segmented_mesh, face_labels).
            If return_all_scales is True, returns dict of results per scale.
        """
        # Preprocess mesh
        data_dict, pcd_inverse, face_index = self.preprocess_mesh(mesh)
        
        if scales_list is None:
            scales_list = [scale]
        
        results = {}
        
        for s in scales_list:
            # Extract features
            features = self.extract_features(data_dict, scale=s)
            
            # Cluster
            labels = self.cluster_features(features, data_dict["coord"])
            
            # Map back through inverse
            labels_full = labels[pcd_inverse]
            
            # Transfer to mesh faces
            face_labels = self.mesh_voting(mesh, labels_full, face_index[pcd_inverse])
            
            # Colorize mesh
            mesh_copy = mesh.copy()
            colored_mesh = self.colorize_mesh(mesh_copy, face_labels)
            
            results[s] = {
                "mesh": colored_mesh,
                "face_labels": face_labels,
                "point_labels": labels_full,
                "num_segments": int(face_labels.max() + 1),
            }
            
            print(f"Scale {s}: {results[s]['num_segments']} segments")
        
        if return_all_scales:
            return results
        else:
            # Return result for the requested scale
            s = scale if scale in results else scales_list[0]
            return results[s]["mesh"], results[s]["face_labels"]


def main():
    """Test the inference module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAMPart3D Inference")
    parser.add_argument("mesh_path", type=str, help="Path to input mesh file")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--backbone-weight", type=str, default=None, help="Path to backbone weights")
    parser.add_argument("--model-weight", type=str, default=None, help="Path to model weights")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--scale", type=float, default=1.0, help="Segmentation scale")
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = SAMPart3DInference(
        config_path=args.config,
        backbone_weight_path=args.backbone_weight,
        model_weight_path=args.model_weight,
    )
    
    # Run segmentation
    mesh, labels = engine.segment_mesh(
        mesh_path=args.mesh_path,
        scale=args.scale,
    )
    
    # Save result
    output_path = args.output or args.mesh_path.replace(".glb", "_segmented.glb")
    mesh.export(output_path)
    print(f"Segmented mesh saved to: {output_path}")


if __name__ == "__main__":
    main()
