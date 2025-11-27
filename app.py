"""
Gradio UI for mesh segmentation.
Load a GLB file, segment it using SAMPart3D, and export the segmented result.
"""

import gradio as gr
import trimesh
import numpy as np
from pathlib import Path
import tempfile
from typing import Optional, Tuple
import os

from sampart3d_inference import SAMPart3DInference


# Initialize inference engine (lazy loaded)
inference_engine = None


def initialize_inference(
    config_path: Optional[str] = None,
    backbone_weight_path: Optional[str] = None,
    model_weight_path: Optional[str] = None,
) -> bool:
    """Initialize the inference engine if not already done."""
    global inference_engine
    
    if inference_engine is not None:
        return True
    
    try:
        print("Initializing SAMPart3D inference engine...")
        inference_engine = SAMPart3DInference(
            config_path=config_path,
            backbone_weight_path=backbone_weight_path,
            model_weight_path=model_weight_path,
        )
        return True
    except Exception as e:
        print(f"Failed to initialize inference engine: {e}")
        return False


def segment_mesh(
    mesh_file: Optional[str],
    config_path: Optional[str] = None,
    backbone_weight_path: Optional[str] = None,
    model_weight_path: Optional[str] = None,
    scale: float = 1.0,
) -> Tuple[Optional[str], str]:
    """
    Segment a mesh using SAMPart3D.
    
    Args:
        mesh_file: Path to input mesh file
        config_path: Path to config file (optional)
        backbone_weight_path: Path to backbone weights
        model_weight_path: Path to model weights
        scale: Scale parameter for segmentation
    
    Returns:
        Tuple of (output_file_path, status_message)
    """
    if mesh_file is None:
        return None, "❌ No mesh file provided"
    
    global inference_engine
    
    try:
        # Initialize inference engine if needed
        if not initialize_inference(config_path, backbone_weight_path, model_weight_path):
            return None, "❌ Failed to initialize inference engine"
        
        if inference_engine is None:
            return None, "❌ Inference engine not initialized"
        
        print(f"Processing mesh: {mesh_file}")
        
        # Run segmentation
        segmented_mesh, labels = inference_engine.segment_mesh(
            mesh_path=mesh_file,
            scale=scale,
        )
        
        if segmented_mesh is None or segmented_mesh.is_empty:
            return None, "❌ Segmentation produced empty result"
        
        print(f"Segmentation completed with {len(np.unique(labels))} segments")
        
        # Export result
        basename = Path(mesh_file).stem
        output_filename = f"{basename}_segmented.glb"
        output_dir = Path(tempfile.gettempdir()) / "segmentation_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        segmented_mesh.export(str(output_path))
        print(f"Segmented mesh exported to: {output_path}")
        
        status = f"✅ Segmentation complete!\nSegments: {len(np.unique(labels))}\nOutput: {output_path}"
        return str(output_path), status
        
    except Exception as e:
        print(f"Error during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""
    
    with gr.Blocks(title="Mesh Segmentation") as demo:
        gr.Markdown(
            """# 🧬 SAMPart3D Mesh Segmentation
            
Upload a GLB/OBJ mesh file and segment it using SAMPart3D."""
        )
        
        with gr.Row():
            # Input mesh
            with gr.Column():
                input_mesh = gr.Model3D(label="Input Mesh", display_mode="solid")
            
            # Output mesh
            with gr.Column():
                output_mesh = gr.Model3D(label="Segmented Mesh", display_mode="solid")
        
        # Configuration section (collapsed)
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                config_path = gr.Textbox(
                    label="Config Path",
                    placeholder="Optional: Path to config file",
                    value=""
                )
                backbone_weight = gr.Textbox(
                    label="Backbone Weight Path",
                    placeholder="Optional: Path to backbone weights",
                    value=""
                )
            with gr.Row():
                model_weight = gr.Textbox(
                    label="Model Weight Path",
                    placeholder="Optional: Path to model weights",
                    value=""
                )
                scale = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Segmentation Scale"
                )
        
        # Action buttons
        with gr.Row():
            segment_btn = gr.Button("🚀 Segment Mesh", variant="primary", size="lg")
        
        # Status output
        status_text = gr.Textbox(
            label="Status",
            interactive=False,
            value="Ready"
        )
        
        # Event handler
        segment_btn.click(
            segment_mesh,
            inputs=[
                input_mesh,
                config_path,
                backbone_weight,
                model_weight,
                scale,
            ],
            outputs=[output_mesh, status_text]
        )
    
    return demo


if __name__ == "__main__":
    print("🚀 Launching Mesh Segmentation UI...")
    try:
        demo = build_ui()
        demo.launch(share=False, server_name="0.0.0.0")
    except Exception as e:
        print(f"❌ Failed to launch UI: {e}")

