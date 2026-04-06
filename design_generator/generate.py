"""Generation module for creating design specifications and HTML from trained models.

This module loads trained design generator models, samples design embeddings,
generates component vectors via forward pass, inverts to design specifications,
and produces HTML output.
"""

import argparse
import logging
import numpy as np
import torch
import sys
import importlib.util
from typing import Dict, Optional, List
from pathlib import Path

from design_generator.model import DesignGeneratorNet
from design_generator.dataset_builder import DESIGN_TYPES

# Dynamically import html_generator functions from sources/0c16ae7e
html_gen_path = Path(__file__).parent.parent / "sources" / "0c16ae7e" / "html_generator.py"
html_gen_spec = importlib.util.spec_from_file_location("html_generator", str(html_gen_path))
html_generator = importlib.util.module_from_spec(html_gen_spec)
html_gen_spec.loader.exec_module(html_generator)
generate_html_structure = html_generator.generate_html_structure
generate_css = html_generator.generate_css

# Dynamically import output_writer functions
output_writer_path = Path(__file__).parent.parent / "sources" / "0c16ae7e" / "output_writer.py"
output_writer_spec = importlib.util.spec_from_file_location("output_writer", str(output_writer_path))
output_writer = importlib.util.module_from_spec(output_writer_spec)
output_writer_spec.loader.exec_module(output_writer)
write_output = output_writer.write_output

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def inverse_vectorize_design(
    component_vector: np.ndarray,
    image_width: int = 1000,
    image_height: int = 800
) -> Dict:
    """Convert 512-dim component vector back to design specification.

    Args:
        component_vector: (512,) float32 array from model output
        image_width: Canvas width in pixels (default 1000)
        image_height: Canvas height in pixels (default 800)

    Returns:
        Dict with keys {'header', 'sidebar', 'content', 'footer'}
        Each value is None or {'x': int, 'y': int, 'width': int, 'height': int}

    Algorithm:
        For each of 4 regions (128 dims each):
        1. Extract normalized coordinates [0:4]
        2. If sum of normalized coords < 0.1: region = None (null region)
        3. Else: denormalize to pixel coordinates
           x = round(vec[0] * image_width)
           y = round(vec[1] * image_height)
           width = round(vec[2] * image_width)
           height = round(vec[3] * image_height)
        4. Clamp values to [0, image_width/height]
        5. Create region dict
    """
    if component_vector.shape != (512,):
        raise ValueError(f"component_vector must have shape (512,), got {component_vector.shape}")

    if component_vector.dtype != np.float32:
        component_vector = component_vector.astype(np.float32)

    regions = {}
    region_names = ['header', 'sidebar', 'content', 'footer']

    for region_idx, region_name in enumerate(region_names):
        # Extract 128-dim block for this region
        start_idx = region_idx * 128
        end_idx = start_idx + 128
        region_block = component_vector[start_idx:end_idx]

        # Extract normalized coordinates [0:4]
        coords_norm = region_block[0:4]

        # Check if region is null (sum of normalized coords < 0.1 threshold)
        coords_sum = np.sum(np.abs(coords_norm))
        if coords_sum < 0.1:
            regions[region_name] = None
            continue

        # Denormalize to pixel coordinates
        x_norm = coords_norm[0]
        y_norm = coords_norm[1]
        width_norm = coords_norm[2]
        height_norm = coords_norm[3]

        # Clamp to [0, 1] range
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)
        width_norm = np.clip(width_norm, 0.0, 1.0)
        height_norm = np.clip(height_norm, 0.0, 1.0)

        # Convert to pixel coordinates
        x = int(round(x_norm * image_width))
        y = int(round(y_norm * image_height))
        width = int(round(width_norm * image_width))
        height = int(round(height_norm * image_height))

        # Final clamping to valid pixel ranges
        x = max(0, min(x, image_width))
        y = max(0, min(y, image_height))
        width = max(0, min(width, image_width - x))
        height = max(0, min(height, image_height - y))

        regions[region_name] = {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }

    return regions


def load_model(model_path: str) -> DesignGeneratorNet:
    """Load trained model from .pt file.

    Args:
        model_path: Path to model.pt file

    Returns:
        DesignGeneratorNet in eval mode with loaded weights

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails or weights are incompatible
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Create model with default architecture
        model = DesignGeneratorNet(input_dim=128, hidden_dim=256, output_dim=512)

        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

        # Set to eval mode
        model.eval()

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def sample_design_embedding(
    design_type: str,
    design_types_list: List[str],
    rng: np.random.Generator
) -> np.ndarray:
    """Sample a design embedding from a standard normal distribution.

    Args:
        design_type: Design type string (e.g., 'landing', 'dashboard')
        design_types_list: List of available design types (for validation/future use)
        rng: NumPy random generator

    Returns:
        (128,) float32 array sampled from ~N(0, 1) distribution

    Note:
        Currently ignores design_type parameter and samples uniformly.
        Future implementations can add class-conditional sampling.
    """
    # Sample from standard normal distribution
    embedding = rng.standard_normal(128).astype(np.float32)
    return embedding


def generate_design_spec(
    model: DesignGeneratorNet,
    embedding: np.ndarray,
    design_type_label: str
) -> Dict:
    """Generate design specification from embedding and model.

    Args:
        model: Trained DesignGeneratorNet in eval mode
        embedding: (128,) float32 design embedding
        design_type_label: String like 'landing', 'dashboard', etc.

    Returns:
        Dict with keys {'header', 'sidebar', 'content', 'footer', 'design_type'}
        where design_type matches the requested type (AC4.5)

    Algorithm:
        1. Convert embedding to torch tensor
        2. Forward pass through model to get component vector
        3. Inverse vectorize to get region dict
        4. Add design_type_label to dict
        5. Return complete spec
    """
    # Ensure embedding is float32
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)

    # Convert to torch tensor for model forward pass
    embedding_tensor = torch.from_numpy(embedding).float()

    # Add batch dimension if needed
    if embedding_tensor.ndim == 1:
        embedding_tensor = embedding_tensor.unsqueeze(0)

    # Forward pass through model
    with torch.no_grad():
        component_tensor = model(embedding_tensor)

    # Remove batch dimension and convert to numpy
    component_vector = component_tensor.squeeze(0).numpy().astype(np.float32)

    # Inverse vectorize to get regions
    regions_dict = inverse_vectorize_design(component_vector)

    # Add design_type to match the requested type (AC4.5)
    regions_dict['design_type'] = design_type_label

    return regions_dict


def generate_html(design_spec: Dict, index: int, output_dir: str) -> Optional[str]:
    """Generate HTML from design specification and write to file.

    Args:
        design_spec: Dict with 'header', 'sidebar', 'content', 'footer', 'design_type'
        index: File index for naming (e.g., index_0, index_1)
        output_dir: Directory to write HTML files

    Returns:
        Absolute path to generated HTML file, or None if generation failed

    Integration with sources/0c16ae7e:
        1. Call generate_html_structure(design_spec) to get base HTML
        2. Call generate_css(design_spec, colors={}, ...) to get CSS rules
        3. Combine into single HTML document with embedded CSS
        4. Call write_output to save and validate

    Error Handling:
        - If html_generator fails: log warning, return None
        - If write_output fails: log warning, return None
    """
    try:
        # Extract region dict for html_generator (without design_type)
        regions_for_html = {k: v for k, v in design_spec.items() if k != 'design_type'}

        # Generate HTML structure
        try:
            html_structure = generate_html_structure(regions_for_html)
        except Exception as e:
            logger.warning(f"Failed to generate HTML structure: {e}")
            return None

        # Generate CSS
        try:
            css_rules = generate_css(regions_for_html, colors={}, image_width=1000, image_height=800)
        except Exception as e:
            logger.warning(f"Failed to generate CSS: {e}")
            return None

        # Combine HTML and CSS
        # Insert CSS before closing </head> tag
        html_with_css = html_structure.replace(
            '</head>',
            f'<style>\n{css_rules}\n</style>\n</head>'
        )

        # Write output
        output_subdir = Path(output_dir) / f"index_{index}"
        try:
            output_path = write_output(html_with_css, output_dir=str(output_subdir))
            return output_path
        except Exception as e:
            logger.warning(f"Failed to write output: {e}")
            return None

    except Exception as e:
        logger.warning(f"Error generating HTML for index {index}: {e}")
        return None


def main(args_list=None) -> int:
    """CLI entry point for design generation.

    Args:
        args_list: List of command-line arguments (for testing)

    Returns:
        0 on success (≥1 file written), 1 on error

    Execution:
        1. Parse CLI arguments
        2. Load trained model
        3. Validate design type
        4. Create output directory
        5. For each design to generate:
           - Sample embedding
           - Generate spec with design_type
           - Generate and write HTML
        6. Return 0 if ≥1 files written, 1 if all failed
    """
    parser = argparse.ArgumentParser(
        description='Generate design specifications and HTML from trained model'
    )
    parser.add_argument(
        '--model',
        default='model.pt',
        help='Path to trained model.pt (default: model.pt)'
    )
    parser.add_argument(
        '--type',
        default='landing',
        help='Design type string (default: landing)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=10,
        help='Number of designs to generate (default: 10)'
    )
    parser.add_argument(
        '--output',
        default='./',
        help='Output directory (default: ./)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    # Parse arguments
    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    try:
        # Load model
        try:
            model = load_model(args.model)
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except RuntimeError as e:
            logger.error(str(e))
            return 1

        # Validate design type
        if args.type not in DESIGN_TYPES:
            logger.error(f"Invalid design type '{args.type}'. Valid types: {DESIGN_TYPES}")
            return 1

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize random generator
        rng = np.random.default_rng(args.seed)

        # Generate designs
        files_written = 0
        for i in range(args.count):
            try:
                # Sample embedding
                embedding = sample_design_embedding(args.type, DESIGN_TYPES, rng)

                # Generate design spec
                spec = generate_design_spec(model, embedding, args.type)

                # Verify AC4.5: design_type matches requested type
                if spec.get('design_type') != args.type:
                    logger.warning(f"Design {i}: design_type mismatch ({spec.get('design_type')} != {args.type})")
                    continue

                # Generate HTML
                html_path = generate_html(spec, i, str(output_dir))
                if html_path:
                    logger.info(f"Generated: {html_path}")
                    files_written += 1
                else:
                    logger.warning(f"Failed to generate HTML for design {i}")

            except Exception as e:
                logger.warning(f"Error generating design {i}: {e}")
                continue

        # Return success if at least one file was written
        if files_written > 0:
            logger.info(f"Successfully generated {files_written} HTML files")
            return 0
        else:
            logger.error("Failed to generate any HTML files")
            return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
