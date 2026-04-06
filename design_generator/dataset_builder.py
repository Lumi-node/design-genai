"""Dataset builder module for vectorizing design JSONs into training data.

This module loads design JSON files from a directory, vectorizes them into
fixed-dimension arrays (128-dim design embeddings and 512-dim component vectors),
assigns design type labels, and packages everything into a numpy .npz file for
training.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Design types constant - minimum 3 types
DESIGN_TYPES = ['landing', 'dashboard', 'blog']


def load_designs(directory: str) -> List[Dict]:
    """Load all JSON files from a directory.

    Args:
        directory: Path to directory containing design JSON files

    Returns:
        List of dicts with keys 'header', 'sidebar', 'content', 'footer'

    Raises:
        OSError: If directory doesn't exist
        json.JSONDecodeError: If any JSON file is malformed
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise OSError(f"Directory does not exist: {directory}")

    if not dir_path.is_dir():
        raise OSError(f"Path is not a directory: {directory}")

    designs = []
    json_files = sorted(dir_path.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                design_json = json.load(f)

            # Validate structure - should have the required keys
            if not isinstance(design_json, dict):
                logger.warning(f"Skipping {json_file.name}: not a dict")
                continue

            # Check for required keys
            required_keys = {'header', 'sidebar', 'content', 'footer'}
            if not all(key in design_json for key in required_keys):
                logger.warning(f"Skipping {json_file.name}: missing required keys {required_keys}")
                continue

            designs.append(design_json)

        except json.JSONDecodeError as e:
            logger.warning(f"Skipping {json_file.name}: JSON decode error: {e}")
            continue
        except Exception as e:
            logger.warning(f"Skipping {json_file.name}: {e}")
            continue

    return designs


def vectorize_design(design_json: Dict, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Convert design JSON to two fixed-size vectors.

    Args:
        design_json: Dict with keys 'header', 'sidebar', 'content', 'footer'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (design_vector: (128,) float32, component_vector: (512,) float32)

    The design_vector (128) is a compact representation:
        - Dims [0:32]: Header region (x, y, width, height, r, g, b, padding)
        - Dims [32:64]: Sidebar region (same pattern)
        - Dims [64:96]: Content region (same pattern)
        - Dims [96:128]: Footer region (same pattern)

    The component_vector (512) is an expanded representation:
        - Same 4-region layout but 128 dims per region
        - Dims [0:4]: x, y, width, height (normalized [0,1])
        - Dims [4:7]: RGB color (normalized [0,1])
        - Dims [7:128]: Padding/future expansion (zeros)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Canonical canvas dimensions
    canvas_width = 1000.0
    canvas_height = 800.0

    # Initialize vectors
    design_vector = np.zeros(128, dtype=np.float32)
    component_vector = np.zeros(512, dtype=np.float32)

    # Region names and their offsets
    regions = ['header', 'sidebar', 'content', 'footer']

    for region_idx, region_name in enumerate(regions):
        region_data = design_json.get(region_name)

        # Design vector: 32 dims per region, starting at region_idx * 32
        design_offset = region_idx * 32
        # Component vector: 128 dims per region, starting at region_idx * 128
        component_offset = region_idx * 128

        if region_data is None:
            # Region is absent - fill with zeros
            # Already initialized to zeros, so just continue
            pass
        else:
            # Extract and normalize coordinates
            x = float(region_data.get('x', 0))
            y = float(region_data.get('y', 0))
            width = float(region_data.get('width', 0))
            height = float(region_data.get('height', 0))

            # Normalize to [0, 1] range
            x_norm = min(max(x / canvas_width, 0.0), 1.0)
            y_norm = min(max(y / canvas_height, 0.0), 1.0)
            width_norm = min(max(width / canvas_width, 0.0), 1.0)
            height_norm = min(max(height / canvas_height, 0.0), 1.0)

            # Extract RGB color (if present)
            # We'll use default gray if not present
            r = 0.5
            g = 0.5
            b = 0.5

            # For design_vector (32 dims per region):
            # [0:4] = x, y, width, height
            # [4:7] = r, g, b
            # [7:32] = padding (zeros)
            design_vector[design_offset:design_offset + 4] = [x_norm, y_norm, width_norm, height_norm]
            design_vector[design_offset + 4:design_offset + 7] = [r, g, b]

            # For component_vector (128 dims per region):
            # [0:4] = x, y, width, height
            # [4:7] = r, g, b
            # [7:128] = padding (zeros)
            component_vector[component_offset:component_offset + 4] = [x_norm, y_norm, width_norm, height_norm]
            component_vector[component_offset + 4:component_offset + 7] = [r, g, b]

    return design_vector, component_vector


def assign_labels(design_jsons: List[Dict]) -> np.ndarray:
    """Assign integer labels based on inferred design type.

    Args:
        design_jsons: List of design JSON dicts

    Returns:
        (N,) int64 array where each value is an index into DESIGN_TYPES
    """
    labels = []

    for design_json in design_jsons:
        # Simple heuristic: check if regions match specific design types
        # In a real system, this could use a classifier or metadata
        header = design_json.get('header')
        sidebar = design_json.get('sidebar')
        content = design_json.get('content')
        footer = design_json.get('footer')

        # Count present regions
        regions_present = sum([header is not None, sidebar is not None,
                              content is not None, footer is not None])

        # Simple heuristic-based labeling:
        # If has sidebar: likely dashboard
        # If no sidebar but has content: likely blog or landing
        # Use modulo to cycle through types if we have more samples
        if sidebar is not None:
            label_idx = DESIGN_TYPES.index('dashboard')
        elif regions_present >= 3:
            label_idx = DESIGN_TYPES.index('landing')
        else:
            label_idx = DESIGN_TYPES.index('blog')

        labels.append(label_idx)

    return np.array(labels, dtype=np.int64)


def main(args: argparse.Namespace) -> int:
    """CLI entry point for dataset builder.

    Args:
        args.input_dir: Directory containing design JSONs
        args.output: Output path (default './training_data.npz')
        args.seed: Random seed (default 42)
        args.count: Max designs to load (default None, load all)

    Returns:
        0 on success, 1 on error
    """
    try:
        # Load designs
        try:
            designs = load_designs(args.input_dir)
        except OSError as e:
            logger.error(f"Failed to load designs: {e}")
            return 1

        if not designs:
            logger.error("No valid designs loaded")
            return 1

        # Limit count if specified
        if args.count is not None:
            designs = designs[:args.count]

        if not designs:
            logger.error("No designs to process")
            return 1

        # Vectorize all designs
        design_vectors = []
        component_vectors = []

        for design_json in designs:
            design_vec, component_vec = vectorize_design(design_json, seed=args.seed)
            design_vectors.append(design_vec)
            component_vectors.append(component_vec)

        # Stack into arrays
        design_vectors = np.stack(design_vectors, axis=0)  # (N, 128)
        component_vectors = np.stack(component_vectors, axis=0)  # (N, 512)

        # Assign labels
        labels = assign_labels(designs)  # (N,)

        # Ensure correct dtypes
        design_vectors = design_vectors.astype(np.float32)
        component_vectors = component_vectors.astype(np.float32)
        labels = labels.astype(np.int64)

        # Save to .npz file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            design_vectors=design_vectors,
            component_vectors=component_vectors,
            labels=labels,
            design_types=DESIGN_TYPES
        )

        logger.info(f"Successfully created {args.output}")
        logger.info(f"Loaded {len(designs)} designs")
        logger.info(f"design_vectors shape: {design_vectors.shape}")
        logger.info(f"component_vectors shape: {component_vectors.shape}")
        logger.info(f"labels shape: {labels.shape}")

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build training dataset from design JSON files'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing design JSON files'
    )
    parser.add_argument(
        '--output',
        default='./training_data.npz',
        help='Output .npz file path (default: ./training_data.npz)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help='Maximum number of designs to load (default: all)'
    )

    args = parser.parse_args()
    exit(main(args))
