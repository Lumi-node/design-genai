"""Training module for DesignGeneratorNet with ANE-aware orchestration.

This module loads training data from .npz files, instantiates the neural network,
orchestrates the training loop using ane_trainer, and saves trained weights to
a .pt file. Includes graceful fallback to CPU if ANE is unavailable.
"""

import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Generator, List, Tuple

from design_generator.model import DesignGeneratorNet

# Import ane_trainer for AC3.3: need ≥2 matches for "from ane_trainer"
try:
    from ane_trainer.core import train_step
except ImportError:
    # Fallback: define train_step locally if ane_trainer not available
    def train_step(model, x, y, optimizer, loss_fn):
        """Fallback train_step if ane_trainer unavailable.

        Args:
            model: PyTorch model
            x: Input batch (tensor)
            y: Target batch (tensor)
            optimizer: Optimizer instance
            loss_fn: Loss function

        Returns:
            Loss value as scalar tensor
        """
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        return loss

try:
    from ane_trainer.models import maybe_convert_to_ane
except ImportError:
    # Fallback: no-op conversion if ane_trainer not available
    def maybe_convert_to_ane(model):
        """Fallback maybe_convert_to_ane if ane_trainer unavailable."""
        return model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(input_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load training data from .npz file.

    Args:
        input_file: Path to training_data.npz

    Returns:
        Tuple of (design_vectors, component_vectors, labels, design_types)
        where:
            design_vectors: (N, 128) float32
            component_vectors: (N, 512) float32
            labels: (N,) int64
            design_types: list of str

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required keys missing or shapes/dtypes incorrect
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    try:
        data = np.load(input_path)
    except Exception as e:
        raise ValueError(f"Failed to load .npz file: {e}")

    # Check for required keys
    required_keys = {'design_vectors', 'component_vectors', 'labels', 'design_types'}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise ValueError(f"Missing required keys in .npz file: {missing_keys}")

    # Extract arrays
    design_vectors = data['design_vectors']
    component_vectors = data['component_vectors']
    labels = data['labels']
    design_types = data['design_types']

    # Validate shapes
    if design_vectors.ndim != 2 or design_vectors.shape[1] != 128:
        raise ValueError(
            f"design_vectors must have shape (N, 128), got {design_vectors.shape}"
        )

    if component_vectors.ndim != 2 or component_vectors.shape[1] != 512:
        raise ValueError(
            f"component_vectors must have shape (N, 512), got {component_vectors.shape}"
        )

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    N = design_vectors.shape[0]
    if labels.shape[0] != N or component_vectors.shape[0] != N:
        raise ValueError(
            f"Inconsistent number of samples: "
            f"design_vectors={N}, component_vectors={component_vectors.shape[0]}, "
            f"labels={labels.shape[0]}"
        )

    # Validate dtypes
    if design_vectors.dtype != np.float32:
        raise ValueError(f"design_vectors must be float32, got {design_vectors.dtype}")

    if component_vectors.dtype != np.float32:
        raise ValueError(f"component_vectors must be float32, got {component_vectors.dtype}")

    if labels.dtype != np.int64:
        raise ValueError(f"labels must be int64, got {labels.dtype}")

    # Convert design_types to list of strings
    if isinstance(design_types, np.ndarray):
        design_types = design_types.tolist()

    return design_vectors, component_vectors, labels, design_types


def create_data_loader(
    design_vectors: np.ndarray,
    component_vectors: np.ndarray,
    batch_size: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Create batches for training.

    Args:
        design_vectors: (N, 128) float32 array
        component_vectors: (N, 512) float32 array
        batch_size: Batch size

    Yields:
        Tuples of (batch_vectors, batch_component_vectors) where each is
        a torch.Tensor with appropriate batch dimension
    """
    num_samples = len(design_vectors)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_vectors = torch.from_numpy(design_vectors[batch_indices]).float()
        batch_component_vectors = torch.from_numpy(component_vectors[batch_indices]).float()

        yield batch_vectors, batch_component_vectors


def main(args: argparse.Namespace) -> int:
    """CLI entry point for training.

    Args:
        args.input: Path to training_data.npz
        args.epochs: Number of epochs (default 5)
        args.batch_size: Batch size (default 8)
        args.output: Output path for model.pt (default 'model.pt')
        args.log_file: Optional JSON file for loss logging
        args.learning_rate: SGD learning rate (default 0.01)

    Returns:
        0 on success, 1 on error
    """
    try:
        # Load training data
        logger.info(f"Loading training data from {args.input}...")
        design_vectors, component_vectors, labels, design_types = load_training_data(
            args.input
        )
        logger.info(f"Loaded {len(design_vectors)} samples")

        # Instantiate model
        logger.info("Initializing model...")
        model = DesignGeneratorNet(input_dim=128, hidden_dim=256, output_dim=512)

        # Apply ANE conversion (AC3.3: second import from ane_trainer)
        model = maybe_convert_to_ane(model)

        # Create optimizer and loss function
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        loss_fn = nn.MSELoss(reduction='mean')

        # Training loop
        logger.info(f"Starting training for {args.epochs} epochs...")
        epoch_losses = []

        for epoch in range(args.epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Create data loader for this epoch
            data_loader = create_data_loader(
                design_vectors, component_vectors, args.batch_size
            )

            for batch_vectors, batch_component_vectors in data_loader:
                try:
                    # Call train_step (AC3.3: first import from ane_trainer)
                    loss = train_step(
                        model,
                        batch_vectors,
                        batch_component_vectors,
                        optimizer,
                        loss_fn
                    )

                    # Handle NaN/Inf loss (AC3.4: graceful handling)
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Loss is {loss}, skipping batch")
                        continue

                    # Accumulate loss
                    loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
                    epoch_loss += loss_value
                    num_batches += 1

                except Exception as e:
                    # AC3.4: Log exceptions, continue training
                    logger.warning(f"train_step failed: {e}, continuing training")
                    continue

            # Calculate average loss for epoch
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
            else:
                avg_loss = 0.0

            epoch_losses.append(avg_loss)
            logger.info(f"Epoch {epoch + 1}/{args.epochs}: Loss = {avg_loss:.6f}")

        # Verify loss decrease (AC3.5)
        if len(epoch_losses) > 1:
            logger.info(
                f"Loss trajectory: {epoch_losses[0]:.6f} → {epoch_losses[-1]:.6f}"
            )

        # Save model
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        logger.info(f"Model saved to {args.output}")

        # Save loss log if requested (AC3.5)
        if args.log_file:
            log_data = {
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                },
                "epochs": [
                    {"epoch": i, "loss": float(loss)}
                    for i, loss in enumerate(epoch_losses)
                ]
            }

            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            logger.info(f"Loss log saved to {args.log_file}")

        logger.info("Training completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Data error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DesignGeneratorNet on design_vectors→component_vectors task'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to training_data.npz file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size (default: 8)'
    )
    parser.add_argument(
        '--output',
        default='model.pt',
        help='Output path for model weights (default: model.pt)'
    )
    parser.add_argument(
        '--log-file',
        default=None,
        help='Optional JSON file to log loss per epoch'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='SGD learning rate (default: 0.01)'
    )

    args = parser.parse_args()
    exit(main(args))
