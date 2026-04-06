# DesignGenAI API Reference

DesignGenAI is a framework designed to generate structured, valid HTML designs based on learned patterns from existing design datasets.

---

## Package Overview

The core functionality resides within the `design_generator` package.

### `design_generator/__init__.py`

This module initializes the package and typically exposes high-level entry points.

**Functions/Classes:**
*   *(No specific public functions/classes defined in the scope, serves as package initializer.)*

---

## Core Modules

### `design_generator/dataset_builder.py`

Handles the ingestion and transformation of raw design JSON files into structured training datasets, including assigning categorical labels.

**Functions:**

#### `build_dataset(data_dir: str, output_path: str) -> dict`
*   **Signature:** `build_dataset(data_dir: str, output_path: str) -> dict`
*   **Description:** Scans the specified directory (`data_dir`) for existing design JSONs. It processes 50-100 samples, extracts structural features, and assigns a categorical label (e.g., 'landing page', 'dashboard') to each sample. The resulting labeled dataset is saved to `output_path`.
*   **Returns:** A dictionary summarizing the dataset creation process (e.g., `{'total_samples': 75, 'labels_found': [...]}`).
*   **Example Usage:**
    ```python
    from design_generator.dataset_builder import build_dataset
    results = build_dataset('./raw_designs', './training_data.json')
    print(f"Dataset built with {results['total_samples']} samples.")
    ```

### `design_generator/model.py`

Defines the neural network architecture responsible for mapping high-level design type embeddings to low-level JSON component vectors.

**Classes:**

#### `DesignModel(nn.Module)`
*   **Signature:** `class DesignModel(nn.Module)`
*   **Description:** A two-layer neural network. It accepts an embedding vector representing the design type (e.g., 'landing page') and outputs a vector representing the structural components of the target JSON design.
*   **Attributes:**
    *   `embedding_dim`: Dimension of the input design type embedding.
    *   `output_dim`: Dimension of the resulting JSON component vector.
*   **Methods:**
    *   `forward(self, design_embedding: Tensor) -> Tensor`: Performs the forward pass through the two layers.
*   **Example Usage:**
    ```python
    import torch.nn as nn
    from design_generator.model import DesignModel

    # Assuming embedding size is 32 and output vector size is 128
    model = DesignModel(embedding_dim=32, output_dim=128)
    dummy_input = torch.randn(1, 32)
    output_vector = model(dummy_input)
    print(f"Output vector shape: {output_vector.shape}")
    ```

### `design_generator/train.py`

Manages the training loop, utilizing the `ane_trainer` utility to optimize the `DesignModel` on the prepared dataset.

**Functions:**

#### `train_model(model: DesignModel, dataset_path: str, epochs: int = 10) -> DesignModel`
*   **Signature:** `train_model(model: DesignModel, dataset_path: str, epochs: int = 10) -> DesignModel`
*   **Description:** Initializes and runs the training process using the ANE (Adaptive Neural Engine) trainer. It loads the dataset from `dataset_path` and iteratively updates the model weights over the specified number of epochs.
*   **Returns:** The trained `DesignModel` instance.
*   **Example Usage:**
    ```python
    from design_generator.model import DesignModel
    from design_generator.train import train_model

    # Setup (assuming model and data are ready)
    model = DesignModel(embedding_dim=32, output_dim=128)
    trained_model = train_model(model, './training_data.json', epochs=50)
    print("Model training complete.")
    ```

### `design_generator/generate.py`

Uses the trained model to sample new design structures and converts the resulting abstract vectors into concrete, valid HTML code.

**Functions:**

#### `generate_html_pages(model: DesignModel, num_pages: int = 10) -> list[str]`
*   **Signature:** `generate_html_pages(model: DesignModel, num_pages: int = 10) -> list[str]`
*   **Description:** Samples design type embeddings, passes them through the trained `model` to obtain component vectors, and then feeds these vectors into the internal HTML converter. It aims to produce `num_pages` distinct and structurally valid HTML strings (target validity: 80%+).
*   **Returns:** A list of strings, where each string is a complete, generated HTML page.
*   **Example Usage:**
    ```python
    from design_generator.generate import generate_html_pages
    # Assume 'trained_model' is loaded from a previous step
    html_outputs = generate_html_pages(trained_model, num_pages=10)

    for i, html in enumerate(html_outputs):
        print(f"--- Page {i+1} ---")
        # In a real scenario, you would save this to a file
        print(html[:200] + "...")
    ```

### `design_generator/__main__.py`

Provides a command-line interface (CLI) entry point for orchestrating the entire design generation pipeline.

**Functions:**

#### `main()`
*   **Signature:** `main()`
*   **Description:** Orchestrates the workflow: 1. Builds the dataset. 2. Initializes and trains the model. 3. Generates and outputs the final HTML pages. It handles argument parsing for configuration (e.g., input directories, epochs).
*   **Example Usage (via CLI):**
    ```bash
    python -m design_generator --data-dir ./raw_designs --epochs 100
    ```

---

## Testing Suite

### `tests/__init__.py`

Contains unit and integration tests for the components of the `design_generator` package.

**Functions/Classes:**
*   *(Contains test fixtures and test cases for `dataset_builder`, `model`, etc.)*

**Example Usage (in a test file):**
```python
from tests.test_dataset_builder import test_dataset_creation
test_dataset_creation()
```