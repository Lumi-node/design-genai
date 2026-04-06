# 🚀 DesignGenAI Quick Start Guide

Welcome to DesignGenAI! This guide will get you up and running quickly to train a design generation model and generate high-quality HTML outputs.

DesignGenAI is a package designed to learn design patterns from existing JSON blueprints and generate new, structurally valid HTML pages.

## Prerequisites

Ensure you have the package installed:

```bash
pip install design_genai
```

## Core Workflow Overview

The typical workflow involves four main steps:

1. **Data Preparation:** Convert raw design JSONs into labeled training data.
2. **Model Definition:** Define the neural network architecture.
3. **Training:** Train the model using the prepared data.
4. **Generation:** Sample from the trained model to produce new designs, which are then converted to HTML.

---

## 🛠️ Usage Examples

Here are three practical examples demonstrating the core functionalities of DesignGenAI.

### Example 1: Building the Training Dataset

Before training, you must convert your collection of existing design JSONs into a structured training dataset, assigning a label (e.g., 'landing page') to each design.

This uses the `dataset_builder.py` module.

```python
from design_generator.dataset_builder import DatasetBuilder

# Assume 'path/to/your/designs/' contains 50-100 design JSON files
input_dir = "path/to/your/designs/"
output_file = "training_data.jsonl"
design_type_label = "landing page"

builder = DatasetBuilder(input_dir)

# Process and build the dataset
print(f"Building dataset for type: {design_type_label}...")
builder.build_dataset(label=design_type_label, output_path=output_file)

print(f"✅ Dataset successfully built and saved to {output_file}")
```

### Example 2: Training the Design Model

Once you have your training data (e.g., `training_data.jsonl`), you can train the model using the `train.py` module, which leverages the internal `ane_trainer`.

```python
from design_generator.train import train_model

# Paths defined in the previous step
training_data_path = "training_data.jsonl"
model_save_path = "trained_design_model.pth"

print("🚀 Starting model training...")

# Train the model using the ANE trainer backend
success = train_model(
    data_path=training_data_path,
    model_output_path=model_save_path,
    epochs=50,  # Adjust epochs based on convergence needs
    batch_size=32
)

if success:
    print(f"🎉 Model training complete! Saved to {model_save_path}")
else:
    print("❌ Model training failed.")
```

### Example 3: Generating and Validating HTML Pages

With a trained model, you can now generate new design structures and convert them into usable HTML using `generate.py`.

```python
from design_generator.generate import DesignGenerator

# Load the model trained in Example 2
model_path = "trained_design_model.pth"
generator = DesignGenerator(model_path=model_path)

num_pages_to_generate = 10
generated_html_files = []

print(f"✨ Generating {num_pages_to_generate} distinct design blueprints...")

for i in range(num_pages_to_generate):
    # Sample a new design structure from the trained model
    design_json = generator.sample_design()
    
    # Convert the generated JSON structure into a valid HTML string
    html_output = generator.to_html(design_json)
    
    # Save the result
    filename = f"generated_design_{i+1}.html"
    with open(filename, 'w') as f:
        f.write(html_output)
    
    generated_html_files.append(filename)

print("\n=====================================================")
print(f"✅ Success! Generated {len(generated_html_files)} HTML files.")
print("Check the generated files for structural validity (80%+ target).")
print("=====================================================")
```

---

## 📚 Module Reference

| Module | Primary Function | Description |
| :--- | :--- | :--- |
| `dataset_builder.py` | `DatasetBuilder` | Handles ingestion of raw JSONs and conversion into labeled training samples. |
| `model.py` | (Internal) | Defines the 2-layer network mapping embeddings to component vectors. |
| `train.py` | `train_model` | Orchestrates the training process using the ANE trainer on the prepared dataset. |
| `generate.py` | `DesignGenerator` | Loads the trained model, samples new designs, and converts the resulting JSON structure to HTML. |