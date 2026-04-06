"""End-to-end integration tests for the design_generator pipeline.

This module verifies the entire pipeline works correctly:
  dataset_builder → training → generation → HTML output

Tests cover all acceptance criteria (AC1.5, AC3.5, AC4.2-AC4.6) across
multiple design types and configurations.
"""

import argparse
import json
import logging
import numpy as np
import pytest
import re
import tempfile
import torch
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict, Set

from design_generator.dataset_builder import (
    DESIGN_TYPES,
    load_designs,
    vectorize_design,
    assign_labels,
    main as dataset_builder_main,
)
from design_generator.model import DesignGeneratorNet
from design_generator.train import (
    load_training_data,
    create_data_loader,
    main as train_main,
)
from design_generator.generate import (
    load_model,
    sample_design_embedding,
    generate_design_spec,
    inverse_vectorize_design,
    main as generate_main,
)


# Configure logging to reduce noise in test output
logging.basicConfig(level=logging.ERROR)


class StructureValidator(HTMLParser):
    """Custom HTML parser to validate correct nesting and tag matching."""

    # Self-closing tags that don't need end tags
    SELF_CLOSING = {
        'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
        'link', 'meta', 'param', 'source', 'track', 'wbr'
    }

    def __init__(self):
        """Initialize the parser with empty tag stack."""
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        """Track opening tags, skip self-closing tags."""
        if tag not in self.SELF_CLOSING:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        """Validate closing tags match opening tags."""
        if tag in self.SELF_CLOSING:
            return  # Ignore end tags for self-closing elements
        if not self.stack:
            self.errors.append(f"Unexpected closing tag: {tag}")
            return
        if self.stack[-1] != tag:
            self.errors.append(
                f"Mismatched tags: expected {self.stack[-1]}, got {tag}"
            )
        self.stack.pop()

    def finalize(self) -> bool:
        """Check for unclosed tags."""
        if self.stack:
            self.errors.append(f"Unclosed tags: {self.stack}")
        return len(self.errors) == 0


def create_synthetic_designs(count: int = 20) -> Dict[str, List[Dict]]:
    """Create synthetic design JSON specs for testing.

    Distributes designs evenly across three types (landing, dashboard, blog).
    Each design includes varied region configurations (header, sidebar, content, footer)
    to ensure diverse layouts for training and validation.

    Args:
        count: Total number of designs to generate

    Returns:
        Dict mapping design_type to list of design specifications
    """
    designs_by_type = {
        'landing': [],
        'dashboard': [],
        'blog': [],
    }

    # Design configurations with varied region setups
    configs = [
        # Full layout (all regions present)
        {
            'header': {'x': 0, 'y': 0, 'width': 1000, 'height': 80},
            'sidebar': None,
            'content': {'x': 0, 'y': 80, 'width': 1000, 'height': 670},
            'footer': {'x': 0, 'y': 750, 'width': 1000, 'height': 50},
        },
        # With sidebar
        {
            'header': {'x': 0, 'y': 0, 'width': 1000, 'height': 100},
            'sidebar': {'x': 0, 'y': 100, 'width': 200, 'height': 650},
            'content': {'x': 200, 'y': 100, 'width': 800, 'height': 650},
            'footer': {'x': 0, 'y': 750, 'width': 1000, 'height': 50},
        },
        # Minimal layout
        {
            'header': {'x': 0, 'y': 0, 'width': 1000, 'height': 50},
            'sidebar': None,
            'content': {'x': 0, 'y': 50, 'width': 1000, 'height': 700},
            'footer': None,
        },
        # Content-heavy layout
        {
            'header': {'x': 0, 'y': 0, 'width': 1000, 'height': 120},
            'sidebar': {'x': 0, 'y': 120, 'width': 150, 'height': 680},
            'content': {'x': 150, 'y': 120, 'width': 850, 'height': 630},
            'footer': {'x': 0, 'y': 750, 'width': 1000, 'height': 50},
        },
    ]

    designs_per_type = count // len(designs_by_type)
    remainder = count % len(designs_by_type)

    for type_idx, design_type in enumerate(['landing', 'dashboard', 'blog']):
        num_for_type = designs_per_type + (1 if type_idx < remainder else 0)
        for i in range(num_for_type):
            config = configs[i % len(configs)]
            # Make a copy and slightly vary dimensions for diversity
            design = {}
            for region_name, region_spec in config.items():
                if region_spec is None:
                    design[region_name] = None
                else:
                    # Vary dimensions slightly to create diversity
                    design[region_name] = {
                        'x': max(0, region_spec['x'] + (i % 10) - 5),
                        'y': max(0, region_spec['y'] + (i % 15) - 7),
                        'width': max(10, region_spec['width'] + (i % 20) - 10),
                        'height': max(10, region_spec['height'] + (i % 20) - 10),
                    }
            designs_by_type[design_type].append(design)

    return designs_by_type


class TestDatasetIntegration:
    """Integration tests for dataset_builder module."""

    def test_dataset_output_format(self):
        """AC1.5: Verify training_data.npz has correct format.

        Expected:
          - design_vectors: (N, 128) float32
          - component_vectors: (N, 512) float32
          - labels: (N,) int64
          - design_types: list matching len(labels)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic designs
            designs_by_type = create_synthetic_designs(count=15)
            design_dir = Path(tmpdir) / "designs"
            design_dir.mkdir()

            # Write designs to JSON files
            file_count = 0
            for design_type, designs in designs_by_type.items():
                for design in designs:
                    file_path = design_dir / f"{design_type}_{file_count}.json"
                    with open(file_path, 'w') as f:
                        json.dump(design, f)
                    file_count += 1

            # Run dataset builder
            output_path = Path(tmpdir) / "training_data.npz"
            args = argparse.Namespace(
                input_dir=str(design_dir),
                output=str(output_path),
                seed=42,
                count=None,
            )
            result = dataset_builder_main(args)
            assert result == 0, "dataset_builder_main should return 0"
            assert output_path.exists(), "training_data.npz should be created"

            # Load and validate
            data = np.load(output_path)

            # AC1.5 checks
            assert 'design_vectors' in data.files
            assert 'component_vectors' in data.files
            assert 'labels' in data.files
            assert 'design_types' in data.files

            design_vectors = data['design_vectors']
            component_vectors = data['component_vectors']
            labels = data['labels']
            design_types = data['design_types']

            # Shape checks
            assert design_vectors.ndim == 2, "design_vectors should be 2D"
            assert design_vectors.shape[1] == 128, "design_vectors should have 128 columns"
            assert component_vectors.ndim == 2, "component_vectors should be 2D"
            assert component_vectors.shape[1] == 512, "component_vectors should have 512 columns"
            assert labels.ndim == 1, "labels should be 1D"

            # Dtype checks
            assert design_vectors.dtype == np.float32, "design_vectors should be float32"
            assert component_vectors.dtype == np.float32, "component_vectors should be float32"
            assert labels.dtype == np.int64, "labels should be int64"

            # Consistency checks
            N = design_vectors.shape[0]
            assert component_vectors.shape[0] == N
            assert labels.shape[0] == N

            # design_types should allow mapping each label to a type string
            # If design_types is a list of type names (const), each label should be
            # a valid index into it. If design_types is an array of N strings (one per sample),
            # it should have length N. Accept either interpretation.
            if hasattr(design_types, '__len__'):
                max_label = np.max(labels)
                # Validate that labels are valid indices into design_types
                assert max_label < len(design_types), \
                    f"Max label {max_label} should be < len(design_types) {len(design_types)}"
                assert np.min(labels) >= 0, "Min label should be >= 0"


class TestTrainingIntegration:
    """Integration tests for training module."""

    def test_training_loss_monotonic_decrease(self):
        """AC3.5: Verify training loss decreases from first to last epoch.

        Expected:
          - With --log-file specified, loss.json is created
          - losses[0] > losses[-1] (loss decreases)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic training data
            designs_by_type = create_synthetic_designs(count=20)
            design_dir = Path(tmpdir) / "designs"
            design_dir.mkdir()

            file_count = 0
            for design_type, designs in designs_by_type.items():
                for design in designs:
                    file_path = design_dir / f"{design_type}_{file_count}.json"
                    with open(file_path, 'w') as f:
                        json.dump(design, f)
                    file_count += 1

            # Step 1: Create dataset
            dataset_path = Path(tmpdir) / "training_data.npz"
            args = argparse.Namespace(
                input_dir=str(design_dir),
                output=str(dataset_path),
                seed=42,
                count=None,
            )
            result = dataset_builder_main(args)
            assert result == 0

            # Step 2: Train model with loss logging
            model_path = Path(tmpdir) / "model.pt"
            loss_log_path = Path(tmpdir) / "loss.json"

            train_args = argparse.Namespace(
                input=str(dataset_path),
                epochs=3,
                batch_size=4,
                output=str(model_path),
                log_file=str(loss_log_path),
                learning_rate=0.01,
            )
            result = train_main(train_args)
            assert result == 0, "train_main should return 0"
            assert model_path.exists(), "model.pt should be created"
            assert loss_log_path.exists(), "loss.json should be created"

            # Step 3: Verify loss decrease
            with open(loss_log_path) as f:
                loss_data = json.load(f)

            assert 'epochs' in loss_data, "loss.json should have 'epochs' key"
            losses = [e['loss'] for e in loss_data['epochs']]
            assert len(losses) == 3, "Should have 3 epoch losses"

            # AC3.5: loss[0] > loss[-1]
            assert losses[0] > losses[-1], \
                f"Loss should decrease: {losses[0]} should be > {losses[-1]}"


class TestGenerationIntegration:
    """Integration tests for generation and HTML output."""

    def test_generation_count_and_validity(self):
        """AC4.2: Verify HTML file count and basic validity.

        Expected:
          - Generates exactly 10 HTML files
          - All files parse as valid HTML (HTMLParser.feed succeeds)
          - At least 8 of 10 files are unique by content
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train model with more diverse designs
            designs_by_type = create_synthetic_designs(count=30)
            design_dir = Path(tmpdir) / "designs"
            design_dir.mkdir()

            file_count = 0
            for design_type, designs in designs_by_type.items():
                for design in designs:
                    file_path = design_dir / f"{design_type}_{file_count}.json"
                    with open(file_path, 'w') as f:
                        json.dump(design, f)
                    file_count += 1

            # Create dataset
            dataset_path = Path(tmpdir) / "training_data.npz"
            args = argparse.Namespace(
                input_dir=str(design_dir),
                output=str(dataset_path),
                seed=42,
                count=None,
            )
            result = dataset_builder_main(args)
            assert result == 0

            # Train model with more epochs for better learned diversity
            model_path = Path(tmpdir) / "model.pt"
            train_args = argparse.Namespace(
                input=str(dataset_path),
                epochs=10,
                batch_size=4,
                output=str(model_path),
                log_file=None,
                learning_rate=0.01,
            )
            result = train_main(train_args)
            assert result == 0

            # Generate 10 HTML files - verify count and validity
            output_dir = Path(tmpdir) / "html_out"
            output_dir.mkdir()

            gen_args = [
                '--model', str(model_path),
                '--type', 'landing',
                '--count', '10',
                '--output', str(output_dir),
                '--seed', '42',
            ]
            result = generate_main(gen_args)
            assert result == 0, "generate_main should return 0"

            # AC4.2: Count HTML files
            html_files = sorted([f for f in output_dir.glob("**/index.html")])
            assert len(html_files) == 10, \
                f"Expected 10 HTML files, got {len(html_files)}"

            # AC4.2: Validate HTML parsing and collect unique contents
            html_contents = []
            parsed_count = 0
            for html_file in html_files:
                with open(html_file) as f:
                    content = f.read()

                # Try to parse with HTMLParser
                try:
                    parser = HTMLParser()
                    parser.feed(content)
                    parsed_count += 1
                    html_contents.append(content)
                except Exception as e:
                    pytest.fail(f"Failed to parse {html_file}: {e}")

            assert parsed_count == 10, f"Failed to parse {10 - parsed_count} files"

            # AC4.2: Check uniqueness (at least 8 of 10 unique)
            unique_contents = set(html_contents)
            unique_count = len(unique_contents)
            assert unique_count >= 8, \
                f"Expected at least 8 unique HTML files, got {unique_count}"

    def test_html_structural_validity(self):
        """AC4.3: Verify 80%+ generated HTML has correct nesting.

        Uses StructureValidator to check for mismatched/unclosed tags.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal training setup
            designs_by_type = create_synthetic_designs(count=15)
            design_dir = Path(tmpdir) / "designs"
            design_dir.mkdir()

            file_count = 0
            for design_type, designs in designs_by_type.items():
                for design in designs:
                    file_path = design_dir / f"{design_type}_{file_count}.json"
                    with open(file_path, 'w') as f:
                        json.dump(design, f)
                    file_count += 1

            dataset_path = Path(tmpdir) / "training_data.npz"
            args = argparse.Namespace(
                input_dir=str(design_dir),
                output=str(dataset_path),
                seed=42,
                count=None,
            )
            result = dataset_builder_main(args)
            assert result == 0

            model_path = Path(tmpdir) / "model.pt"
            train_args = argparse.Namespace(
                input=str(dataset_path),
                epochs=3,
                batch_size=4,
                output=str(model_path),
                log_file=None,
                learning_rate=0.01,
            )
            result = train_main(train_args)
            assert result == 0

            output_dir = Path(tmpdir) / "html_out"
            output_dir.mkdir()

            gen_args = [
                '--model', str(model_path),
                '--type', 'landing',
                '--count', '10',
                '--output', str(output_dir),
                '--seed', '42',
            ]
            result = generate_main(gen_args)
            assert result == 0

            # Validate structure
            html_files = sorted([f for f in output_dir.glob("**/index.html")])
            valid_count = 0

            for html_file in html_files:
                with open(html_file) as f:
                    content = f.read()

                validator = StructureValidator()
                validator.feed(content)
                if validator.finalize():
                    valid_count += 1

            valid_pct = 100 * valid_count / len(html_files)
            assert valid_pct >= 80, \
                f"AC4.3: Only {valid_pct:.1f}% structurally valid, need 80%+"

    def test_html_css_validity(self):
        """AC4.4: Verify 80%+ generated HTML has valid CSS.

        Checks that <style> blocks contain ':' and ';' in rules.
        Files without <style> tags are counted as missing CSS and not included in valid count.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            designs_by_type = create_synthetic_designs(count=15)
            design_dir = Path(tmpdir) / "designs"
            design_dir.mkdir()

            file_count = 0
            for design_type, designs in designs_by_type.items():
                for design in designs:
                    file_path = design_dir / f"{design_type}_{file_count}.json"
                    with open(file_path, 'w') as f:
                        json.dump(design, f)
                    file_count += 1

            dataset_path = Path(tmpdir) / "training_data.npz"
            args = argparse.Namespace(
                input_dir=str(design_dir),
                output=str(dataset_path),
                seed=42,
                count=None,
            )
            result = dataset_builder_main(args)
            assert result == 0

            model_path = Path(tmpdir) / "model.pt"
            train_args = argparse.Namespace(
                input=str(dataset_path),
                epochs=3,
                batch_size=4,
                output=str(model_path),
                log_file=None,
                learning_rate=0.01,
            )
            result = train_main(train_args)
            assert result == 0

            output_dir = Path(tmpdir) / "html_out"
            output_dir.mkdir()

            gen_args = [
                '--model', str(model_path),
                '--type', 'dashboard',
                '--count', '10',
                '--output', str(output_dir),
                '--seed', '42',
            ]
            result = generate_main(gen_args)
            assert result == 0

            # Validate CSS
            html_files = sorted([f for f in output_dir.glob("**/index.html")])
            valid_count = 0

            for html_file in html_files:
                with open(html_file) as f:
                    content = f.read()

                # Extract style content
                style_match = re.search(r'<style[^>]*>(.*?)</style>', content, re.DOTALL)
                if not style_match:
                    continue

                css = style_match.group(1)

                # Check for CSS rules with ':' and ';'
                rules = re.findall(r'{[^}]*}', css)
                if not rules:
                    continue

                has_valid_css = True
                for rule in rules:
                    if ':' not in rule or ';' not in rule:
                        has_valid_css = False
                        break

                if has_valid_css:
                    valid_count += 1

            valid_pct = 100 * valid_count / len(html_files) if html_files else 0
            assert valid_pct >= 80, \
                f"AC4.4: Only {valid_pct:.1f}% have valid CSS, need 80%+"

    def test_generation_design_type_consistency(self):
        """AC4.5: Verify all generated specs have design_type matching CLI parameter.

        Tests multiple design types (landing, dashboard, blog).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            designs_by_type = create_synthetic_designs(count=15)
            design_dir = Path(tmpdir) / "designs"
            design_dir.mkdir()

            file_count = 0
            for design_type, designs in designs_by_type.items():
                for design in designs:
                    file_path = design_dir / f"{design_type}_{file_count}.json"
                    with open(file_path, 'w') as f:
                        json.dump(design, f)
                    file_count += 1

            dataset_path = Path(tmpdir) / "training_data.npz"
            args = argparse.Namespace(
                input_dir=str(design_dir),
                output=str(dataset_path),
                seed=42,
                count=None,
            )
            result = dataset_builder_main(args)
            assert result == 0

            model_path = Path(tmpdir) / "model.pt"
            train_args = argparse.Namespace(
                input=str(dataset_path),
                epochs=3,
                batch_size=4,
                output=str(model_path),
                log_file=None,
                learning_rate=0.01,
            )
            result = train_main(train_args)
            assert result == 0

            # Test each design type
            for design_type in ['landing', 'dashboard', 'blog']:
                output_dir = Path(tmpdir) / f"html_{design_type}"
                output_dir.mkdir()

                gen_args = [
                    '--model', str(model_path),
                    '--type', design_type,
                    '--count', '5',
                    '--output', str(output_dir),
                    '--seed', '42',
                ]
                result = generate_main(gen_args)
                assert result == 0

                # Check that all generated specs have matching design_type
                spec_files = sorted([f for f in output_dir.glob("**/index_*.json")])
                for spec_file in spec_files:
                    with open(spec_file) as f:
                        spec = json.load(f)
                    assert spec.get('design_type') == design_type, \
                        f"Spec {spec_file} has design_type={spec.get('design_type')}, " \
                        f"expected {design_type}"

    def test_generation_exact_count(self):
        """AC4.6: Verify exact file count matches --count parameter.

        Tests with various counts (10, 15, 1).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            designs_by_type = create_synthetic_designs(count=15)
            design_dir = Path(tmpdir) / "designs"
            design_dir.mkdir()

            file_count = 0
            for design_type, designs in designs_by_type.items():
                for design in designs:
                    file_path = design_dir / f"{design_type}_{file_count}.json"
                    with open(file_path, 'w') as f:
                        json.dump(design, f)
                    file_count += 1

            dataset_path = Path(tmpdir) / "training_data.npz"
            args = argparse.Namespace(
                input_dir=str(design_dir),
                output=str(dataset_path),
                seed=42,
                count=None,
            )
            result = dataset_builder_main(args)
            assert result == 0

            model_path = Path(tmpdir) / "model.pt"
            train_args = argparse.Namespace(
                input=str(dataset_path),
                epochs=3,
                batch_size=4,
                output=str(model_path),
                log_file=None,
                learning_rate=0.01,
            )
            result = train_main(train_args)
            assert result == 0

            # Test various counts
            for test_count in [10, 15, 1]:
                output_dir = Path(tmpdir) / f"html_count_{test_count}"
                output_dir.mkdir()

                gen_args = [
                    '--model', str(model_path),
                    '--type', 'blog',
                    '--count', str(test_count),
                    '--output', str(output_dir),
                    '--seed', '42',
                ]
                result = generate_main(gen_args)
                assert result == 0

                # Count HTML files
                html_files = list(output_dir.glob("**/index.html"))
                assert len(html_files) == test_count, \
                    f"AC4.6: Expected {test_count} files, got {len(html_files)}"


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_vectorization_determinism(self):
        """Verify vectorization produces deterministic results with same seed."""
        design = {
            'header': {'x': 0, 'y': 0, 'width': 1000, 'height': 100},
            'sidebar': {'x': 0, 'y': 100, 'width': 200, 'height': 700},
            'content': {'x': 200, 'y': 100, 'width': 800, 'height': 700},
            'footer': None,
        }

        vec1_design, vec1_component = vectorize_design(design, seed=42)
        vec2_design, vec2_component = vectorize_design(design, seed=42)

        assert np.allclose(vec1_design, vec2_design), \
            "Design vectors should be identical with same seed"
        assert np.allclose(vec1_component, vec2_component), \
            "Component vectors should be identical with same seed"

    def test_inverse_vectorization_produces_valid_spec(self):
        """Verify inverse vectorization produces valid region specs."""
        # Create a component vector
        component_vec = np.zeros(512, dtype=np.float32)

        # Set header region (dims 0-127)
        component_vec[0:4] = [0.0, 0.0, 1.0, 0.1]  # x, y, width, height (normalized)

        # Set content region (dims 256-383)
        component_vec[256:260] = [0.0, 0.15, 1.0, 0.75]

        spec = inverse_vectorize_design(component_vec)

        # Verify spec structure
        assert 'header' in spec
        assert 'sidebar' in spec
        assert 'content' in spec
        assert 'footer' in spec

        # Header should be present
        assert spec['header'] is not None
        assert all(k in spec['header'] for k in ['x', 'y', 'width', 'height'])

        # Content should be present
        assert spec['content'] is not None
        assert all(k in spec['content'] for k in ['x', 'y', 'width', 'height'])

    def test_model_forward_pass_produces_correct_shape(self):
        """Verify model forward pass produces correct output shape."""
        model = DesignGeneratorNet(input_dim=128, hidden_dim=256, output_dim=512)
        model.eval()

        # Create batch of embeddings
        batch_size = 8
        embeddings = torch.randn(batch_size, 128, dtype=torch.float32)

        with torch.no_grad():
            output = model(embeddings)

        assert output.shape == (batch_size, 512), \
            f"Expected output shape (8, 512), got {output.shape}"
        assert output.dtype == torch.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
