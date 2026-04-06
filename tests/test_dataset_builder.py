"""Tests for dataset_builder module."""

import pytest
import numpy as np
import json
import tempfile
import argparse
from pathlib import Path
from design_generator.dataset_builder import (
    DESIGN_TYPES,
    load_designs,
    vectorize_design,
    assign_labels,
    main
)


class TestDesignTypes:
    """Test DESIGN_TYPES constant."""

    def test_design_types_length(self):
        """AC1.2: DESIGN_TYPES should have at least 3 types."""
        assert len(DESIGN_TYPES) >= 3

    def test_design_types_contains_expected(self):
        """Test that DESIGN_TYPES contains expected design types."""
        assert 'landing' in DESIGN_TYPES
        assert 'dashboard' in DESIGN_TYPES
        assert 'blog' in DESIGN_TYPES


class TestLoadDesigns:
    """Test load_designs function."""

    def test_load_designs_valid_directory(self):
        """Test loading designs from a valid directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test design files
            design1 = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
            design2 = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 80},
                "sidebar": {"x": 0, "y": 80, "width": 200, "height": 720},
                "content": {"x": 200, "y": 80, "width": 800, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }

            with open(f"{tmpdir}/design1.json", "w") as f:
                json.dump(design1, f)
            with open(f"{tmpdir}/design2.json", "w") as f:
                json.dump(design2, f)

            designs = load_designs(tmpdir)
            assert len(designs) == 2
            assert designs[0] == design1
            assert designs[1] == design2

    def test_load_designs_empty_directory(self):
        """Test loading from an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            designs = load_designs(tmpdir)
            assert len(designs) == 0

    def test_load_designs_nonexistent_directory(self):
        """Test that OSError is raised for nonexistent directory."""
        with pytest.raises(OSError):
            load_designs("/nonexistent/directory/path")

    def test_load_designs_skips_malformed_json(self):
        """Test that malformed JSON files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid design
            valid_design = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
            with open(f"{tmpdir}/valid.json", "w") as f:
                json.dump(valid_design, f)

            # Malformed JSON
            with open(f"{tmpdir}/malformed.json", "w") as f:
                f.write("{invalid json content")

            designs = load_designs(tmpdir)
            assert len(designs) == 1
            assert designs[0] == valid_design

    def test_load_designs_skips_missing_required_keys(self):
        """Test that designs with missing required keys are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid design
            valid_design = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
            with open(f"{tmpdir}/valid.json", "w") as f:
                json.dump(valid_design, f)

            # Missing 'footer' key
            incomplete_design = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650}
            }
            with open(f"{tmpdir}/incomplete.json", "w") as f:
                json.dump(incomplete_design, f)

            designs = load_designs(tmpdir)
            assert len(designs) == 1
            assert designs[0] == valid_design


class TestVectorizeDesign:
    """Test vectorize_design function."""

    def test_vectorize_design_shape(self):
        """AC1.5: Test that vectorize_design returns correct shapes."""
        design = {
            "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
            "sidebar": None,
            "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
            "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
        }

        design_vector, component_vector = vectorize_design(design)

        assert design_vector.shape == (128,)
        assert component_vector.shape == (512,)

    def test_vectorize_design_dtype(self):
        """AC1.5: Test that vectorize_design returns float32 dtype."""
        design = {
            "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
            "sidebar": None,
            "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
            "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
        }

        design_vector, component_vector = vectorize_design(design)

        assert design_vector.dtype == np.float32
        assert component_vector.dtype == np.float32

    def test_vectorize_design_determinism(self):
        """AC1.4: Test that same seed produces identical vectors."""
        design = {
            "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
            "sidebar": {"x": 0, "y": 100, "width": 200, "height": 700},
            "content": {"x": 200, "y": 100, "width": 800, "height": 650},
            "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
        }

        seed = 42
        design_vec1, component_vec1 = vectorize_design(design, seed=seed)
        design_vec2, component_vec2 = vectorize_design(design, seed=seed)

        assert np.allclose(design_vec1, design_vec2)
        assert np.allclose(component_vec1, component_vec2)

    def test_vectorize_design_with_none_regions(self):
        """Test vectorization with None regions."""
        design = {
            "header": None,
            "sidebar": None,
            "content": {"x": 0, "y": 0, "width": 1000, "height": 800},
            "footer": None
        }

        design_vector, component_vector = vectorize_design(design)

        # Should still have correct shape and dtype
        assert design_vector.shape == (128,)
        assert component_vector.shape == (512,)
        assert design_vector.dtype == np.float32
        assert component_vector.dtype == np.float32

        # Content region (dims 64-96) should have non-zero values
        assert np.any(design_vector[64:96] != 0)

    def test_vectorize_design_normalization(self):
        """Test that coordinates are properly normalized to [0, 1]."""
        design = {
            "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
            "sidebar": None,
            "content": None,
            "footer": None
        }

        design_vector, component_vector = vectorize_design(design)

        # Header should be at region 0 (dims 0-31)
        # x=0 -> 0.0, y=0 -> 0.0, width=1000 -> 1.0, height=100 -> 0.125
        assert design_vector[0] == pytest.approx(0.0)
        assert design_vector[1] == pytest.approx(0.0)
        assert design_vector[2] == pytest.approx(1.0)
        assert design_vector[3] == pytest.approx(0.125, abs=0.01)

    def test_vectorize_design_all_regions(self):
        """Test vectorization with all regions present."""
        design = {
            "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
            "sidebar": {"x": 0, "y": 100, "width": 200, "height": 700},
            "content": {"x": 200, "y": 100, "width": 800, "height": 650},
            "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
        }

        design_vector, component_vector = vectorize_design(design)

        # All regions should have non-zero content
        for region_idx in range(4):
            offset = region_idx * 32
            assert np.any(design_vector[offset:offset + 4] != 0)

            comp_offset = region_idx * 128
            assert np.any(component_vector[comp_offset:comp_offset + 4] != 0)


class TestAssignLabels:
    """Test assign_labels function."""

    def test_assign_labels_shape(self):
        """AC1.5: Test that assign_labels returns correct shape."""
        designs = [
            {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            },
            {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 80},
                "sidebar": {"x": 0, "y": 80, "width": 200, "height": 720},
                "content": {"x": 200, "y": 80, "width": 800, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
        ]

        labels = assign_labels(designs)

        assert labels.shape == (2,)

    def test_assign_labels_dtype(self):
        """AC1.5: Test that assign_labels returns int64 dtype."""
        designs = [
            {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
        ]

        labels = assign_labels(designs)

        assert labels.dtype == np.int64

    def test_assign_labels_valid_indices(self):
        """Test that assigned labels are valid indices into DESIGN_TYPES."""
        designs = [
            {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            },
            {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 80},
                "sidebar": {"x": 0, "y": 80, "width": 200, "height": 720},
                "content": {"x": 200, "y": 80, "width": 800, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
        ]

        labels = assign_labels(designs)

        for label in labels:
            assert 0 <= label < len(DESIGN_TYPES)

    def test_assign_labels_dashboard_detection(self):
        """Test that designs with sidebar are labeled as dashboard."""
        design_with_sidebar = {
            "header": {"x": 0, "y": 0, "width": 1000, "height": 80},
            "sidebar": {"x": 0, "y": 80, "width": 200, "height": 720},
            "content": {"x": 200, "y": 80, "width": 800, "height": 650},
            "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
        }

        labels = assign_labels([design_with_sidebar])

        assert labels[0] == DESIGN_TYPES.index('dashboard')


class TestCLIIntegration:
    """Test CLI entry point and integration."""

    def test_cli_valid_execution(self):
        """AC1.1: Test CLI with valid inputs creates output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample designs
            design1 = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
            input_dir = Path(tmpdir) / "designs"
            input_dir.mkdir()
            with open(input_dir / "design1.json", "w") as f:
                json.dump(design1, f)

            output_path = Path(tmpdir) / "training_data.npz"

            # Create args
            args = argparse.Namespace(
                input_dir=str(input_dir),
                output=str(output_path),
                seed=42,
                count=None
            )

            # Run main
            result = main(args)

            assert result == 0
            assert output_path.exists()

    def test_cli_output_file_keys(self):
        """AC1.1: Test that output file has required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample design
            design = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
            input_dir = Path(tmpdir) / "designs"
            input_dir.mkdir()
            with open(input_dir / "design1.json", "w") as f:
                json.dump(design, f)

            output_path = Path(tmpdir) / "training_data.npz"

            args = argparse.Namespace(
                input_dir=str(input_dir),
                output=str(output_path),
                seed=42,
                count=None
            )

            main(args)

            # Load and verify
            data = np.load(output_path)
            assert 'design_vectors' in data.files
            assert 'component_vectors' in data.files
            assert 'labels' in data.files
            assert 'design_types' in data.files

    def test_cli_count_parameter(self):
        """AC1.3: Test that --count parameter limits designs loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 5 sample designs
            input_dir = Path(tmpdir) / "designs"
            input_dir.mkdir()
            for i in range(5):
                design = {
                    "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                    "sidebar": None if i % 2 == 0 else {"x": 0, "y": 100, "width": 200, "height": 700},
                    "content": {"x": 0 if i % 2 == 0 else 200, "y": 100, "width": 1000 if i % 2 == 0 else 800, "height": 650},
                    "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
                }
                with open(input_dir / f"design{i}.json", "w") as f:
                    json.dump(design, f)

            output_path = Path(tmpdir) / "training_data.npz"

            args = argparse.Namespace(
                input_dir=str(input_dir),
                output=str(output_path),
                seed=42,
                count=3
            )

            result = main(args)

            assert result == 0
            data = np.load(output_path)
            assert len(data['labels']) == 3

    def test_cli_seed_parameter(self):
        """AC1.4: Test that --seed parameter produces deterministic results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample design
            design = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": {"x": 0, "y": 100, "width": 200, "height": 700},
                "content": {"x": 200, "y": 100, "width": 800, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
            input_dir = Path(tmpdir) / "designs"
            input_dir.mkdir()
            with open(input_dir / "design.json", "w") as f:
                json.dump(design, f)

            # Run twice with same seed
            output_path1 = Path(tmpdir) / "training_data1.npz"
            output_path2 = Path(tmpdir) / "training_data2.npz"

            args1 = argparse.Namespace(
                input_dir=str(input_dir),
                output=str(output_path1),
                seed=42,
                count=None
            )
            args2 = argparse.Namespace(
                input_dir=str(input_dir),
                output=str(output_path2),
                seed=42,
                count=None
            )

            main(args1)
            main(args2)

            data1 = np.load(output_path1)
            data2 = np.load(output_path2)

            assert np.allclose(data1['design_vectors'], data2['design_vectors'])
            assert np.allclose(data1['component_vectors'], data2['component_vectors'])

    def test_cli_with_multiple_designs(self):
        """AC1.5: Test output shapes and dtypes with multiple designs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple designs
            input_dir = Path(tmpdir) / "designs"
            input_dir.mkdir()
            for i in range(5):
                design = {
                    "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                    "sidebar": None if i % 2 == 0 else {"x": 0, "y": 100, "width": 200, "height": 700},
                    "content": {"x": 0 if i % 2 == 0 else 200, "y": 100, "width": 1000 if i % 2 == 0 else 800, "height": 650},
                    "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
                }
                with open(input_dir / f"design{i}.json", "w") as f:
                    json.dump(design, f)

            output_path = Path(tmpdir) / "training_data.npz"

            args = argparse.Namespace(
                input_dir=str(input_dir),
                output=str(output_path),
                seed=42,
                count=None
            )

            result = main(args)

            assert result == 0
            data = np.load(output_path)

            # Verify shapes and dtypes
            assert data['design_vectors'].shape == (5, 128)
            assert data['design_vectors'].dtype == np.float32
            assert data['component_vectors'].shape == (5, 512)
            assert data['component_vectors'].dtype == np.float32
            assert data['labels'].shape == (5,)
            assert data['labels'].dtype == np.int64
            assert len(data['design_types']) == len(DESIGN_TYPES)

    def test_cli_nonexistent_input_dir(self):
        """Test that CLI returns error code for nonexistent input directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                input_dir="/nonexistent/path",
                output=str(Path(tmpdir) / "training_data.npz"),
                seed=42,
                count=None
            )

            result = main(args)

            assert result == 1

    def test_cli_empty_input_dir(self):
        """Test that CLI returns error code for empty input directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "designs"
            input_dir.mkdir()

            args = argparse.Namespace(
                input_dir=str(input_dir),
                output=str(Path(tmpdir) / "training_data.npz"),
                seed=42,
                count=None
            )

            result = main(args)

            assert result == 1
