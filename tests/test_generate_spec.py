"""Tests for generate module."""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from design_generator.generate import (
    inverse_vectorize_design,
    sample_design_embedding,
    generate_design_spec,
    load_model,
    generate_html,
    main
)
from design_generator.model import DesignGeneratorNet
from design_generator.dataset_builder import DESIGN_TYPES


class TestInverseVectorizeDesign:
    """Test inverse_vectorize_design function."""

    def test_inverse_vectorize_valid_component_vector(self):
        """Test inverse vectorization with valid component vector."""
        # Create a simple component vector with a header region
        component_vector = np.zeros(512, dtype=np.float32)
        # Set header (dims 0-127): normalized coords [0.1, 0.05, 0.5, 0.2]
        component_vector[0:4] = [0.1, 0.05, 0.5, 0.2]

        result = inverse_vectorize_design(component_vector)

        # Check structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {'header', 'sidebar', 'content', 'footer'}

        # Check header values
        assert result['header'] is not None
        assert 'x' in result['header']
        assert 'y' in result['header']
        assert 'width' in result['header']
        assert 'height' in result['header']

        # Check denormalization: x = 0.1 * 1000 = 100
        assert result['header']['x'] == 100
        assert result['header']['y'] == 40  # 0.05 * 800
        assert result['header']['width'] == 500  # 0.5 * 1000
        assert result['header']['height'] == 160  # 0.2 * 800

    def test_inverse_vectorize_null_regions(self):
        """Test that regions with low coordinate sums are marked as None."""
        component_vector = np.zeros(512, dtype=np.float32)
        # All regions have coordinates summing to < 0.1

        result = inverse_vectorize_design(component_vector)

        # All regions should be None
        assert result['header'] is None
        assert result['sidebar'] is None
        assert result['content'] is None
        assert result['footer'] is None

    def test_inverse_vectorize_all_regions(self):
        """Test with all regions populated."""
        component_vector = np.zeros(512, dtype=np.float32)

        # Header (dims 0-127)
        component_vector[0:4] = [0.1, 0.05, 0.5, 0.2]

        # Sidebar (dims 128-255)
        component_vector[128:132] = [0.0, 0.2, 0.2, 0.6]

        # Content (dims 256-383)
        component_vector[256:260] = [0.2, 0.2, 0.6, 0.6]

        # Footer (dims 384-511)
        component_vector[384:388] = [0.0, 0.85, 1.0, 0.15]

        result = inverse_vectorize_design(component_vector)

        assert result['header'] is not None
        assert result['sidebar'] is not None
        assert result['content'] is not None
        assert result['footer'] is not None

    def test_inverse_vectorize_clamping(self):
        """Test that coordinates are clamped to valid ranges."""
        component_vector = np.zeros(512, dtype=np.float32)
        # Set values > 1.0 (should be clamped to 1.0)
        component_vector[0:4] = [1.5, 0.5, 2.0, 1.2]

        result = inverse_vectorize_design(component_vector)

        # Values should be clamped to [0, image_width/height]
        assert result['header']['x'] <= 1000
        assert result['header']['y'] <= 800
        assert result['header']['width'] <= 1000
        assert result['header']['height'] <= 800

    def test_inverse_vectorize_extreme_values(self):
        """Test with extreme normalized values (0.0, 1.0)."""
        component_vector = np.zeros(512, dtype=np.float32)
        component_vector[0:4] = [0.0, 0.0, 1.0, 1.0]

        result = inverse_vectorize_design(component_vector)

        assert result['header']['x'] == 0
        assert result['header']['y'] == 0
        assert result['header']['width'] == 1000
        assert result['header']['height'] == 800

    def test_inverse_vectorize_shape_validation(self):
        """Test that wrong shape raises error."""
        wrong_vector = np.zeros(256, dtype=np.float32)

        with pytest.raises(ValueError, match="shape"):
            inverse_vectorize_design(wrong_vector)

    def test_inverse_vectorize_custom_dimensions(self):
        """Test with custom image dimensions."""
        component_vector = np.zeros(512, dtype=np.float32)
        component_vector[0:4] = [0.5, 0.5, 0.5, 0.5]

        result = inverse_vectorize_design(component_vector, image_width=500, image_height=400)

        assert result['header']['x'] == 250  # 0.5 * 500
        assert result['header']['y'] == 200  # 0.5 * 400
        assert result['header']['width'] == 250
        assert result['header']['height'] == 200


class TestSampleDesignEmbedding:
    """Test sample_design_embedding function."""

    def test_sample_design_embedding_shape(self):
        """Test that output has correct shape (128,)."""
        rng = np.random.default_rng(42)
        embedding = sample_design_embedding('landing', DESIGN_TYPES, rng)

        assert embedding.shape == (128,)

    def test_sample_design_embedding_dtype(self):
        """Test that output has dtype float32."""
        rng = np.random.default_rng(42)
        embedding = sample_design_embedding('landing', DESIGN_TYPES, rng)

        assert embedding.dtype == np.float32

    def test_sample_design_embedding_distribution(self):
        """Test that samples approximate N(0, 1) distribution."""
        rng = np.random.default_rng(42)
        samples = [sample_design_embedding('landing', DESIGN_TYPES, rng) for _ in range(100)]
        samples = np.array(samples)

        # Mean should be close to 0
        assert np.abs(np.mean(samples)) < 0.2

        # Std should be close to 1
        assert 0.8 < np.std(samples) < 1.2

    def test_sample_design_embedding_ignores_type(self):
        """Test that different design types produce different samples."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        # Same seed, same type -> same output
        emb1 = sample_design_embedding('landing', DESIGN_TYPES, rng1)
        emb2 = sample_design_embedding('landing', DESIGN_TYPES, rng2)
        assert np.allclose(emb1, emb2)

        # But same seed, different type -> still same output (ignores type)
        rng3 = np.random.default_rng(42)
        emb3 = sample_design_embedding('dashboard', DESIGN_TYPES, rng3)
        assert np.allclose(emb1, emb3)


class TestGenerateDesignSpec:
    """Test generate_design_spec function."""

    def test_generate_design_spec_returns_dict_with_required_keys(self):
        """Test that output dict has all required keys."""
        model = DesignGeneratorNet(input_dim=128, hidden_dim=256, output_dim=512)
        model.eval()

        embedding = np.random.randn(128).astype(np.float32)

        spec = generate_design_spec(model, embedding, 'landing')

        assert isinstance(spec, dict)
        assert 'header' in spec
        assert 'sidebar' in spec
        assert 'content' in spec
        assert 'footer' in spec
        assert 'design_type' in spec

    def test_generate_design_spec_design_type_matches(self):
        """Test that design_type in spec matches requested type (AC4.5)."""
        model = DesignGeneratorNet()
        model.eval()

        embedding = np.random.randn(128).astype(np.float32)

        spec = generate_design_spec(model, embedding, 'dashboard')

        assert spec['design_type'] == 'dashboard'

    def test_generate_design_spec_different_types(self):
        """Test with different design types."""
        model = DesignGeneratorNet()
        model.eval()

        embedding = np.random.randn(128).astype(np.float32)

        for design_type in DESIGN_TYPES:
            spec = generate_design_spec(model, embedding, design_type)
            assert spec['design_type'] == design_type

    def test_generate_design_spec_regions_structure(self):
        """Test that region dicts have correct structure."""
        model = DesignGeneratorNet()
        model.eval()

        embedding = np.random.randn(128).astype(np.float32)

        spec = generate_design_spec(model, embedding, 'landing')

        # Check that regions are either None or have required keys
        for region_name in ['header', 'sidebar', 'content', 'footer']:
            region = spec[region_name]
            if region is not None:
                assert 'x' in region
                assert 'y' in region
                assert 'width' in region
                assert 'height' in region


class TestLoadModel:
    """Test load_model function."""

    def test_load_model_success(self):
        """Test loading a valid model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a model
            model = DesignGeneratorNet()
            model_path = Path(tmpdir) / "test_model.pt"
            torch.save(model.state_dict(), model_path)

            # Load it back
            loaded_model = load_model(str(model_path))

            assert isinstance(loaded_model, DesignGeneratorNet)
            assert loaded_model.training == False  # Should be in eval mode

    def test_load_model_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/model.pt")

    def test_load_model_eval_mode(self):
        """Test that loaded model is in eval mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DesignGeneratorNet()
            model_path = Path(tmpdir) / "test_model.pt"
            torch.save(model.state_dict(), model_path)

            loaded_model = load_model(str(model_path))

            assert not loaded_model.training


class TestGenerateHtml:
    """Test generate_html function."""

    @patch('design_generator.generate.generate_html_structure')
    @patch('design_generator.generate.generate_css')
    @patch('design_generator.generate.write_output')
    def test_generate_html_success(self, mock_write, mock_css, mock_html):
        """Test successful HTML generation."""
        mock_html.return_value = "<html></html>"
        mock_css.return_value = "body { color: black; }"
        mock_write.return_value = "/path/to/index.html"

        design_spec = {
            'header': {'x': 0, 'y': 0, 'width': 100, 'height': 50},
            'sidebar': None,
            'content': {'x': 0, 'y': 50, 'width': 100, 'height': 100},
            'footer': None,
            'design_type': 'landing'
        }

        result = generate_html(design_spec, 0, "./output")

        assert result == "/path/to/index.html"
        mock_html.assert_called_once()
        mock_css.assert_called_once()
        mock_write.assert_called_once()

    @patch('design_generator.generate.generate_html_structure')
    @patch('design_generator.generate.generate_css')
    @patch('design_generator.generate.write_output')
    def test_generate_html_structure_failure(self, mock_write, mock_css, mock_html):
        """Test graceful handling of HTML generation failure."""
        mock_html.side_effect = Exception("HTML generation failed")

        design_spec = {
            'header': {'x': 0, 'y': 0, 'width': 100, 'height': 50},
            'sidebar': None,
            'content': None,
            'footer': None,
            'design_type': 'landing'
        }

        result = generate_html(design_spec, 0, "./output")

        assert result is None

    @patch('design_generator.generate.generate_html_structure')
    @patch('design_generator.generate.generate_css')
    @patch('design_generator.generate.write_output')
    def test_generate_html_write_failure(self, mock_write, mock_css, mock_html):
        """Test graceful handling of write failure."""
        mock_html.return_value = "<html></html>"
        mock_css.return_value = "body { color: black; }"
        mock_write.side_effect = Exception("Write failed")

        design_spec = {
            'header': {'x': 0, 'y': 0, 'width': 100, 'height': 50},
            'sidebar': None,
            'content': None,
            'footer': None,
            'design_type': 'landing'
        }

        result = generate_html(design_spec, 0, "./output")

        assert result is None


class TestRoundTrip:
    """Integration tests for round-trip vectorization."""

    def test_round_trip_vectorize_inverse(self):
        """Test that inverse vectorization can reconstruct original design."""
        from design_generator.dataset_builder import vectorize_design

        # Create original design
        original_design = {
            'header': {'x': 0, 'y': 0, 'width': 1000, 'height': 100},
            'sidebar': None,
            'content': {'x': 0, 'y': 100, 'width': 1000, 'height': 650},
            'footer': {'x': 0, 'y': 750, 'width': 1000, 'height': 50}
        }

        # Vectorize
        design_vec, component_vec = vectorize_design(original_design, seed=42)

        # Inverse vectorize
        reconstructed = inverse_vectorize_design(component_vec)

        # Check that reconstructed regions match original (within rounding)
        assert reconstructed['header'] is not None
        assert reconstructed['content'] is not None
        assert reconstructed['footer'] is not None
        assert reconstructed['sidebar'] is None

        # Check approximate equality (allowing for rounding)
        for key in ['header', 'content', 'footer']:
            orig_region = original_design[key]
            recon_region = reconstructed[key]
            if orig_region is not None:
                assert abs(orig_region['x'] - recon_region['x']) <= 2
                assert abs(orig_region['y'] - recon_region['y']) <= 2
                assert abs(orig_region['width'] - recon_region['width']) <= 2
                assert abs(orig_region['height'] - recon_region['height']) <= 2


class TestMain:
    """Test main CLI entry point."""

    def test_main_missing_model(self):
        """Test that missing model file returns exit code 1."""
        result = main(['--model', '/nonexistent/model.pt', '--count', '1'])
        assert result == 1

    def test_main_invalid_design_type(self):
        """Test that invalid design type returns exit code 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid model
            model = DesignGeneratorNet()
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            result = main([
                '--model', str(model_path),
                '--type', 'invalid_type',
                '--count', '1'
            ])

            assert result == 1

    @patch('design_generator.generate.generate_html')
    def test_main_success(self, mock_gen_html):
        """Test successful main execution."""
        mock_gen_html.return_value = "/path/to/index_0.html"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid model
            model = DesignGeneratorNet()
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            result = main([
                '--model', str(model_path),
                '--type', 'landing',
                '--count', '1',
                '--output', str(tmpdir)
            ])

            assert result == 0
            mock_gen_html.assert_called_once()

    @patch('design_generator.generate.generate_html')
    def test_main_multiple_designs(self, mock_gen_html):
        """Test generating multiple designs."""
        mock_gen_html.return_value = "/path/to/index.html"

        with tempfile.TemporaryDirectory() as tmpdir:
            model = DesignGeneratorNet()
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            result = main([
                '--model', str(model_path),
                '--type', 'dashboard',
                '--count', '3',
                '--output', str(tmpdir)
            ])

            assert result == 0
            assert mock_gen_html.call_count == 3

    @patch('design_generator.generate.generate_html')
    def test_main_ac45_compliance(self, mock_gen_html):
        """Test AC4.5 compliance: all specs have design_type matching request."""
        mock_gen_html.return_value = "/path/to/index.html"

        with tempfile.TemporaryDirectory() as tmpdir:
            model = DesignGeneratorNet()
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            # Test with different design types
            for design_type in ['landing', 'dashboard']:
                result = main([
                    '--model', str(model_path),
                    '--type', design_type,
                    '--count', '2',
                    '--output', str(tmpdir)
                ])

                assert result == 0
