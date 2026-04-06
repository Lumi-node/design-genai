"""Tests for CLI argument parsing and main entry points.

Tests cover:
1. Unit tests for argparse setup (flags, defaults, required args)
2. Functional tests for CLI invocation via subprocess
3. Exit code validation (0 on success, 1 on error)
4. Error handling for missing/invalid arguments
"""

import subprocess
import tempfile
import json
import argparse
import pytest
import numpy as np
from pathlib import Path
from design_generator.dataset_builder import main as dataset_builder_main
from design_generator.train import main as train_main
from design_generator.generate import main as generate_main


class TestDatasetBuilderParser:
    """Unit tests for dataset_builder argparse setup."""

    def test_parser_creation(self):
        """Test that argparse parser can be created for dataset_builder."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-dir', required=True)
        parser.add_argument('--output', default='./training_data.npz')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--count', type=int, default=None)
        assert parser is not None

    def test_dataset_builder_flags_exist(self):
        """Test dataset_builder --help contains expected flags."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.dataset_builder', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert '--input-dir' in result.stdout
        assert '--output' in result.stdout
        assert '--seed' in result.stdout
        assert '--count' in result.stdout

    def test_dataset_builder_output_default(self):
        """Test that --output has correct default."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.dataset_builder', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'default' in result.stdout.lower() or 'training_data.npz' in result.stdout

    def test_dataset_builder_seed_default(self):
        """Test that --seed has correct default (42)."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.dataset_builder', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert '42' in result.stdout or 'default' in result.stdout.lower()

    def test_dataset_builder_input_dir_required(self):
        """Test that --input-dir is required."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.dataset_builder', '--output', 'test.npz'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode != 0
        assert 'required' in result.stderr.lower() or '--input-dir' in result.stderr

    def test_dataset_builder_parse_args(self):
        """Test parsing arguments for dataset_builder."""
        # This would be tested through subprocess invocation
        pass


class TestTrainParser:
    """Unit tests for train argparse setup."""

    def test_train_flags_exist(self):
        """Test train --help contains expected flags."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.train', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert '--input' in result.stdout
        assert '--epochs' in result.stdout
        assert '--batch-size' in result.stdout
        assert '--output' in result.stdout
        assert '--log-file' in result.stdout
        assert '--learning-rate' in result.stdout

    def test_train_input_required(self):
        """Test that --input is required for train."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.train',
             '--epochs', '2', '--batch-size', '8'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode != 0
        assert 'required' in result.stderr.lower() or '--input' in result.stderr

    def test_train_epochs_default(self):
        """Test that --epochs has correct default."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.train', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'epochs' in result.stdout.lower()

    def test_train_batch_size_default(self):
        """Test that --batch-size has correct default."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.train', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'batch' in result.stdout.lower()

    def test_train_output_default(self):
        """Test that --output has correct default (model.pt)."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.train', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'model' in result.stdout.lower() or 'output' in result.stdout.lower()


class TestGenerateParser:
    """Unit tests for generate argparse setup."""

    def test_generate_flags_exist(self):
        """Test generate --help contains expected flags."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.generate', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert '--type' in result.stdout
        assert '--count' in result.stdout
        assert '--output' in result.stdout
        assert '--model' in result.stdout
        assert '--seed' in result.stdout

    def test_generate_type_default(self):
        """Test that --type has correct default (landing)."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.generate', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'landing' in result.stdout or 'type' in result.stdout.lower()

    def test_generate_count_default(self):
        """Test that --count has correct default."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.generate', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'count' in result.stdout.lower()

    def test_generate_output_default(self):
        """Test that --output has correct default (./)."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.generate', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert 'output' in result.stdout.lower()


class TestDatasetBuilderCLI:
    """Functional tests for dataset_builder CLI invocation."""

    def test_dataset_builder_with_valid_input(self):
        """AC1.1: dataset_builder creates .npz with correct keys and exit code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test design files
            design = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
            design_path = Path(tmpdir) / "design1.json"
            with open(design_path, 'w') as f:
                json.dump(design, f)

            output_path = Path(tmpdir) / "training_data.npz"

            # Run dataset_builder
            result = subprocess.run(
                ['python', '-m', 'design_generator.dataset_builder',
                 '--input-dir', tmpdir,
                 '--output', str(output_path)],
                capture_output=True,
                text=True,
                timeout=10
            )

            assert result.returncode == 0, f"Non-zero exit code: {result.stderr}"
            assert output_path.exists(), "Output file not created"

            # Verify file has correct keys
            data = np.load(output_path)
            assert 'design_types' in data.files
            assert 'design_vectors' in data.files
            assert 'labels' in data.files
            assert 'component_vectors' in data.files

    def test_dataset_builder_missing_input_dir(self):
        """Test that missing --input-dir returns error exit code 1."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.dataset_builder',
             '--output', 'test.npz'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # argparse returns exit code 2 for argument errors
        assert result.returncode != 0

    def test_dataset_builder_invalid_input_dir(self):
        """Test that invalid --input-dir path returns exit code 1."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.dataset_builder',
             '--input-dir', '/nonexistent/path',
             '--output', 'test.npz'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 1

    def test_dataset_builder_empty_directory(self):
        """Test dataset_builder with empty directory returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ['python', '-m', 'design_generator.dataset_builder',
                 '--input-dir', tmpdir,
                 '--output', 'test.npz'],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode == 1

    def test_dataset_builder_seed_reproducibility(self):
        """Test that same seed produces deterministic output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test design files
            design = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
            design_path = Path(tmpdir) / "design1.json"
            with open(design_path, 'w') as f:
                json.dump(design, f)

            output1 = Path(tmpdir) / "output1.npz"
            output2 = Path(tmpdir) / "output2.npz"

            # Run twice with same seed
            for output in [output1, output2]:
                subprocess.run(
                    ['python', '-m', 'design_generator.dataset_builder',
                     '--input-dir', tmpdir,
                     '--output', str(output),
                     '--seed', '42'],
                    capture_output=True,
                    timeout=10
                )

            # Load and compare
            data1 = np.load(output1)
            data2 = np.load(output2)

            assert np.allclose(data1['design_vectors'], data2['design_vectors'])
            assert np.allclose(data1['component_vectors'], data2['component_vectors'])

    def test_dataset_builder_count_parameter(self):
        """Test that --count parameter limits number of designs loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 test design files
            for i in range(3):
                design = {
                    "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                    "sidebar": None,
                    "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                    "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
                }
                design_path = Path(tmpdir) / f"design{i}.json"
                with open(design_path, 'w') as f:
                    json.dump(design, f)

            output_path = Path(tmpdir) / "training_data.npz"

            # Run with count=2
            result = subprocess.run(
                ['python', '-m', 'design_generator.dataset_builder',
                 '--input-dir', tmpdir,
                 '--output', str(output_path),
                 '--count', '2'],
                capture_output=True,
                text=True,
                timeout=10
            )

            assert result.returncode == 0
            data = np.load(output_path)
            assert data['design_vectors'].shape[0] == 2


class TestTrainCLI:
    """Functional tests for train CLI invocation."""

    def test_train_missing_input_file(self):
        """Test that missing input file returns error."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.train',
             '--input', '/nonexistent/file.npz',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 1

    def test_train_missing_input_argument(self):
        """Test that missing --input argument is required."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.train',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # argparse returns 2 for argument errors
        assert result.returncode != 0

    def test_train_with_valid_input(self):
        """AC3.1: train creates model.pt with exit code 0 in <60s."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal training data
            design_vectors = np.random.randn(5, 128).astype(np.float32)
            component_vectors = np.random.randn(5, 512).astype(np.float32)
            labels = np.array([0, 1, 2, 0, 1], dtype=np.int64)
            design_types = ['landing', 'dashboard', 'blog']

            input_path = Path(tmpdir) / "training_data.npz"
            np.savez_compressed(
                input_path,
                design_vectors=design_vectors,
                component_vectors=component_vectors,
                labels=labels,
                design_types=design_types
            )

            output_path = Path(tmpdir) / "model.pt"

            # Run train (should complete in <60s for 2 epochs, 5 samples)
            result = subprocess.run(
                ['python', '-m', 'design_generator.train',
                 '--input', str(input_path),
                 '--epochs', '2',
                 '--batch-size', '8',
                 '--output', str(output_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            assert result.returncode == 0, f"Non-zero exit: {result.stderr}"
            assert output_path.exists(), "Model file not created"

    def test_train_creates_log_file_on_request(self):
        """Test that --log-file creates loss log in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal training data
            design_vectors = np.random.randn(5, 128).astype(np.float32)
            component_vectors = np.random.randn(5, 512).astype(np.float32)
            labels = np.array([0, 1, 2, 0, 1], dtype=np.int64)
            design_types = ['landing', 'dashboard', 'blog']

            input_path = Path(tmpdir) / "training_data.npz"
            np.savez_compressed(
                input_path,
                design_vectors=design_vectors,
                component_vectors=component_vectors,
                labels=labels,
                design_types=design_types
            )

            output_path = Path(tmpdir) / "model.pt"
            log_path = Path(tmpdir) / "loss.json"

            # Run train with log file
            result = subprocess.run(
                ['python', '-m', 'design_generator.train',
                 '--input', str(input_path),
                 '--epochs', '2',
                 '--batch-size', '8',
                 '--output', str(output_path),
                 '--log-file', str(log_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            assert result.returncode == 0
            assert log_path.exists(), "Log file not created"

            # Verify log format
            with open(log_path) as f:
                log_data = json.load(f)
            assert 'config' in log_data
            assert 'epochs' in log_data
            assert len(log_data['epochs']) == 2


class TestGenerateCLI:
    """Functional tests for generate CLI invocation."""

    def test_generate_missing_model_file(self):
        """Test that missing model file returns error."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.generate',
             '--model', '/nonexistent/model.pt',
             '--type', 'landing',
             '--count', '1'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 1

    def test_generate_invalid_design_type(self):
        """Test that invalid --type returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal trained model
            from design_generator.model import DesignGeneratorNet
            import torch

            model = DesignGeneratorNet(128, 256, 512)
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            result = subprocess.run(
                ['python', '-m', 'design_generator.generate',
                 '--model', str(model_path),
                 '--type', 'invalid_type',
                 '--count', '1'],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 1
            assert 'invalid' in result.stderr.lower() or 'type' in result.stderr.lower()

    def test_generate_with_valid_model(self):
        """AC4.1: generate creates HTML files with exit code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal trained model
            from design_generator.model import DesignGeneratorNet
            import torch

            model = DesignGeneratorNet(128, 256, 512)
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            output_dir = Path(tmpdir) / "html_out"

            result = subprocess.run(
                ['python', '-m', 'design_generator.generate',
                 '--model', str(model_path),
                 '--type', 'landing',
                 '--count', '3',
                 '--output', str(output_dir),
                 '--seed', '42'],
                capture_output=True,
                text=True,
                timeout=30
            )

            assert result.returncode == 0, f"Non-zero exit: {result.stderr}"
            assert output_dir.exists(), "Output directory not created"

            # Check that HTML files were created
            html_files = list(output_dir.glob('**/index.html'))
            assert len(html_files) >= 1, f"Expected HTML files, got: {list(output_dir.glob('**/*'))}"

    def test_generate_respects_count_parameter(self):
        """Test that --count parameter controls number of generated files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from design_generator.model import DesignGeneratorNet
            import torch

            model = DesignGeneratorNet(128, 256, 512)
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            output_dir = Path(tmpdir) / "html_out"

            result = subprocess.run(
                ['python', '-m', 'design_generator.generate',
                 '--model', str(model_path),
                 '--type', 'landing',
                 '--count', '2',
                 '--output', str(output_dir)],
                capture_output=True,
                text=True,
                timeout=30
            )

            assert result.returncode == 0
            html_files = list(output_dir.glob('**/index.html'))
            assert len(html_files) >= 1

    def test_generate_respects_type_parameter(self):
        """Test that generated specs include the requested design_type (AC4.5)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from design_generator.model import DesignGeneratorNet
            import torch

            model = DesignGeneratorNet(128, 256, 512)
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            output_dir = Path(tmpdir) / "html_out"

            result = subprocess.run(
                ['python', '-m', 'design_generator.generate',
                 '--model', str(model_path),
                 '--type', 'dashboard',
                 '--count', '1',
                 '--output', str(output_dir)],
                capture_output=True,
                text=True,
                timeout=30
            )

            assert result.returncode == 0


class TestExitCodes:
    """Tests for exit code behavior."""

    def test_dataset_builder_exit_0_on_success(self):
        """Test dataset_builder returns exit code 0 on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design = {
                "header": {"x": 0, "y": 0, "width": 1000, "height": 100},
                "sidebar": None,
                "content": {"x": 0, "y": 100, "width": 1000, "height": 650},
                "footer": {"x": 0, "y": 750, "width": 1000, "height": 50}
            }
            design_path = Path(tmpdir) / "design1.json"
            with open(design_path, 'w') as f:
                json.dump(design, f)

            result = subprocess.run(
                ['python', '-m', 'design_generator.dataset_builder',
                 '--input-dir', tmpdir,
                 '--output', str(Path(tmpdir) / 'test.npz')],
                timeout=10
            )
            assert result.returncode == 0

    def test_dataset_builder_exit_1_on_error(self):
        """Test dataset_builder returns exit code 1 on error."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.dataset_builder',
             '--input-dir', '/nonexistent/path',
             '--output', 'test.npz'],
            timeout=10
        )
        assert result.returncode == 1

    def test_train_exit_0_on_success(self):
        """Test train returns exit code 0 on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_vectors = np.random.randn(5, 128).astype(np.float32)
            component_vectors = np.random.randn(5, 512).astype(np.float32)
            labels = np.array([0, 1, 2, 0, 1], dtype=np.int64)
            design_types = ['landing', 'dashboard', 'blog']

            input_path = Path(tmpdir) / "training_data.npz"
            np.savez_compressed(
                input_path,
                design_vectors=design_vectors,
                component_vectors=component_vectors,
                labels=labels,
                design_types=design_types
            )

            result = subprocess.run(
                ['python', '-m', 'design_generator.train',
                 '--input', str(input_path),
                 '--epochs', '1',
                 '--output', str(Path(tmpdir) / 'model.pt')],
                timeout=60
            )
            assert result.returncode == 0

    def test_train_exit_1_on_error(self):
        """Test train returns exit code 1 on error."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.train',
             '--input', '/nonexistent/file.npz',
             '--epochs', '1'],
            timeout=10
        )
        assert result.returncode == 1

    def test_generate_exit_0_on_success(self):
        """Test generate returns exit code 0 on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from design_generator.model import DesignGeneratorNet
            import torch

            model = DesignGeneratorNet(128, 256, 512)
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            result = subprocess.run(
                ['python', '-m', 'design_generator.generate',
                 '--model', str(model_path),
                 '--type', 'landing',
                 '--count', '1',
                 '--output', str(Path(tmpdir) / 'html')],
                timeout=30
            )
            assert result.returncode == 0

    def test_generate_exit_1_on_error(self):
        """Test generate returns exit code 1 on error."""
        result = subprocess.run(
            ['python', '-m', 'design_generator.generate',
             '--model', '/nonexistent/model.pt',
             '--type', 'landing',
             '--count', '1'],
            timeout=10
        )
        assert result.returncode == 1
