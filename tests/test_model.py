"""Unit tests for DesignGeneratorNet model.

Tests cover instantiation, forward pass shapes, trainable parameters,
and layer structure validation per AC2.1-AC2.4.
"""

import pytest
import torch
import torch.nn as nn

from design_generator.model import DesignGeneratorNet


class TestModelInstantiation:
    """Tests for model instantiation (AC2.1)."""

    def test_instantiate_with_default_arguments(self):
        """Test DesignGeneratorNet(128, 256, 512) succeeds and returns nn.Module."""
        model = DesignGeneratorNet(128, 256, 512)
        assert isinstance(model, nn.Module)
        assert isinstance(model, torch.nn.Module)

    def test_instantiate_with_default_parameters(self):
        """Test DesignGeneratorNet() with no arguments uses correct defaults."""
        model = DesignGeneratorNet()
        assert isinstance(model, nn.Module)

    def test_instantiate_with_custom_dimensions(self):
        """Test instantiation with custom input/hidden/output dimensions."""
        model = DesignGeneratorNet(input_dim=64, hidden_dim=128, output_dim=256)
        assert isinstance(model, nn.Module)


class TestForwardPass:
    """Tests for forward pass shape correctness (AC2.2)."""

    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    def test_forward_pass_shape_correctness(self, batch_size):
        """Test forward pass with various batch sizes produces correct output shape."""
        model = DesignGeneratorNet(128, 256, 512)
        x = torch.randn(batch_size, 128)
        y = model(x)
        assert y.shape == (batch_size, 512), f"Expected ({batch_size}, 512), got {y.shape}"

    def test_forward_pass_dtype_preservation(self):
        """Test forward pass preserves float32 dtype."""
        model = DesignGeneratorNet(128, 256, 512)
        x = torch.randn(16, 128, dtype=torch.float32)
        y = model(x)
        assert y.dtype == torch.float32

    def test_forward_pass_batch_size_one(self):
        """Test forward pass with batch_size=1 (edge case)."""
        model = DesignGeneratorNet(128, 256, 512)
        x = torch.randn(1, 128)
        y = model(x)
        assert y.shape == (1, 512)

    def test_forward_pass_custom_dimensions(self):
        """Test forward pass with custom input/output dimensions."""
        model = DesignGeneratorNet(input_dim=64, hidden_dim=128, output_dim=256)
        x = torch.randn(16, 64)
        y = model(x)
        assert y.shape == (16, 256)


class TestTrainableParameters:
    """Tests for trainable parameter validation (AC2.3)."""

    def test_has_trainable_parameters(self):
        """Test model has >0 trainable parameters."""
        model = DesignGeneratorNet(128, 256, 512)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0, f"Expected >0 trainable params, got {trainable_params}"

    def test_trainable_parameter_count(self):
        """Test trainable parameter count matches expected calculation."""
        model = DesignGeneratorNet(128, 256, 512)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # fc1: 128*256 weights + 256 biases = 32896
        # fc2: 256*512 weights + 512 biases = 131584
        # Total: 164480
        expected = 128 * 256 + 256 + 256 * 512 + 512
        assert trainable_params == expected, f"Expected {expected}, got {trainable_params}"

    def test_all_parameters_require_grad(self):
        """Test all parameters are trainable (requires_grad=True)."""
        model = DesignGeneratorNet(128, 256, 512)
        for param in model.parameters():
            assert param.requires_grad, "Parameter should require grad"


class TestLayerStructure:
    """Tests for layer architecture validation (AC2.4)."""

    def test_exactly_two_linear_layers(self):
        """Test model contains exactly 2 Linear layers."""
        model = DesignGeneratorNet(128, 256, 512)
        linear_count = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
        assert linear_count == 2, f"Expected 2 Linear layers, got {linear_count}"

    def test_has_relu_activation(self):
        """Test model contains ReLU activation."""
        model = DesignGeneratorNet(128, 256, 512)
        has_relu = any(isinstance(m, nn.ReLU) for m in model.modules())
        assert has_relu, "Model should contain ReLU activation"

    def test_layer_dimensions(self):
        """Test Linear layers have correct input/output dimensions."""
        model = DesignGeneratorNet(128, 256, 512)
        # fc1: 128 → 256
        assert model.fc1.in_features == 128
        assert model.fc1.out_features == 256
        # fc2: 256 → 512
        assert model.fc2.in_features == 256
        assert model.fc2.out_features == 512

    def test_no_unused_modules(self):
        """Test model contains only fc1, relu, fc2 (no extra modules)."""
        model = DesignGeneratorNet(128, 256, 512)
        # Count all modules (including the model itself)
        all_modules = list(model.modules())
        # Should be: DesignGeneratorNet, Linear, Linear, ReLU (4 total)
        assert len(all_modules) == 4, f"Expected 4 modules total, got {len(all_modules)}"


class TestAcceptanceCriteria:
    """Integration tests verifying acceptance criteria AC2.1-AC2.4."""

    def test_ac2_1_instantiation(self):
        """AC2.1: Model instantiation succeeds, returns torch.nn.Module."""
        model = DesignGeneratorNet(128, 256, 512)
        assert isinstance(model, torch.nn.Module)

    def test_ac2_2_forward_pass_shape(self):
        """AC2.2: Input (16, 128) produces output (16, 512)."""
        model = DesignGeneratorNet(128, 256, 512)
        x = torch.randn(16, 128)
        y = model(x)
        assert y.shape == (16, 512)

    def test_ac2_3_trainable_parameters(self):
        """AC2.3: Model has >0 trainable parameters."""
        model = DesignGeneratorNet(128, 256, 512)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0

    def test_ac2_4_exactly_two_linear_layers(self):
        """AC2.4: Model contains exactly 2 Linear layers."""
        model = DesignGeneratorNet(128, 256, 512)
        linear_count = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
        assert linear_count == 2
