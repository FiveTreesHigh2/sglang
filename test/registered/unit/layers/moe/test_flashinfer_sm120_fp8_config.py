from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.arg_groups.overrides import (
    _moe_runner_backend_quant_constraints,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.server_args import MOE_RUNNER_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _view(**overrides):
    values = dict(
        quantization="fp8",
        moe_runner_backend="flashinfer_sm120_fp8",
        tp_size=1,
        ep_size=1,
        moe_a2a_backend="none",
        enable_lora=False,
        lora_paths=[],
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_backend_is_registered():
    backend = MoeRunnerBackend("flashinfer_sm120_fp8")
    assert backend is MoeRunnerBackend.FLASHINFER_SM120_FP8
    assert backend.is_flashinfer_sm120_fp8()
    assert "flashinfer_sm120_fp8" in MOE_RUNNER_BACKEND_CHOICES


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"quantization": "modelopt_fp8"}, "blockwise FP8"),
        ({"quantization": "mxfp8"}, "blockwise FP8"),
        ({"tp_size": 2}, "tp_size=1"),
        ({"ep_size": 2}, "ep_size=1"),
        ({"moe_a2a_backend": "deepep"}, "moe_a2a_backend=none"),
        ({"enable_lora": True}, "LoRA"),
        ({"lora_paths": ["adapter"]}, "LoRA"),
        ({"enable_two_batch_overlap": True}, "TBO"),
        ({"enable_single_batch_overlap": True}, "SBO"),
    ],
)
def test_backend_rejects_out_of_scope_server_config(override, message):
    with pytest.raises(ValueError, match=message):
        _moe_runner_backend_quant_constraints(_view(**override))


def test_backend_accepts_fp8_or_autodetected_quantization():
    assert _moe_runner_backend_quant_constraints(_view()) == {}
    assert _moe_runner_backend_quant_constraints(_view(quantization=None)) == {}


def test_weight_scale_conversion_is_transpose_contiguous():
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        prepare_flashinfer_sm120_fp8_weight_scales,
    )

    w13 = torch.arange(2 * 8 * 4, dtype=torch.float32).view(2, 8, 4)
    w2 = torch.arange(2 * 4 * 8, dtype=torch.float32).view(2, 4, 8)
    w13_fi, w2_fi = prepare_flashinfer_sm120_fp8_weight_scales(w13, w2)

    assert w13_fi.shape == (2, 4, 8)
    assert w2_fi.shape == (2, 8, 4)
    assert w13_fi.is_contiguous()
    assert w2_fi.is_contiguous()
    torch.testing.assert_close(w13_fi, w13.transpose(1, 2))
    torch.testing.assert_close(w2_fi, w2.transpose(1, 2))


def test_moe_runner_uses_registered_fused_func():
    from sglang.srt.layers.moe.moe_runner.base import (
        FusedOpPool,
        MoeRunnerConfig,
    )
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeA2ABackend

    with patch(
        "sglang.srt.layers.moe.moe_runner.runner.get_moe_a2a_backend",
        return_value=MoeA2ABackend.NONE,
    ):
        runner = MoeRunner(
            MoeRunnerBackend.FLASHINFER_SM120_FP8,
            MoeRunnerConfig(),
        )

    registered = FusedOpPool.get_fused_func("none", "flashinfer_sm120_fp8")
    assert registered is not None
    assert runner.runner_core is None
    assert runner.fused_func is registered


def _quant_config(block_shape=(128, 128), *, use_mxfp8=False, fp4=False):
    return SimpleNamespace(
        use_mxfp8=use_mxfp8,
        weight_block_size=list(block_shape),
        is_fp4_experts=fp4,
        dequant_fp4_to_fp8=False,
    )


def _method_for_blockwise_test(block_shape=(128, 128)):
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    method = Fp8MoEMethod.__new__(Fp8MoEMethod)
    method.block_quant = True
    method.use_mxfp8 = False
    method.is_fp4_expert = False
    method.weight_block_size = list(block_shape)
    method.quant_config = SimpleNamespace(weight_block_size=list(block_shape))
    return method


def test_prepare_layer_registers_flashinfer_weight_scales():
    method = _method_for_blockwise_test()
    layer = torch.nn.Module()
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.arange(2 * 8 * 4, dtype=torch.float32).view(2, 8, 4),
        requires_grad=False,
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.arange(2 * 4 * 8, dtype=torch.float32).view(2, 4, 8),
        requires_grad=False,
    )
    original_w13 = layer.w13_weight_scale_inv.detach().clone()
    original_w2 = layer.w2_weight_scale_inv.detach().clone()

    with patch(
        "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
        return_value=MoeRunnerBackend.FLASHINFER_SM120_FP8,
    ), patch.object(method, "process_weights_after_loading_block_quant"):
        method.process_weights_after_loading(layer)

    assert layer.w13_weight_scale_fi.shape == (2, 4, 8)
    assert layer.w2_weight_scale_fi.shape == (2, 8, 4)
    assert layer.w13_weight_scale_fi.is_contiguous()
    assert layer.w2_weight_scale_fi.is_contiguous()
    torch.testing.assert_close(layer.w13_weight_scale_inv, original_w13)
    torch.testing.assert_close(layer.w2_weight_scale_inv, original_w2)
    torch.testing.assert_close(
        layer.w13_weight_scale_fi,
        original_w13.transpose(1, 2),
    )
    torch.testing.assert_close(
        layer.w2_weight_scale_fi,
        original_w2.transpose(1, 2),
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_quant_config((64, 128)), r"block_shape must be \(128, 128\)"),
        (_quant_config(use_mxfp8=True), "MXFP8"),
        (_quant_config(fp4=True), "FP4 expert"),
    ],
)
def test_fp8_method_rejects_unsupported_flashinfer_sm120_quant(config, message):
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    with patch(
        "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
        return_value=MoeRunnerBackend.FLASHINFER_SM120_FP8,
    ), pytest.raises(ValueError, match=message):
        Fp8MoEMethod(config)


def test_fp8_method_creates_flashinfer_sm120_runner():
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.utils import MoeA2ABackend
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    with patch(
        "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
        return_value=MoeRunnerBackend.FLASHINFER_SM120_FP8,
    ), patch(
        "sglang.srt.layers.moe.moe_runner.runner.get_moe_a2a_backend",
        return_value=MoeA2ABackend.NONE,
    ):
        method = Fp8MoEMethod(_quant_config())
        method.create_moe_runner(torch.nn.Module(), MoeRunnerConfig())

    assert method.runner.runner_backend is MoeRunnerBackend.FLASHINFER_SM120_FP8
