from types import SimpleNamespace

import pytest

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
