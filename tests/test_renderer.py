from typing import Any

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import vulkan_forge.backend as backend
import vulkan_forge.renderer as renderer


class DummyProps:
    def __init__(self, device_type: int):
        self.deviceType = device_type


def test_select_device_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    vk = backend.vk
    monkeypatch.setattr(backend, "_create_instance", lambda: "instance")
    monkeypatch.setattr(vk, "vkEnumeratePhysicalDevices", lambda instance: [])
    device = backend.select_device()
    assert device.device_type == vk.VK_PHYSICAL_DEVICE_TYPE_CPU


def test_pipeline_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    vk = backend.vk
    device_info = backend.DeviceInfo(handle="dev", properties=DummyProps(vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU), device_type=vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    monkeypatch.setattr(backend, "select_device", lambda: device_info)

    calls: dict[str, Any] = {}
    def fake_rp(d: Any, i: Any, a: Any, out: Any) -> int:
        calls["rp"] = True
        return vk.VK_SUCCESS

    monkeypatch.setattr(vk, "vkCreateRenderPass2", fake_rp)
    def fake_shader_module(d: Any, i: Any, a: Any) -> str:
        calls["sm"] = calls.get("sm", 0) + 1
        return f"sm{calls['sm']}"

    monkeypatch.setattr(vk, "vkCreateShaderModule", fake_shader_module)
    monkeypatch.setattr(vk, "vkCreatePipelineLayout", lambda d, i, a: calls.setdefault("pl", True) or "pl")
    monkeypatch.setattr(vk, "vkCreateGraphicsPipelines", lambda d, c, n, i, a: calls.setdefault("gp", True) or ["gp"])

    r = renderer.VulkanRenderer(4, 4)
    assert calls.get("rp") and calls.get("pl") and calls.get("gp")
    assert calls.get("sm") == 2


def test_cpu_render(monkeypatch: pytest.MonkeyPatch) -> None:
    vk = backend.vk
    device_info = backend.DeviceInfo(handle=None, properties=None, device_type=vk.VK_PHYSICAL_DEVICE_TYPE_CPU)
    monkeypatch.setattr(backend, "select_device", lambda: device_info)

    r = renderer.VulkanRenderer(2, 3)
    img = r.render()
    assert img.shape == (3, 2, 4)
    assert np.all(img[:, :, :3] == 128)
    assert np.all(img[:, :, 3] == 255)
