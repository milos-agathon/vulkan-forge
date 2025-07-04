# AGENTS.md

<!--
AGENTS.md ────────────────────────────────────────────────────────────────────
This file is parsed automatically by AI coding agents (e.g. OpenAI Codex,
GitHub Copilot Agents). Keep it **authoritative & current**. A stale
AGENTS.md ⇒ stale AI output.
-->

# 🛠️ Vulkan‑Forge — Machine‑Readable Developer Manual 

## 1  Project Scope & Architecture

* **Purpose**  A high‑performance Vulkan backend for ray‑tracing and real‑time
  rendering, with a Pythonic façade and a C++20 core.
* **High‑level data‑flow**

```
python/vulkan_forge                 ← public API, helper glue
   │   ⤷ pybind11 ↔ cpp/src, cpp/include/vf
   │
examples/        → user demos / perf benchmarks
assets/          → pre‑compiled SPIR‑V shaders
 tests/          → pytest suite (CPU + GPU)
```

| Task type                           | Touch these files                               |
| ----------------------------------- | ----------------------------------------------- |
| Python utility / helper             | `python/vulkan_forge/*.py`                      |
| Perf‑critical feature / Vulkan call | `cpp/src/*.cpp`, `cpp/include/vf/*.hpp`         |
| Shader tweak                        | `cpp/shaders/*.glsl` → run `compile_shaders.py` |
| Docs / examples                     | `examples/*.py`, `README.md`                    |
| Unit or perf test                   | `tests/` (pytest ‑m performance)                |

---

## 2  Setup & Dependencies

> Agents **must** replicate these steps in CI containers or devboxes.

1. **System**  Windows 10/11 x64 (Visual Studio 2022) or Linux (GLIBC ≥ 2.34,
   `clang 15`).  GPU driver ≥ Vulkan 1.3.
2. **SDKs / Tooling**

   ```bash
   # Vulkan SDK
   choco install vulkan-sdk  # Windows
   # or
   sudo apt install vulkan-sdk             # Linux

   # Optional: RenderDoc + Nsight for GPU debugging
   ```
3. **Python 3.9 – 3.12** + pip ≥ 23.
4. **Editable install & shader build**

   ```bash
   pip install -e .[dev]          # builds native extension via CMake
   python python/vulkan_forge/compile_shaders.py  # regenerates SPIR‑V
   ```

   Installs: pytest, mypy, ruff, black, clang‑format pre‑commit hooks.

---

## 3  Build / Run / Test Commands

| Purpose            | Command                                                         |
| ------------------ | --------------------------------------------------------------- |
| Full build         | `pip install -e .`                                              |
| Unit tests         | `pytest -q`  (GPU tests auto‑skip)                              |
| Lint (Python)      | `ruff check python tests`                                       |
| Type‑check         | `mypy python/vulkan_forge`                                      |
| Format             | `black python tests` · `clang-format -i cpp/src cpp/include/vf` |
| Perf suite         | `pytest -m performance -q`                                      |
| Flamegraph (Linux) | `scripts/profile_flamegraph.sh render_demo`                     |

Agents **must** run every command above and resolve failures **before** closing a task.

---

## 4  Coding Conventions

### 4.1 Python

* Black‑formatted, line ≤ 120.
* snake\_case for funcs/vars; PascalCase for classes.
* Full `typing` on all publiс APIs.
* Docstrings → **Google style**.
* Prefer `loguru` logger wrappers in `python/vulkan_forge/log.py`.

### 4.2 C++20

* Header‑only helpers in `vf/detail/*.hpp` where practical.
* RAII mandatory for all Vulkan handles—use wrappers in `vk_common.hpp` / `vma_util.hpp`.
* Avoid throwing across C++/Python boundary—return error codes and raise on Python side.
* Optimisation flags: `-O3 -march=native -flto` (MSVC `/O2 /GL`).

### 4.3 Git Hygiene

* Prefix commits with `[Fix]`, `[Feat]`, `[Refactor]`, `[Perf]`, `[Docs]`, `[Tests]`.
* One logical change per PR; reference issues (`Fixes #123`).

---

## 5  AI Agent Playbook  🚀

| Scenario             | Recommended prompt snippet                                                          |
| -------------------- | ----------------------------------------------------------------------------------- |
| Generate new feature | "Add a function `foo_bar` in `python/vulkan_forge/…` that … — follow style guide."  |
| Debug failing test   | "`pytest -q` fails at `test_allocator` with … — propose patch and update tests."    |
| Optimise hot loop    | "Profile shows `transfer_vertices` dominates — refactor in C++ with pybind11."      |
| Remove duplication   | "Find funcs with identical bodies in `*.py`; consolidate and update imports/tests." |

When creating patches the agent **must**:

1. Include **context lines** for reliable patching.
2. Update/append **unit tests**.
3. Run **all** build + test commands (section 3).
4. Mention changed files in PR description.

---

## 6  Debugging & Profiling Cheatsheet

### 6.1 GPU / Vulkan

* Enable validation layers: `set VF_ENABLE_VALIDATION=1` or `export VF_ENABLE_VALIDATION=1`.
* Capture a frame: `render_demo.py --capture my.rdc` and open in **RenderDoc**.
* Memory leaks: run with `VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_simulation` + `vktrace`.

### 6.2 Native Extension

* Rebuild in **Debug**: `pip install -e . --config-settings build_type=Debug`.
* Check missing symbols with `dumpbin /exports _vulkan_forge_native.pyd` (Win) or `nm -g` (Linux).

### 6.3 CPU Profiling

* Linux: `perf record -g -- python examples/…` → `perf script | flamegraph.pl > flame.svg`.
* Windows: **VTune** project template in `scripts/vtune/tpl.vproj`.

---

## 7  Performance Guidelines

* Heavy math → NumPy vectorised or C++ (never Python loops > 1k).
* Re‑use pipelines & descriptor sets; avoid per‑frame recreation.
* All GPU allocations through **VMA** wrappers.
* Use **staging buffers** for uploads > 8 MiB.
* Batch `vkCmdPipelineBarrier` calls.

---

## 8  Known Pitfalls & Quick Fixes

| Symptom                               | Likely cause / fix                                               |
| ------------------------------------- | ---------------------------------------------------------------- |
| `create_allocator unavailable`        | Native module built without VMA; rebuild with `-DVF_WITH_VMA=ON` |
| `VK_ERROR_DEVICE_LOST` on laptop GPUs | Discrete GPU selected while on battery – force `--integrated`.   |
| Blank window on first frame           | Forgot to `vkDeviceWaitIdle` before swapchain rebuild.           |

---

## 9  Pull‑Request Checklist

1. Build & install succeeds (`pip install -e .`).
2. `pytest -q` passes—no new skips.
3. `ruff`, `mypy` clean.
4. GPU path tested on at least one Vulkan 1.3 device.
5. Docs & CHANGELOG updated.
6. **AGENTS.md** updated if workflow, build, or style rules changed.

---

## 10  How to Update This File

* Edit in the **same PR** that changes build/tests/workflow.
* Keep bullets concise; external links for deep dives.
* Run `markdownlint-cli2 AGENTS.md` before commit.


---

*(End of AGENTS.md)*

