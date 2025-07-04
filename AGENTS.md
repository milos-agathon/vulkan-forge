# AGENTS.md

## Project Scope & Architecture

* **Purpose** *Vulkan-Forge* provides a high-performance Vulkan GPU backend for
  ray-tracing and real-time rendering, with a Pythonic front-end and C++20 core.
* **High-level data-flow**

```
Python API (python/vulkan_forge)
     │  ⤷ pybind11 bindings ↔ C++ core (cpp/src, cpp/include/vf)
     │
examples/    → user demos / benchmarks
tests/       → pytest suite (CPU + GPU)
assets/      → pre-compiled SPIR-V shaders
```

* **When to edit where**

  | Task type                           | Primary location(s)                             |
  | ----------------------------------- | ----------------------------------------------- |
  | New Python-side utility             | `python/vulkan_forge/*.py`                      |
  | Perf-critical feature / Vulkan call | `cpp/src/*.cpp` + `*.hpp`                       |
  | Shader tweak                        | `cpp/shaders/*.glsl` → run `compile_shaders.py` |
  | Example / docs                      | `examples/*.py` / `README.md`                   |
  | Unit / perf test                    | `tests/` (pytest)                               |

## Setup & Dependencies

1. **System** Windows 10/11 64-bit (tested), Visual Studio 2022 Build Tools.
   Linux ≥ GLIBC 2.34 + `clang 15` also supported (CI matrix).
2. **Required SDKs**

   ```bash
   # Vulkan SDK 1.3.x – set env var
   choco install vulkan-sdk
   setx VULKAN_SDK "C:\VulkanSDK\1.3.xxx.x"
   ```
3. **Python 3.9 – 3.12** with pip ≥ 23.
4. **Build & install (editable)**

   ```bash
   # From repo root
   pip install -e .[dev]        # compiles C++ extension via CMake
   python python/vulkan_forge/compile_shaders.py
   ```

   *Installs pytest, flake8, mypy, black, and clang-format hooks.*

## Build / Run / Test Commands

| Purpose            | Command (run from repo root)                                                  |
| ------------------ | ----------------------------------------------------------------------------- |
| **Full build**     | `pip install -e .`                                                            |
| **Run unit tests** | `pytest -q` (GPU tests auto-skip if no device)                                |
| **Lint (Python)**  | `flake8 python tests`                                                         |
| **Type-check**     | `mypy python/vulkan_forge`                                                    |
| **Format**         | `black python tests`  ·  `clang-format -i cpp/src/*.cpp cpp/include/vf/*.hpp` |
| **Perf suite**     | `pytest -m performance -q`                                                    |

Agents **must** run **all** commands above and fix any failures before considering a task complete.

## Coding Conventions

### Python

* PEP 8 via **Black** (line ≤ 120, 4-space indent).
* Snake\_case for functions & variables; PascalCase for classes.
* Use `typing` throughout; public APIs require full type hints.
* Docstrings → Google style.

### C++20

* Classes & structs → PascalCase (`HeightFieldRenderer`).
* Methods & free functions → snake\_case (`set_vertex_buffer`).
* Member variables prefix `m_`.
* RAII required: manage all Vulkan handles via helper wrappers in
  `vk_common.hpp` / `vma_util.hpp`.
* Do **not** throw across the C++/Python boundary—return error codes or
  raise in bindings.

### Git Hygiene

* Commit message prefix `[Fix]`, `[Feat]`, `[Refactor]`, `[Docs]`, `[Tests]`.
* One logical change per PR; reference issues (`Fixes #123`).

## Performance Guidelines

* Heavy math → NumPy or C++ (`numpy_bridge.hpp`)—**never** Python loops for

  > 1 k elements.
* Minimise GPU ↔ CPU transfers inside render loops.
* Re-use Vulkan pipelines & descriptor sets; avoid per-frame recreation.
* If adding new GPU memory allocations, route through **VulkanMemoryAllocator
  (VMA)** abstractions in `vma_util.*`.
* Use staging buffers for large resource uploads > 8 MiB.

## Known Pitfalls & Tips

* Always call `vkDeviceWaitIdle` **before** destroying swapchain resources.
* Coordinate system is **Y-up** (shader code assumes this).
* `renderer.render()` auto-falls back to CPU if device init fails—do **not**
  remove fallback path.
* Shaders live in `cpp/shaders/`; after editing GLSL, run
  `python python/vulkan_forge/compile_shaders.py` to regenerate SPIR-V.

## Pull-Request Checklist

1. Code builds (`pip install -e .` succeeds)
2. `pytest -q` passes, no skips introduced
3. `flake8` + `mypy` clean
4. GPU code tested on at least one Vulkan 1.3 device (use `examples/`)
5. Docs & CHANGELOG updated where relevant
6. AGENTS.md updated **if** build/test/style/workflow rules changed

## Frequently Used Symbols

| Python Constant        | Meaning                                 |
| ---------------------- | --------------------------------------- |
| `BUFFER_USAGE_VERTEX`  | `vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT`  |
| `BUFFER_USAGE_INDEX`   | `vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT`   |
| `BUFFER_USAGE_STORAGE` | `vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` |

## Command Snippets & Examples

<details><summary>Compile & run a demo (GPU)</summary>

```bash
# Create virtualenv, install deps, and run cube demo
python -m venv vforge-env
.\vforge-env\Scripts\activate
pip install -e .[dev]
python examples/01_render_cube.py --frames 120
```

</details>

<details><summary>Create a new C++ feature and expose it to Python</summary>

1. Implement in `cpp/src/my_feature.cpp` + header.
2. Bind via `bindings.cpp` with pybind11.
3. Re-run `pip install -e .` (wheel rebuilds).
4. Add Python wrapper in `python/vulkan_forge/my_feature.py`.
5. Write tests in `tests/test_my_feature.py`.

</details>

---

### How to Update This File

* **Edit in the same PR** that changes build/tests/conventions.
* Keep bullet points concise; reference external docs for long explanations.
* Run `markdownlint-cli2 AGENTS.md` before commit.

---

*(End of AGENTS.md)*

