# pyglrec

pyglrec is a lightweight OpenGL frame recorder for Python. It streams GPU render targets straight into NVENC using a custom CUDA/OpenGL interop plugin, so you can capture headless renders or interactive viewports without round-tripping pixels through the CPU. A CPU path is also available when NVENC hardware is missing.

## Highlights

- **Zero-copy pipeline** - OpenGL multisample FBOs resolve directly into CUDA buffers; NVENC consumes NV12 frames without touching system memory.
- **Two recorder backends** - `NVENCFrameRecorder` targets MP4 (H.264/HEVC) while `UncompressedFrameCPURecorder` stores raw RGBA frames and assembles them via FFmpeg.
- **Off-screen or on-screen** - Use `pyglrec.FrameBuffer` to drive headless workers or sample `Quad` to preview the resolved texture.
- **Drop-in shader helpers** - Minimal shader/quad utilities let you focus on scene code.
- **Examples included** - `examples/render_cube.py` demonstrates recorder toggles and NVENC vs CPU workflows.

## Requirements

### Hardware

- NVIDIA GPU with CUDA compute capability 7.5+ for best performance (NVENC path).
- NVENC-enabled driver (R535+). The CPU recorder works on non-NVIDIA GPUs.

### Software

- Python 3.11 or newer.
- CUDA Toolkit 12.x (or compatible) with `nvcc` on `PATH`. The CUDA/OpenGL plugin is built the first time it is needed.
- For Windows, Microsoft Visual Studio Build Tools 2019/2022 (C++ workload) so `cl.exe` and headers are available.
- Recent GPU drivers with OpenGL 4.6 support (4.1 on macOS; NVENC is unavailable there, so use the CPU recorder).

### Python dependencies

- Core: `PyNvVideoCodec`, `glfw`, `PyOpenGL`, `numpy`, `imageio[ffmpeg]`, `cachetools`, `pybind11`.
- Optional example extras: `click`, `pyglm`, `pillow`.

## Installation

```sh
# Using uv
uv add git+https://github.com/ShineiArakawa/pyglrec.git

# Using pip
pip install git+https://github.com/ShineiArakawa/pyglrec.git
```

When the recorder first accesses CUDA, pyglrec compiles `src/pyglrec/custom_ops/cuda_gl_interop.{cpp,cu}` into a cached extension under your cache directory. Ensure `CUDA_HOME`, `CUDA_PATH`, or `nvcc` is resolvable so the build succeeds.

## Quickstart

```python
import glfw
import OpenGL.GL as gl
from pyglrec import NVENCFrameRecorder, UncompressedFrameCPURecorder

WIDTH, HEIGHT = 1920, 1080
recorder = NVENCFrameRecorder(
    width=WIDTH,
    height=HEIGHT,
    fps=60,
    avg_bitrate=20_000_000,
    codec="hevc",         # "h264" or "hevc"
    preset="P4",          # Optional NVENC preset (P1=fast ... P7=quality)
)

# If you do not have NVENC hardware, fall back to:
# recorder = UncompressedFrameCPURecorder(WIDTH, HEIGHT, fps=60, bitrate="20M")

while not glfw.window_should_close(window):
    glfw.poll_events()

    with recorder.record():
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        draw_scene()

    # Optionally blit to the default framebuffer using recorder.texture_id
    swap_buffers()

recorder.finalize("outputs/demo.mp4")
```

- `recorder.record()` wraps the draw pass that should be captured.
- The framebuffer is multisampled internally; the resolved texture ID is exposed via `recorder.texture_id` for preview quads or blitting.
- `finalize(path)` flushes threads/encoders and produces either an MP4 (NVENC) or FFmpeg-encoded video from saved raw frames (CPU recorder).

## Examples

Install the optional dependencies, then run:

```sh
uv run python examples/render_cube.py --help

uv run python examples/render_cube.py            # Record with CPU recorder
uv run python examples/render_cube.py --nvenc    # Record with NVENC recorder
```

Key flags:

- `--nvenc` toggles NVENC frame recording (defaults to CPU recorder).
- `--offscreen` disables the preview quad and synchronous buffer swaps.
- `--fps_limit` throttles rendering (defaults to your display refresh).
- `--out_dir` controls where MP4/NPZ files are written (defaults to `outputs/render_cube`).

Each demo leaves the recorded file in the chosen output directory, making it easy to diff NVENC vs CPU output.

## How it works

1. `FrameBuffer` builds a 4x MSAA FBO for drawing plus a resolve buffer for readback.
2. `cuda_gl_interop` lazily loads a pybind11 + CUDA extension that maps OpenGL textures into CUDA memory and converts RGBA -> NV12.
3. `NVENCFrameRecorder` feeds the CUDA surface into `PyNvVideoCodec` and appends Annex B bitstreams to a temporary file before `imageio-ffmpeg` remuxes to MP4.
4. `UncompressedFrameCPURecorder` reads pixels into cached NumPy arrays, spills them to `.npz` shards in the background, and stitches them with FFmpeg during `finalize`.

This design minimizes GPU->CPU copies and keeps the render loop responsive even when capturing 4K streams.

## License

MIT License. See [`LICENSE`](LICENSE) for details.
