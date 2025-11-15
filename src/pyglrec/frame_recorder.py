"""
frame_recorder
==============

Recorder classes for OpenGL frame data.
"""


import contextlib
import os
import pathlib
import platform
import subprocess
import tempfile
import threading
import typing

import cachetools
import imageio
import imageio_ffmpeg
import numpy as np
import OpenGL.GL as gl

if platform.system() != 'Darwin':
    import PyNvVideoCodec as nvcodec
else:
    nvcodec = None  # NVENC is not supported on macOS

import pyglrec.cuda_gl_interop as cuda_gl_interop
import pyglrec.frame_buffer as frame_buffer
import pyglrec.util as util

# --------------------------------------------------------------------------------------------------------------
# CPU Frame Recorder


class UncompressedFrameCPURecorder:
    """Recorder for uncompressed CPU frame data."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 60,
        bitrate: str = '30M',
        cache_size: int = 512
    ):
        """Initialize the uncompressed frame recorder to record OpenGL frames. The recorder reads back frame data from the GPU to the CPU and saves them as a video file using FFmpeg.

        Parameters
        ----------
        width : int
            The width of the frames to record.
        height : int
            The height of the frames to record.
        fps : int, optional
            The frame rate of the recording, by default 60.
        bitrate : str, optional
            The target bitrate for the output video file, by default '30M'.
        cache_size : int, optional
            The number of frames to cache in memory before saving to disk, by default 512.

        Example
        -------
        >>> recorder = UncompressedFrameCPURecorder(width=1920, height=1080, fps=30)
        >>> with recorder.record():
        ...     # OpenGL drawing calls here
        >>> recorder.finalize("output.mp4")
        """

        self._fps = fps
        self._bitrate = bitrate

        self._tmp_dir = tempfile.TemporaryDirectory()

        self._frame_buffer = frame_buffer.FrameBuffer(width, height)

        self._cache_size = cache_size
        self._frame_cache: dict[int, np.ndarray] = cachetools.FIFOCache(maxsize=2 * cache_size)
        self._cur_frame_idx = 0

        self._saving_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def __del__(self):
        self._tmp_dir.cleanup()

    @property
    def texture_id(self) -> int:
        """The OpenGL texture ID of the framebuffer's color attachment.
        """
        return self._frame_buffer.texture_id

    @contextlib.contextmanager
    def record(self, enabled: bool = True):
        """Record a frame within the context.

        Parameters
        ----------
        enabled : bool, optional
            Whether to record the frame or not, by default True.
        """

        # First invoke drawing calls and render to the framebuffer
        with self._frame_buffer.ctx():
            yield

        if not enabled:
            return

        # After exiting the context, read back the frame data
        tex_data = np.zeros((self._frame_buffer.height, self._frame_buffer.width, 4), dtype=np.uint8)

        with self._frame_buffer.ctx_resolved_buffer():
            gl.glReadPixels(
                0,
                0,
                self._frame_buffer.width,
                self._frame_buffer.height,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                tex_data
            )

        with self._lock:
            self._frame_cache[self._cur_frame_idx] = np.flipud(tex_data)  # Flip vertically to match image coordinate system

        self._cur_frame_idx += 1

        with self._lock:
            if len(self._frame_cache) > self._cache_size:
                # Start a background thread to save frames to disk
                if self._saving_thread is None or not self._saving_thread.is_alive():
                    self._saving_thread = threading.Thread(target=self._save_frames_to_disk)
                    self._saving_thread.start()

    def _save_frames_to_disk(self, num: int | None = None) -> None:
        if num == 0:
            return

        with self._lock:
            num = min(num or self._cache_size, len(self._frame_cache))
            frame_indices = list(self._frame_cache.keys())[:num]

        frames = []
        for frame_idx in frame_indices:
            with self._lock:
                frame_data = self._frame_cache.pop(frame_idx)
            frames.append(frame_data)

        np.savez(os.path.join(self._tmp_dir.name, f"frames_{frame_indices[0]:08d}_{frame_indices[-1]:08d}.npz"), *frames)

    def finalize(self, out_file: str | pathlib.Path) -> None:
        """Finalize the recording and save all frames to a video file.

        Parameters
        ----------
        out_file : str | pathlib.Path
            The output video file path.
        """

        if isinstance(out_file, str):
            out_file = pathlib.Path(out_file)
        out_file = out_file.resolve()

        # Wait for any ongoing saving thread to finish
        if self._saving_thread is not None:
            self._saving_thread.join()

        # Save any remaining frames in the cache
        self._save_frames_to_disk(num=len(self._frame_cache))

        # Gather all saved .npz files
        npz_files = sorted([f for f in os.listdir(self._tmp_dir.name) if f.endswith('.npz')])

        if len(npz_files) > 0:
            # Create writer
            out_file.parent.mkdir(parents=True, exist_ok=True)

            pix_format = 'yuv420p'
            ffmpeg_args = []
            ffmpeg_args += ['-g', '1']  # Set GOP size to 1 for lossless encoding
            ffmpeg_args += ['-color_range', '2']  # Set color range to full
            ffmpeg_args += ['-movflags', '+write_colr']  # Write color profile
            ffmpeg_kwargs = dict(bitrate=self._bitrate, pixelformat=pix_format, output_params=ffmpeg_args)
            writer = imageio.get_writer(str(out_file), format='ffmpeg', mode='I', fps=self._fps, codec='libx265', **ffmpeg_kwargs)

            for npz_file in npz_files:
                data = np.load(os.path.join(self._tmp_dir.name, npz_file))
                for i in range(len(data.files)):
                    frame = data[f'arr_{i}']
                    writer.append_data(frame)

            writer.close()
        else:
            print("No frames were recorded; skipping video file creation.")


# --------------------------------------------------------------------------------------------------------------
# GPU Frame Recorder enhanced by NVENC


class CUDAArrayInterface:
    """CUDA Array Interface (CAI) for GPU memory representation."""

    def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...], typestr: str, gpu_alloc: int):
        """Initialize the CUDA Array Interface.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the array.
        strides : tuple[int, ...]
            The strides of the array.
        typestr : str
            The data type string (e.g., '|u1' for uint8).
        gpu_alloc : int
            The GPU memory address.
        """

        shape_int = tuple([int(x) for x in shape])
        stride_int = tuple([int(x) for x in strides])
        self.__cuda_array_interface__ = {"shape": shape_int, "strides": stride_int, "data": (int(gpu_alloc), False), "typestr": typestr, "version": 3}


class NVENCInputFrame:
    """Input frame for NVENC encoder."""

    def __init__(self, width: int, height: int, fmt: typing.Literal['NV12', 'YUV444']):
        """Initialize the NVENC input frame with CUDA memory.

        Parameters
        ----------
        width : int
            The width of the frame.
        height : int
            The height of the frame.
        fmt : typing.Literal['NV12', 'YUV444']
            The pixel format of the frame.
        """

        self._width = width
        self._height = height
        self._fmt = fmt

        # Initialize CUDA-OpenGL interop plugin
        self._plugin = cuda_gl_interop.get_cuda_plugin()
        if self._plugin is None:
            raise RuntimeError("CUDA-OpenGL interop plugin is not available. Check if 'nvcc' is installed and configured correctly.")

        # Initialize CUDA memories
        self._cuda_tex_mem = cuda_gl_interop.CUDAMemory2D_URGB4()   # For texture data
        self._cuda_tex_mem.allocate(width, height)

        self._cuda_frame_mem = cuda_gl_interop.CUDAMemory1D()  # For NVENC input frame data

        if fmt == "NV12":
            self.frame_size = int(width * height * 3 / 2)

            # Allocate NV12 frame memory
            self._cuda_frame_mem.allocate(self.frame_size)

            # Initialize CUDA Array Interface (CAI)
            self.cai = []
            self.cai.append(CUDAArrayInterface((height, width, 1), (width, 1, 1), "|u1", int(self._cuda_frame_mem.ptr)))
            self.cai.append(CUDAArrayInterface((int(height / 2), int(width / 2), 2), (width, 2, 1), "|u1", int(self._cuda_frame_mem.ptr) + width * height))
        elif fmt == "YUV444":
            self.frame_size = int(width * height * 3)

            # Allocate YUV444 frame memory
            self._cuda_frame_mem.allocate(self.frame_size)

            # Initialize CUDA Array Interface (CAI)
            self.cai = []
            self.cai.append(CUDAArrayInterface((height, width, 1), (width, 1, 1), "|u1", int(self._cuda_frame_mem.ptr)))
            self.cai.append(CUDAArrayInterface((height, width, 1), (width, 1, 1), "|u1", int(self._cuda_frame_mem.ptr) + width * height))
            self.cai.append(CUDAArrayInterface((height, width, 1), (width, 1, 1), "|u1", int(self._cuda_frame_mem.ptr) + width * height * 2))
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def dispose(self) -> None:
        """Gracefully dispose of CUDA memories."""

        if self._cuda_frame_mem:
            self._cuda_frame_mem.free()
            self._cuda_frame_mem = None
        if self._cuda_tex_mem:
            self._cuda_tex_mem.free()
            self._cuda_tex_mem = None

    def cuda(self) -> list[CUDAArrayInterface]:
        """This method is required by PyNvVideoCodec to get the CUDA Array Interface (CAI)."""

        return self.cai

    def process_current_texture(self, texture_id: int) -> None:
        """Retrieve and convert the current OpenGL texture into the NVENC input frame format.

        Parameters
        ----------
        texture_id : int
            OpenGL texture id

        Raises
        ------
        ValueError
            Unsupported pixel format
        """

        # Copy framebuffer texture to CUDA memory
        self._plugin.copy_texture_to_cuda_memory(
            texture_id,
            self._width,
            self._height,
            self._cuda_tex_mem.ptr,
            self._cuda_tex_mem.pitch,
        )

        # Convert RGBA texture to YUV format with chroma subsampling
        if self._fmt == "NV12":
            self._plugin.convert_rgba_to_nv12(
                self._width,
                self._height,
                self._cuda_tex_mem.ptr,
                self._cuda_tex_mem.pitch,
                self._cuda_frame_mem.ptr,
            )
        elif self._fmt == "YUV444":
            self._plugin.convert_rgba_to_yuv444(
                self._width,
                self._height,
                self._cuda_tex_mem.ptr,
                self._cuda_tex_mem.pitch,
                self._cuda_frame_mem.ptr,
            )
        else:
            raise ValueError(f"Unsupported format: {self._fmt}")


class NVENCFrameRecorder:
    """Recorder for GPU frame data using NVENC encoder."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 60,
        codec: typing.Literal['h264', 'hevc'] = 'hevc',
        preset: typing.Literal['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'] | None = None,
        avg_bitrate: int | str = '20M',  # 20 Mbps
        max_bitrate: int | str = '30M',  # 30 Mbps
        chroma_format: typing.Literal['NV12', 'YUV444'] = 'YUV444',
        rate_control_mode: typing.Literal['cbr', 'vbr', 'constqp'] = 'vbr',
    ):
        """Initialize the NVENC frame recorder to record OpenGL frames. The recorder uses CUDA-OpenGL interop to efficiently transfer frame data from OpenGL to CUDA for encoding without touching the system memory.

        Parameters
        ----------
        width : int
            The width of the frames to record.
        height : int
            The height of the frames to record.
        fps : int, optional
            The frame rate of the recording, by default 60.
        codec : typing.Literal['h264', 'hevc'], optional
            The codec to use for encoding ('h264' or 'hevc'), by default 'hevc'.
        preset : typing.Literal['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'] | None,
            The NVENC preset to use ('P1' to 'P7'), by default None.
        avg_bitrate : int, optional
            The average bitrate for encoding in bits per second, by default 20_000_000 (20 Mbps).
        max_bitrate : int, optional
            The maximum bitrate for encoding in bits per second, by default 30_000_000 (30 Mbps).
        chroma_format: typing.Literal['NV12', 'YUV444'] = 'YUV444',
            The chroma subsampling format for encoding ('NV12' or 'YUV444'), by default 'YUV444'.
        rate_control_mode : typing.Literal['cbr', 'vbr', 'constqp'], optional
            The rate control mode ('cbr', 'vbr', or 'constqp'), by default 'vbr'.

        Raises
        ------
        RuntimeError
            If the CUDA-OpenGL interop plugin is not available.

        Example
        -------
        >>> recorder = NVENCFrameRecorder(width=1920, height=1080, fps=30, codec='h264')
        >>> with recorder.record():
        ...     # OpenGL drawing calls here
        >>> recorder.finalize("output.mp4")
        """

        # Check if nvcodec is available
        if nvcodec is None:
            raise RuntimeError("PyNvVideoCodec is not available. NVENC is not supported on this platform.")

        self._width = width
        self._height = height
        self._fps = fps
        self._codec = codec

        # Initialize CUDA-OpenGL interop plugin
        self._plugin = cuda_gl_interop.get_cuda_plugin()
        if self._plugin is None:
            raise RuntimeError("CUDA-OpenGL interop plugin is not available. Check if 'nvcc' is installed and configured correctly.")

        self._frame_buffer = frame_buffer.FrameBuffer(width, height)

        # Get and set CUDA device for the current OpenGL context
        cuda_device_id = int(self._plugin.get_cuda_device_for_current_OpenGL_context())
        self._plugin.set_cuda_device_for_current_OpenGL_context(cuda_device_id)
        print(f"Using CUDA device ID {cuda_device_id} for NVENC encoding.")

        # Initialize NVENC encoder
        encoder_opts = util.EasyDict()
        if preset is not None:
            encoder_opts.preset = preset  # Preset ('P1' to 'P7')

        self._encoder = nvcodec.CreateEncoder(
            gpu_id=cuda_device_id,
            width=width,
            height=height,
            fmt=chroma_format,                          # NVENC prefers NV12 format
            usecpuinputbuffer=False,                    # Use GPU input buffer
            codec=codec,                                # Codec ('h264' or 'hevc')
            # Optional parameters (See "https://docs.nvidia.com/video-technologies/pynvvideocodec/pdf/PyNvVideoCodec_API_ProgGuide.pdf" for details)
            bitrate=avg_bitrate,                        # Bitrate
            fps=fps,                                    # Frames per second
            tuning_info='high_quality',                 # Tuning info ('high_quality', 'low_latency', 'ultra_low_latency', 'lossless')
            maxbitrate=max_bitrate,                     # Maximum bitrate
            vbvinit=max_bitrate,                        # Initial VBV buffer size
            vbvbufsize=max_bitrate,                     # VBV buffer size
            rc=rate_control_mode,                       # Rate control mode ('cbr', 'constqp', 'vbr')
            **encoder_opts
        )

        # Initialize AppFrame for NV12 format
        self._nvenc_frame = NVENCInputFrame(width, height, chroma_format)

        # Create a temporary file to store the encoded bitstream
        self._annex_b_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.h264' if codec == 'h264' else '.hevc', delete=False)  # Keep the file for later processing

    def __del__(self):
        """Dispose of resources upon deletion."""

        if self._annex_b_file is not None:
            self._annex_b_file.close()
            try:
                # Cleanup the temporary file
                os.remove(self._annex_b_file.name)
                self._annex_b_file = None
            except OSError:
                pass

        if self._nvenc_frame is not None:
            self._nvenc_frame.dispose()
            self._nvenc_frame = None

    @property
    def texture_id(self) -> int:
        """The OpenGL texture ID of the framebuffer's color attachment.
        """
        return self._frame_buffer.texture_id

    @contextlib.contextmanager
    def record(self, enabled: bool = True):
        """Record a frame within the context.

        Parameters
        ----------
        enabled : bool, optional
            Whether to record the frame or not, by default True.
        """

        # First invoke drawing calls and render to the framebuffer
        with self._frame_buffer.ctx():
            yield

        if not enabled:
            return

        # After exiting the context, process the current texture
        self._nvenc_frame.process_current_texture(self._frame_buffer.texture_id)

        # Encode the frame using NVENC
        bitstream = self._encoder.Encode(self._nvenc_frame)
        bitstream = bytearray(bitstream)

        # Write the encoded bitstream to the temporary file
        self._annex_b_file.write(bitstream)

    def finalize(self, out_file: str | pathlib.Path) -> None:
        """Finalize the recording and save the encoded video to a file.

        Parameters
        ----------
        out_file : str | pathlib.Path
            The output video file path.
        """

        if isinstance(out_file, str):
            out_file = pathlib.Path(out_file)
        out_file = out_file.resolve()

        # Flush the encoder to process any remaining frames
        bitstream = self._encoder.EndEncode()
        bitstream = bytearray(bitstream)
        self._annex_b_file.write(bitstream)

        # Ensure all data is written to the temporary file
        self._annex_b_file.flush()
        self._annex_b_file.close()  # Close to ensure data is written

        if os.path.getsize(self._annex_b_file.name) > 0:
            # Create output directory if it doesn't exist
            out_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert the Annex B bitstream to MP4 container using imageio
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            cmd = [
                ffmpeg_exe,
                '-y',                                     # Overwrite output file if it exists
                '-r', str(self._fps),                     # Input frame rate
                '-i', self._annex_b_file.name,            # Input file (Annex B bitstream)
                '-c:v', 'copy',                           # Copy the video stream without re-encoding
                '-an',                                    # No audio
                str(out_file),
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg failed to convert Annex B bitstream to MP4: {e}") from e
        else:
            print("No frames were recorded; skipping video file creation.")


# --------------------------------------------------------------------------------------------------------------
