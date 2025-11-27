"""
frame_recorder
==============

Recorder classes for OpenGL frame data.
"""


import abc
import contextlib
import pathlib
import platform
import subprocess
import typing

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
# Base Frame Recorder


class FrameRecorderBase(metaclass=abc.ABCMeta):
    def __init__(self, width: int, height: int):
        """Initialize the base frame recorder.

        Parameters
        ----------
        width : int
            The width of the frames to record.
        height : int
            The height of the frames to record.
        """

        self._width = width
        self._height = height
        self._frame_buffer = frame_buffer.FrameBuffer(width, height)

        # On the fly ffmpeg muxer process and pipe
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._ffmpeg_stdin: typing.IO | None = None

    def dispose(self):
        """Gracefully dispose of resources."""

        if self._ffmpeg_stdin is not None:
            self._ffmpeg_stdin.close()
            self._ffmpeg_stdin = None

        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.terminate()
            self._ffmpeg_proc = None

        if self._frame_buffer is not None:
            self._frame_buffer.dispose()
            self._frame_buffer = None

    @property
    def texture_id(self) -> int:
        """The OpenGL texture ID of the framebuffer's color attachment.
        """
        return self._frame_buffer.texture_id

    @abc.abstractmethod
    def _start_ffmpeg_muxer_proc(self) -> None:
        """Start the FFmpeg muxer process for encoding and saving video."""
        pass

    @contextlib.contextmanager
    def record(self, enabled: bool = True):
        """Record a frame within the context.

        Parameters
        ----------
        enabled : bool, optional
            Whether to record the frame or not, by default True.
        """

        raise NotImplementedError("Subclasses must implement the 'record' method.")

    @abc.abstractmethod
    def finalize(self) -> None:
        """Finalize the recording and save the video file."""
        pass


# --------------------------------------------------------------------------------------------------------------
# CPU Frame Recorder


class UncompressedFrameCPURecorder(FrameRecorderBase):
    """Recorder for uncompressed CPU frame data with on-the-fly FFmpeg muxing."""

    def __init__(
        self,
        width: int,
        height: int,
        out_file: str | pathlib.Path,
        fps: int = 60,
        codec: typing.Literal['h264', 'hevc'] = 'hevc',
        bitrate: str = '30M',
        chroma_format: typing.Literal['YUV420', 'YUV422', 'YUV444'] = 'YUV444',
    ):
        """
        Initialize the uncompressed frame recorder. Each frame is read back
        from the GPU to CPU and sent as rawvideo to FFmpeg via stdin, which
        encodes and muxes on-the-fly.

        Parameters
        ----------
        width : int
            The width of the frames to record.
        height : int
            The height of the frames to record.
        out_file : str | pathlib.Path
            The output video file path (e.g., 'output.mp4').
        fps : int, optional
            The frame rate of the recording, by default 60.
        codec : typing.Literal['h264', 'hevc'], optional
            The codec to use for encoding ('h264' or 'hevc'), by default 'hevc'.
        bitrate : str, optional
            The target bitrate for the output video file, by default '30M'.
        chroma_format : typing.Literal['YUV420', 'YUV422', 'YUV444'], optional
            The chroma subsampling format for encoding, by default 'YUV444'.

        Example
        -------
        >>> recorder = UncompressedFrameCPURecorder(
        ...     width=1920, height=1080,
        ...     out_file="output.mp4", fps=30)
        >>> for frame in range(300):
        ...     with recorder.record():
        ...         # OpenGL drawing calls here
        ...         pass
        >>> recorder.finalize()
        """

        super().__init__(width, height)

        self._fps = fps
        self._codec = codec
        self._bitrate = bitrate
        self._chroma_format = chroma_format

        if isinstance(out_file, str):
            out_file = pathlib.Path(out_file)
        self._out_file = out_file.resolve()
        self._out_file.parent.mkdir(parents=True, exist_ok=True)

        # Start ffmpeg muxer process
        self._start_ffmpeg_muxer_proc()

    def __del__(self):
        """Gracefully dispose of resources."""

        super().__del__()

    def _start_ffmpeg_muxer_proc(self) -> None:
        """Start the FFmpeg muxer process for encoding and saving video."""

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # FFmpeg command to read raw RGBA frames from stdin and encode to MP4
        codec = 'libx264' if self._codec == 'h264' else 'libx265'
        pix_fmt = self._chroma_format.lower() + 'p'  # e.g., 'yuv444p'

        cmd = [
            ffmpeg_exe,
            "-y",
            # Input settings
            "-f", "rawvideo",                       # Input format
            "-pix_fmt", "rgba",                     # Input pixel format
            "-s", f"{self._width}x{self._height}",  # Frame size
            "-framerate", str(self._fps),           # Frame rate
            "-i", "pipe:0",                         # Input from stdin
            # Encoding settings
            "-c:v", codec,                          # Video codec
            "-b:v", str(self._bitrate),             # Target bitrate
            "-g", "1",                              # GOP size = 1 (following original finalize settings)
            "-color_range", "2",                    # full range
            "-movflags", "+write_colr",             # Write color metadata
            "-pix_fmt", pix_fmt,                    # Output pixel format
            # Output settings
            "-vsync", "cfr",                        # Constant frame rate
            "-r", str(self._fps),                   # Output frame rate
            str(self._out_file),
        ]

        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if self._ffmpeg_proc.stdin is None:
            raise RuntimeError("Failed to open FFmpeg stdin")

        self._ffmpeg_stdin = self._ffmpeg_proc.stdin

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

        tex_data = np.flipud(tex_data)  # Flip vertically to match image coordinate system
        tex_data = tex_data.copy()      # Ensure data is contiguous

        # Write raw frame data to ffmpeg stdin
        self._ffmpeg_stdin.write(tex_data.tobytes())

    def finalize(self) -> None:
        """Finalize the recording and save all frames to a video file."""

        if self._ffmpeg_proc is None:
            return

        self._ffmpeg_stdin.flush()
        self._ffmpeg_stdin.close()

        # Wait for ffmpeg process to finish
        ret = self._ffmpeg_proc.wait()
        if ret != 0:
            msg = f"FFmpeg process failed with return code {ret}.\n"
            raise RuntimeError(msg)

        self._ffmpeg_proc = None
        self._ffmpeg_stdin = None

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


class NVENCFrameRecorder(FrameRecorderBase):
    """Recorder for GPU frame data using NVENC encoder."""

    def __init__(
        self,
        width: int,
        height: int,
        out_file: str | pathlib.Path,
        fps: int = 60,
        codec: typing.Literal['h264', 'hevc'] = 'hevc',
        preset: typing.Literal['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'] = 'P1',
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
        out_file : str | pathlib.Path
            The output MP4 video file path.
        fps : int, optional
            The frame rate of the recording, by default 60.
        codec : typing.Literal['h264', 'hevc'], optional
            The codec to use for encoding ('h264' or 'hevc'), by default 'hevc'.
        preset : typing.Literal['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'], optional
            The NVENC preset to use ('P1' to 'P7'), by default 'P3'.
        avg_bitrate : int, optional
            The average bitrate for encoding in bits per second, by default 20_000_000 (20 Mbps).
        max_bitrate : int, optional
            The maximum bitrate for encoding in bits per second, by default 30_000_000 (30 Mbps).
        chroma_format: typing.Literal['NV12', 'YUV444'], optional
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
        >>> recorder.finalize()
        """

        # Check if nvcodec is available
        if nvcodec is None:
            raise RuntimeError("PyNvVideoCodec is not available. NVENC is not supported on this platform.")

        super().__init__(width, height)

        self._out_file = out_file
        self._fps = fps
        self._codec = codec

        # Initialize CUDA-OpenGL interop plugin
        self._plugin = cuda_gl_interop.get_cuda_plugin()
        if self._plugin is None:
            raise RuntimeError("CUDA-OpenGL interop plugin is not available. Check if 'nvcc' is installed and configured correctly.")

        # Get and set CUDA device for the current OpenGL context
        cuda_device_id = int(self._plugin.get_cuda_device_for_current_OpenGL_context())
        self._plugin.set_cuda_device_for_current_OpenGL_context(cuda_device_id)  # Set the same device as current OpenGL context

        # Initialize NVENC encoder
        encoder_opts = util.EasyDict(preset=preset)

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

        # Start ffmpeg muxer process
        self._start_ffmpeg_muxer_proc()

    def dispose(self):
        """Gracefully dispose of resources."""

        if self._nvenc_frame is not None:
            self._nvenc_frame.dispose()
            self._nvenc_frame = None

        super().dispose()

    def _start_ffmpeg_muxer_proc(self) -> None:
        if isinstance(self._out_file, str):
            self._out_file = pathlib.Path(self._out_file)
        self._out_file = self._out_file.resolve()
        self._out_file.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        input_fmt = 'h264' if self._codec == 'h264' else 'hevc'
        cmd = [
            ffmpeg_exe,
            '-y',                                   # Overwrite output file if it exists
            # Input settings
            '-f', input_fmt,                        # Input Annex-B format
            '-framerate', str(self._fps),           # Frame rate
            '-i', 'pipe:0',                         # Input from stdin
            # Output settings
            '-c:v', 'copy',                         # Copy without re-encoding
            '-vsync', 'cfr',                        # Constant frame rate
            '-r', str(self._fps),                   # Output frame rate
            '-an',
            str(self._out_file),
        ]

        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if self._ffmpeg_proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin")

        self._ffmpeg_stdin = self._ffmpeg_proc.stdin

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

        # Write the encoded bitstream to the ffmpeg stdin for muxing
        self._ffmpeg_stdin.write(bitstream)

    def finalize(self, dispose: bool = True) -> None:
        """Finalize the recording and save the encoded video to a file.

        Parameters
        ----------
        dispose : bool, optional
            Whether to dispose of resources after finalization, by default True.
        """

        if self._ffmpeg_proc is None:
            return

        # Flush the encoder to process any remaining frames
        bitstream = self._encoder.EndEncode()
        bitstream = bytearray(bitstream)

        if len(bitstream) > 0:
            # Write the final encoded bitstream to the ffmpeg stdin for muxing
            self._ffmpeg_stdin.write(bitstream)

        self._ffmpeg_stdin.flush()
        self._ffmpeg_stdin.close()

        # Wait for ffmpeg process to finish
        ret = self._ffmpeg_proc.wait()
        if ret != 0:
            msg = f"FFmpeg process failed with return code {ret}.\n"
            raise RuntimeError(msg)

        if dispose:
            self.dispose()


# --------------------------------------------------------------------------------------------------------------
