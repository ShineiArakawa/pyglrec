"""
pyglrec
=======

Render OpenGL scenes to PyTorch tensors using OpenGL-CUDA interoperability.

Contact
-------

- Author: Shinei Arakawa
- Email: arakawashinei1115@gmail.com
"""

# ----------------------------------------------------------------------------
# Check Python version

import sys

if sys.version_info < (3, 11):
    raise ImportError('Python 3.11 or higher is required.')

# ----------------------------------------------------------------------------
# Check the version of this package

import importlib.metadata

try:
    __version__ = importlib.metadata.version('pyglrec')
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = '0.0.0'

# ----------------------------------------------------------------------------
# Import modules

from .frame_buffer import FrameBuffer
from .frame_recorder import NVENCFrameRecorder, UncompressedFrameCPURecorder
from .quad import Quad
from .shader import build_shader_program
from .util import EasyDict, StrEnum

__all__ = [
    '__version__',
    'EasyDict',
    'StrEnum',
    'build_shader_program',
    'FrameBuffer',
    'Quad',
    'UncompressedFrameCPURecorder',
    'NVENCFrameRecorder',
]

# ----------------------------------------------------------------------------
