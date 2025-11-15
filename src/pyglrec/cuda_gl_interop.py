"""
CUDA-OpenGL inter-operability
=============================

This module provides a way to upload images directly from CUDA tensors to OpenGL VBOs using a custom PyTorch extension.

LICENSE
-------

This file includes modified codes from:

- Erik Härkönen's 'PyViewer' library (licensed under CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/): https://github.com/harskish/pyviewer.git
"""


# Code graciously provided by Pauli Kemppinen (github.com/msqrt)

import functools
import glob
import hashlib
import importlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import sysconfig

import pybind11.commands as pybind11_commands

# --------------------------------------------------------------------------------------------------------------------------------
# Pluging loader


@functools.cache
def _get_plugin_impl(
    plugin_name: str,
    source_files: tuple[str, ...] | tuple[pathlib.Path, ...] | str,
    source_folder: str | pathlib.Path = '.',
    ldflags: tuple = None,
    extra_cflags: tuple = (),
    cuda: bool = True,
    verbose=True
):
    """Build and load a C++/CUDA plugin module.

    Parameters
    ----------
    plugin_name : str
        The name of the plugin module.
    source_files : tuple[str, ...] | tuple[pathlib.Path, ...] | str
        The source file(s) for the plugin.
    source_folder : str | pathlib.Path, optional
        The folder containing the source files, by default '.'.
    ldflags : tuple, optional
        Additional linker flags, by default None.
    extra_cflags : tuple, optional
        Additional compiler flags, by default ().
    cuda : bool, optional
        Whether to enable CUDA support, by default True.
    verbose : bool, optional
        Whether to print verbose build output, by default True.
    """

    # normalize
    if isinstance(source_files, str):
        source_files = (pathlib.Path(source_files),)
    elif isinstance(source_files, list):
        source_files = tuple([pathlib.Path(f) for f in source_files])

    source_folder = pathlib.Path(source_folder).resolve()

    # --------------------------------------------------------------------------------------------------------------------------
    # Platform compiler flags and settings

    system = platform.system()

    python_include_dir = sysconfig.get_path('include')
    include_dirs = [
        pybind11_commands.get_include(),
        python_include_dir if python_include_dir else "",
        "/usr/local/include",
        str(source_folder),
    ]
    include_dirs = [p for p in include_dirs if p]  # drop missing paths

    if system == 'Linux':
        cflags = ['-O3', '-std=c++17', '-fPIC', *extra_cflags]
        ldflags = ['-lGL', '-lEGL'] if ldflags is None else list(ldflags)
        shared_ext = '.so'
        cxx = 'g++'
    elif system == 'Darwin':
        cflags = ['-O3', '-std=c++17', '-DGL_SILENCE_DEPRECATION', *extra_cflags]
        ldflags = ['-framework', 'OpenGL', '-framework', 'Cocoa'] if ldflags is None else list(ldflags)
        shared_ext = '.so'
        cxx = 'clang++'
    elif system == 'Windows':
        shared_ext = '.pyd'
        cxx = 'cl.exe'

        # Windows flags
        libs = ['user32', 'opengl32']
        if ldflags is None:
            ldflags = ['/DEFAULTLIB:' + x for x in libs]
        else:
            ldflags = list(ldflags)
        cflags = ['/O2', '/DWIN32', '/std:c++17', '/permissive-', '/w', *extra_cflags]

        # Search for cl.exe
        def find_cl_path():
            for maybe_x86 in ["", " (x86)"]:
                for edition in ['Community', 'Enterprise', 'Professional', 'BuildTools']:
                    paths = sorted(glob.glob(f"C:/Program Files{maybe_x86}/Microsoft Visual Studio/*/{edition}/VC/Tools/MSVC/*/bin/Hostx64/x64"), reverse=True)
                    if paths:
                        return paths[0]

        def find_msvc_include_dir(cl_exe_path: pathlib.Path | None) -> pathlib.Path | None:
            if cl_exe_path is None:
                return None
            # cl.exe usually lives in .../VC/Tools/MSVC/<ver>/bin/Hostx64/x64
            try:
                toolset_dir = cl_exe_path.parent.parent.parent.parent
            except IndexError:
                return None
            include_dir = toolset_dir / 'include'
            return include_dir if include_dir.exists() else None

        def find_msvc_lib_dir(cl_exe_path: pathlib.Path | None) -> pathlib.Path | None:
            include_dir = find_msvc_include_dir(cl_exe_path)
            if include_dir is None:
                return None
            lib_dir = include_dir.parent / 'lib' / 'x64'
            return lib_dir if lib_dir.exists() else None

        def collect_windows_kit_includes() -> list[str]:
            include_paths: list[str] = []
            seen: set[str] = set()

            def add_if_exists(path: pathlib.Path | None):
                if path is None:
                    return
                path_str = str(path)
                if path.exists() and path_str not in seen:
                    include_paths.append(path_str)
                    seen.add(path_str)

            # Windows SDK env vars (if developer tools prompt was used)
            sdk_dir = os.environ.get('WindowsSdkDir')
            sdk_version = os.environ.get('WindowsSDKVersion')
            if sdk_dir and sdk_version:
                normalized_version = sdk_version.strip('\\/')
                sdk_base = pathlib.Path(sdk_dir) / 'Include' / normalized_version
                for sub in ['ucrt', 'shared', 'um', 'winrt', 'cppwinrt']:
                    add_if_exists(sdk_base / sub)

            # Fall back to default installation directories to support plain PowerShell sessions.
            for major_version in ['11', '10']:
                base = pathlib.Path(f"C:/Program Files (x86)/Windows Kits/{major_version}/Include")
                if not base.exists():
                    continue
                version_dirs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
                for version_dir in version_dirs:
                    before = len(include_paths)
                    for sub in ['ucrt', 'shared', 'um', 'winrt', 'cppwinrt']:
                        add_if_exists(version_dir / sub)
                    if len(include_paths) > before:
                        break  # use the latest version that has headers
                if include_paths:
                    break

            return include_paths

        def collect_windows_kit_lib_dirs() -> list[str]:
            lib_paths: list[str] = []
            seen: set[str] = set()

            def add_if_exists(path: pathlib.Path | None):
                if path is None:
                    return
                path_str = str(path)
                if path.exists() and path_str not in seen:
                    lib_paths.append(path_str)
                    seen.add(path_str)

            sdk_dir = os.environ.get('WindowsSdkDir')
            sdk_version = os.environ.get('WindowsSDKVersion')
            if sdk_dir and sdk_version:
                normalized_version = sdk_version.strip('\\/')
                base = pathlib.Path(sdk_dir) / 'Lib' / normalized_version
                for sub in ['ucrt', 'um']:
                    add_if_exists(base / sub / 'x64')

            for major_version in ['11', '10']:
                base = pathlib.Path(f"C:/Program Files (x86)/Windows Kits/{major_version}/Lib")
                if not base.exists():
                    continue
                version_dirs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
                for version_dir in version_dirs:
                    before = len(lib_paths)
                    for sub in ['ucrt', 'um']:
                        add_if_exists(version_dir / sub / 'x64')
                    if len(lib_paths) > before:
                        break
                if lib_paths:
                    break

            return lib_paths

        def discover_python_lib() -> tuple[pathlib.Path | None, str | None]:
            def normalize_libname(name: str) -> list[str]:
                names = [name]
                if name.lower().endswith('.dll'):
                    names.append(name[:-4] + '.lib')
                return names

            major = sys.version_info.major
            minor = sys.version_info.minor

            library_cfg = sysconfig.get_config_var('LIBRARY')
            libname_candidates: list[str] = []
            if library_cfg:
                libname_candidates.extend(normalize_libname(library_cfg))
            libname_candidates.extend([
                f"python{major}{minor}.lib",
                f"python{major}.lib",
                f"python{major}.dll",
            ])

            seen_names: set[str] = set()
            libname_candidates = [name for name in libname_candidates if not (name in seen_names or seen_names.add(name))]

            dir_candidates: list[pathlib.Path] = []
            for key in ('LIBDIR', 'LIBPL'):
                path = sysconfig.get_config_var(key)
                if path:
                    dir_candidates.append(pathlib.Path(path))

            for prefix_attr in ('base_prefix', 'prefix'):
                prefix = getattr(sys, prefix_attr, None)
                if prefix:
                    for sub in ('libs', 'Libs'):
                        dir_candidates.append(pathlib.Path(prefix) / sub)

            seen_paths: set[str] = set()
            deduped_dirs: list[pathlib.Path] = []
            for candidate in dir_candidates:
                candidate_str = str(candidate)
                if candidate_str and candidate_str not in seen_paths:
                    deduped_dirs.append(candidate)
                    seen_paths.add(candidate_str)

            for candidate in deduped_dirs:
                for libname in libname_candidates:
                    if (candidate / libname).exists():
                        return candidate, libname

            fallback_dir = deduped_dirs[0] if deduped_dirs else None
            fallback_name = libname_candidates[0] if libname_candidates else None
            return fallback_dir, fallback_name

        cl_full_path = shutil.which('cl.exe')
        # If cl.exe is not on path, try to find it.
        if cl_full_path is None and os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path
            cl_full_path = str(pathlib.Path(cl_path) / 'cl.exe')
        elif cl_full_path is None:
            # where succeeded but shutil.which failed (rare), resolve via where output later.
            cl_full_path = shutil.which('cl.exe')

        msvc_include_dir = find_msvc_include_dir(pathlib.Path(cl_full_path) if cl_full_path else None)
        if msvc_include_dir is not None:
            include_dirs.append(str(msvc_include_dir))
            msvc_lib_dir = find_msvc_lib_dir(pathlib.Path(cl_full_path) if cl_full_path else None)
            if msvc_lib_dir is not None:
                ldflags.insert(0, '/LIBPATH:' + str(msvc_lib_dir))

        include_dirs.extend(collect_windows_kit_includes())
        for kit_lib_dir in collect_windows_kit_lib_dirs():
            if '/LIBPATH:' + kit_lib_dir not in ldflags:
                ldflags.append('/LIBPATH:' + kit_lib_dir)

        python_lib_dir, python_lib_name = discover_python_lib()
        if python_lib_dir is not None:
            lib_flag = '/LIBPATH:' + str(python_lib_dir)
            if lib_flag not in ldflags:
                ldflags.append(lib_flag)
        if python_lib_name is not None:
            default_lib = '/DEFAULTLIB:' + python_lib_name
            if default_lib not in ldflags:
                ldflags.append(default_lib)
    else:
        raise RuntimeError("Unsupported OS")

    # --------------------------------------------------------------------------------------------------------------------------
    # CUDA settings

    nvcc = None
    cuda_version = 'nocuda'
    if cuda:
        cuda_path = os.getenv('CUDA_PATH')
        if cuda_path is not None and os.path.exists(os.path.join(cuda_path, 'bin', 'nvcc.exe' if system == 'Windows' else 'nvcc')):
            nvcc = os.path.join(cuda_path, 'bin', 'nvcc.exe' if system == 'Windows' else 'nvcc')

            # As of Python 3.8, cwd and $PATH are no longer searched for DLLs
            if system == 'Windows':
                os.add_dll_directory(os.path.join(cuda_path, 'bin'))

            # Try to get CUDA version
            cmd = [nvcc, '--version']
            try:
                output = subprocess.check_output(cmd, universal_newlines=True)
                for line in output.split('\n'):
                    if 'release' in line:
                        # Like: Cuda compilation tools, release 12.9, V12.9.86
                        cuda_version = 'cu' + line.strip().split('release')[-1].split(',')[0].strip()
                        break
            except Exception:
                cuda_version = 'unknown'

            # Add CUDA include dirs and libraries
            include_dirs.append(os.path.join(cuda_path, 'include'))

            if system == 'Windows':
                cuda_lib_dir = pathlib.Path(cuda_path) / 'lib' / 'x64'
                if cuda_lib_dir.exists():
                    lib_flag = '/LIBPATH:' + str(cuda_lib_dir)
                    if lib_flag not in ldflags:
                        ldflags.append(lib_flag)

                for lib_name in ['cudart.lib', 'cuda.lib']:
                    default_lib = '/DEFAULTLIB:' + lib_name
                    if default_lib not in ldflags:
                        ldflags.append(default_lib)
            elif system in ('Linux', 'Darwin'):
                cuda_lib_dir = os.path.join(cuda_path, 'lib64')
                if os.path.isdir(cuda_lib_dir):
                    flag = f'-L{cuda_lib_dir}'
                    if flag not in ldflags:
                        ldflags.append(flag)

                # Add cudart
                if '-lcudart' not in ldflags:
                    ldflags.append('-lcudart')
                # Add cuda
                if '-lcuda' not in ldflags:
                    ldflags.append('-lcuda')
        if nvcc is None:
            raise RuntimeError("CUDA_PATH is not set or nvcc not found in CUDA_PATH/bin")

    # --------------------------------------------------------------------------------------------------------------------------
    # Build directory

    python_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    build_dir = pathlib.Path(__file__).parent / '__pycache__' / 'cpp_extensions' / f'{python_version}_{cuda_version}'
    build_dir.mkdir(parents=True, exist_ok=True)
    module_path = build_dir / f"{plugin_name}{shared_ext}"

    # --------------------------------------------------------------------------------------------------------------------------
    # Compute hash

    hash_dict = {
        'cflags': cflags,
        'ldflags': ldflags,
        'include_dirs': [str(d) for d in include_dirs],
    }

    # Add source files
    hash_dict['source_files'] = {}
    for src in source_files:
        src = source_folder / src
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        hash_dict['source_files'][str(src)] = content

    hash_str = hashlib.sha256(json.dumps(hash_dict, sort_keys=True).encode('utf-8')).hexdigest()

    # --------------------------------------------------------------------------------------------------------------------------
    # Try loading prebuilt

    build_hash_file = module_path.with_suffix('.buildhash')

    if module_path.exists() and build_hash_file.exists():
        # check build hash
        with open(build_hash_file, 'r') as f:
            existing_hash = f.read()
        if existing_hash == hash_str:
            # load module
            prev_path = [*sys.path]
            try:
                sys.path.append(str(build_dir))
                return importlib.import_module(plugin_name)  # .pyd on Windows
            except ImportError as e:
                if e.msg.startswith('DLL load failed') and cuda:
                    print('DLL load failed, make sure all DLLs required by module are available (NB: $PATH and cwd are not searched)')
            except Exception as e:
                pass  # could not load cached, proceed to compilation step
            finally:
                sys.path = prev_path

    # --------------------------------------------------------------------------------------------------------------------------
    # Build object files

    print('Compiling CUDA-PyTorch interop plugin. This may take a while ...')

    obj_suffix = ".obj" if system == "Windows" else ".o"
    obj_files = []

    for src in source_files:
        src = source_folder / src
        obj = build_dir / (src.stem + obj_suffix)

        if verbose:
            print(f"[compile] {src} -> {obj}")

        if src.suffix == ".cu":
            # CUDA compile
            cmd = [
                nvcc, "-c", str(src),
                "-o", str(obj),
                "-Xcompiler", "-fPIC",
                "-std=c++17",
                "-O3",
            ]
            for inc in include_dirs:
                cmd.extend(["-I", str(inc)])
        else:
            # C++ compile
            if system == "Windows":
                cmd = [cxx, "/c", str(src), f"/Fo{obj}", *cflags]
            else:
                cmd = [cxx, "-c", str(src), "-o", str(obj), *cflags]
            for inc in include_dirs:
                cmd.extend(["-I", str(inc)])

        subprocess.check_call(cmd)
        obj_files.append(str(obj))

    # --------------------------------------------------------------------------------------------------------------------------
    # Link

    if verbose:
        print(f"[link] -> {module_path}")

    if system == "Windows":
        # cl/link on Windows
        link_cmd = [
            "link.exe",
            "/DLL",
            "/OUT:" + str(module_path),
        ] + obj_files + list(ldflags)

    else:
        # gcc/clang
        link_cmd = [
            cxx,
            "-shared",
            "-o", str(module_path),
        ] + obj_files + list(ldflags)

    subprocess.check_call(link_cmd)

    print(f"Successfully built plugin: {module_path}")

    # --------------------------------------------------------------------------------------------------------------------------
    # Write build hash

    with open(build_hash_file, 'w') as f:
        f.write(hash_str)

    # --------------------------------------------------------------------------------------------------------------------------
    # Import

    if str(build_dir) not in sys.path:
        sys.path.append(str(build_dir))

    return importlib.import_module(plugin_name)


@functools.lru_cache
def get_cuda_plugin():
    """Get the CUDA-OpenGL interop plugin module."""

    try:
        return _get_plugin_impl(
            'cuda_gl_interop',
            ('cuda_gl_interop.cpp', 'rgba_conversion.cu'),
            pathlib.Path(__file__).parent / 'custom_ops'
        )
    except Exception as e:
        print('Failed to build CUDA-GL plugin:', e)
        return None

# --------------------------------------------------------------------------------------------------------------------------------
# CUDA-OpenGL interop functions


class CUDAMemory1D:
    """Class for managing 1D CUDA device memory."""

    def __init__(self):
        self._plugin = get_cuda_plugin()
        if self._plugin is None:
            raise RuntimeError("CUDA-OpenGL interop plugin is not available")

        self._ptr = 0
        self._num_bytes = 0

    def allocate(self, num_bytes: int):
        if self._ptr != 0:
            self.free()

        ptr = self._plugin.allocate_device_memory_1d(num_bytes)
        if not isinstance(ptr, int):
            raise RuntimeError("CUDA plugin returned unexpected pointer type")

        self._ptr = ptr
        self._num_bytes = num_bytes

    def free(self):
        if self._ptr != 0:
            self._plugin.free_device_memory(self._ptr)
            self._ptr = 0
            self._num_bytes = 0

    @property
    def ptr(self):
        return self._ptr

    @property
    def num_bytes(self):
        return self._num_bytes


class CUDAMemory2D_URGB4:
    """Class for managing linear 2D CUDA device memory for URGB4 format."""

    def __init__(self):
        self._plugin = get_cuda_plugin()
        if self._plugin is None:
            raise RuntimeError("CUDA-OpenGL interop plugin is not available")

        self._ptr = 0
        self._pitch = 0
        self._width = 0
        self._height = 0

    def allocate(self, width: int, height: int):
        if self._ptr != 0:
            self.free()

        ptr, pitch = self._plugin.allocate_device_memory_2d(width, height, 4)  # 4 bytes per pixel for uchar4 struct
        if not isinstance(ptr, int) or not isinstance(pitch, int):
            raise RuntimeError("CUDA plugin returned unexpected pointer type")

        self._ptr = ptr
        self._pitch = pitch
        self._width = width
        self._height = height

    def free(self):
        if self._ptr != 0:
            self._plugin.free_device_memory(self._ptr)
            self._ptr = 0
            self._pitch = 0
            self._width = 0
            self._height = 0

    @property
    def ptr(self):
        return self._ptr

    @property
    def pitch(self):
        return self._pitch

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
