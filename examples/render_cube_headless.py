"""
render_cube_headless
====================

This example demonstrates how to render a rotating colored cube using a headless EGL context and record the frames to a video file.

Unfortunately, EGL is not supported on Windows with PyOpenGL at the moment. Please run this example on Linux or macOS.
"""

# isort: skip_file
# autopep8: off

import os

import tqdm

os.environ["PYOPENGL_PLATFORM"] = "egl"  # Make sure to set 'PYOPENGL_PLATFORM' before importing OpenGL

import ctypes
import math
import platform

if platform.system() in ["Windows", "Darwin"]:
    raise RuntimeError("EGL is not supported on Windows or macOS by default. If you have custom EGL support, please remove this check and try again.")

import click
import cube_object
import gl_utils
import OpenGL.EGL as egl
import OpenGL.EGL.EXT.device_base as egl_device_base
import OpenGL.EGL.EXT.platform_device as egl_platform_device
import OpenGL.GL as gl
import OpenGL.platform as gl_platform
import pyglm.glm as glm

import pyglrec

# autopep8: on

print("\n[PyOpenGL Platform Info]")
print("    PyOpenGL platform :", gl_platform.PLATFORM)
print("    Platform class    :", gl_platform.PLATFORM.__class__.__name__)
print("")

# --------------------------------------------------------------------------------------------
# OpenGL version and GLSL version settings

if platform.system() == "Darwin":
    # On macOS, OpenGL 4.1 is the highest supported version
    OPENGL_VERSION_MAJOR = 4
    OPENGL_VERSION_MINOR = 1
else:
    OPENGL_VERSION_MAJOR = 4
    OPENGL_VERSION_MINOR = 6


# --------------------------------------------------------------------------------------------
# EGL context creation


def create_egl_context(width: int, height: int, device_id: int | None = None):
    if device_id is None:
        device_id = 0
    else:
        plugin = pyglrec.get_cuda_plugin()
        cuda_to_egl_map = plugin.get_cuda_to_egl_device_map()
        if device_id not in cuda_to_egl_map:
            raise RuntimeError(f"CUDA device ID {device_id} does not have EGL device mapping for CUDA/OpenGL interop.")
        device_id = cuda_to_egl_map[device_id]

    # Get EGL display
    max_num_devices = 16
    devices = egl_device_base.egl_get_devices(max_num_devices)
    assert len(devices) > device_id, f"Requested device_id {device_id} but only {len(devices)} devices found"

    egl_device = devices[device_id]
    print(f"Using EGL device {device_id} for EGL rendering context\n")

    display = egl.eglGetPlatformDisplayEXT(egl_platform_device.EGL_PLATFORM_DEVICE_EXT, egl_device, None)

    # display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    if display == egl.EGL_NO_DISPLAY:
        raise RuntimeError("eglGetDisplay failed")

    # Seup EGL version
    major, minor = egl.EGLint(), egl.EGLint()
    if not egl.eglInitialize(display, ctypes.pointer(major), ctypes.pointer(minor)):
        raise RuntimeError("eglInitialize failed")

    # Choose EGL config
    config_attribs = [
        # autopep8: off
        egl.EGL_SURFACE_TYPE,         egl.EGL_PBUFFER_BIT,
        egl.EGL_RED_SIZE,             8,
        egl.EGL_GREEN_SIZE,           8,
        egl.EGL_BLUE_SIZE,            8,
        egl.EGL_DEPTH_SIZE,           24,
        egl.EGL_RENDERABLE_TYPE,      egl.EGL_OPENGL_BIT,
        egl.EGL_SAMPLE_BUFFERS,       1,
        egl.EGL_SAMPLES,              4,  # MSAA 4x
        egl.EGL_NONE,
        # autopep8: on
    ]
    config_attribs = (egl.EGLint * len(config_attribs))(*config_attribs)

    config = egl.EGLConfig()
    num_configs = egl.EGLint()
    if not egl.eglChooseConfig(display, config_attribs, ctypes.pointer(config), 1, ctypes.pointer(num_configs)):
        raise RuntimeError("eglChooseConfig failed")
    if num_configs.value == 0:
        raise RuntimeError("no EGL configs")

    # Create EGL pbuffer surface
    pbuffer_attribs = [
        egl.EGL_WIDTH,  width,
        egl.EGL_HEIGHT, height,
        egl.EGL_NONE,
    ]
    pbuffer_attribs = (egl.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)

    surface = egl.eglCreatePbufferSurface(display, config, pbuffer_attribs)
    if surface == egl.EGL_NO_SURFACE:
        raise RuntimeError("eglCreatePbufferSurface failed")

    # Bind OpenGL API
    if not egl.eglBindAPI(egl.EGL_OPENGL_API):
        raise RuntimeError("eglBindAPI failed")

    # Create EGL context
    context_attribs = [
        egl.EGL_CONTEXT_MAJOR_VERSION, OPENGL_VERSION_MAJOR,
        egl.EGL_CONTEXT_MINOR_VERSION, OPENGL_VERSION_MINOR,
        egl.EGL_CONTEXT_OPENGL_PROFILE_MASK, egl.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        egl.EGL_NONE,
    ]
    context_attribs = (egl.EGLint * len(context_attribs))(*context_attribs)

    context = egl.eglCreateContext(display, config, egl.EGL_NO_CONTEXT, context_attribs)
    if context == egl.EGL_NO_CONTEXT:
        raise RuntimeError("eglCreateContext failed")

    # Make context current
    if not egl.eglMakeCurrent(display, surface, surface, context):
        raise RuntimeError("eglMakeCurrent failed")

    # Print version info
    print(f"[OpenGL/EGL Version Info]")
    print(f"    EGL version     : {major.value}.{minor.value}")
    print(f"    OpenGL version  : {gl.glGetString(gl.GL_VERSION).decode()}")
    print(f"    OpenGL vendor   : {gl.glGetString(gl.GL_VENDOR).decode()}")
    print(f"    OpenGL renderer : {gl.glGetString(gl.GL_RENDERER).decode()}")
    print(f"    GLSL version    : {gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode()}")
    print(f'')

    return display, surface, context


@click.command()
@click.option("--width", default=2560, help="Window width")
@click.option("--height", default=1440, help="Window height")
@click.option("--n_frames", default=500, help="Number of frames to render")
@click.option("--out_dir", default="./outputs/render_cube_headless", help="Output directory to save the recorded video")
@click.option("--fps", default=60.0, help="Frame rate limit")
@click.option("--out_fps", default=60.0, help="Output frame rate for recording")
@click.option("--rot_speed", default=60.0, help="Cube rotation speed in degrees per frame")
@click.option("--nvenc", is_flag=True, help="Enable NVENC frame recording (requires NVIDIA GPU)")
@click.option("--fmt", default='YUV444', type=click.Choice(['NV12', 'YUV444']), help="Chroma format for NVENC recorder")
@click.option("--bitrate", default='10M', type=str, help="Bitrate for NVENC recorder (in bits per second)")
@click.option("--device_id", default=None, type=int, help="EGL device ID to use for rendering. This needs CUDA/OpenGL interop support.")
def main(**args):
    """Render a rotating colored cube using headless EGL context and record the frames to a video file.
    """

    args = pyglrec.EasyDict(args)

    window_width = args.width
    window_height = args.height

    # --------------------------------------------------------------------------------------------
    # Create EGL context

    display, surface, context = create_egl_context(window_width, window_height, device_id=args.device_id)

    # --------------------------------------------------------------------------------------------
    # Camera setup

    camera_pos = glm.vec3(5.0, 0.0, 0.0)
    camera_lookat = glm.vec3(0.0, 0.0, 0.0)
    camera_up = glm.vec3(0.0, 1.0, 0.0)
    view_mat = glm.lookAt(camera_pos, camera_lookat, camera_up)

    model_rot_mat = glm.mat4(1.0)
    model_trans_mat = glm.mat4(1.0)
    model_scale_mat = glm.mat4(1.0)

    proj_mat = gl_utils.perspective(window_width / window_height)

    light_rot_mat = glm.mat4(1.0)
    light_pos = glm.vec3(10.0, 10.0, 10.0)

    # --------------------------------------------------------------------------------------------
    # Create cube object

    cube_obj = cube_object.CubeObject()

    # --------------------------------------------------------------------------------------------
    # Create recorder

    out_file = os.path.join(args.out_dir, 'output_headless_nvenc.mp4' if args.nvenc else 'output_headless.mp4')
    if args.nvenc:
        recorder = pyglrec.NVENCFrameRecorder(window_width, window_height, out_file=out_file, fps=args.out_fps, avg_bitrate=args.bitrate, chroma_format=args.fmt)
    else:
        recorder = pyglrec.UncompressedFrameCPURecorder(window_width, window_height, out_file=out_file, fps=args.out_fps, bitrate=args.bitrate)

    # --------------------------------------------------------------------------------------------
    # Setup states

    gl.glEnable(gl.GL_DEPTH_TEST)   # Enable depth testing
    gl.glEnable(gl.GL_MULTISAMPLE)  # Enable multisampling on default framebuffer

    # --------------------------------------------------------------------------------------------
    # Main loop

    for frame_idx in tqdm.trange(args.n_frames, desc='Rendering frames ... ', unit='frame'):
        # Compute transformation matrices
        model_mat = model_trans_mat * model_rot_mat * model_scale_mat
        mv_mat = view_mat * model_mat
        mvp_mat = proj_mat * mv_mat
        light_pos_camera_space = glm.vec3(view_mat * light_rot_mat * glm.vec4(light_pos, 1.0))

        # Render and record frame
        with recorder.record():
            gl.glViewport(0, 0, window_width, window_height)

            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # Draw cube
            cube_obj.draw(mvp_mat, mv_mat, glm.transpose(glm.inverse(glm.mat3(mv_mat))), light_pos_camera_space)

        # Animate cube
        angle_x = math.sin(frame_idx / args.fps * 0.6) * args.rot_speed
        angle_y = math.sin(frame_idx / args.fps * 0.9) * args.rot_speed
        angle_z = math.sin(frame_idx / args.fps * 1.1) * args.rot_speed

        rot_quaternion_x = glm.angleAxis(glm.radians(angle_x), glm.vec3(1.0, 0.0, 0.0))
        rot_quaternion_y = glm.angleAxis(glm.radians(angle_y), glm.vec3(0.0, 1.0, 0.0))
        rot_quaternion_z = glm.angleAxis(glm.radians(angle_z), glm.vec3(0.0, 0.0, 1.0))
        rot_quaternion = rot_quaternion_x * rot_quaternion_y * rot_quaternion_z
        model_rot_mat = glm.mat4_cast(rot_quaternion)

        # Also rotate light at different speed
        angle_x = math.sin(frame_idx / args.fps * 1.5) * args.rot_speed
        angle_y = math.sin(frame_idx / args.fps * 1.8) * args.rot_speed
        angle_z = math.sin(frame_idx / args.fps * 2.1) * args.rot_speed

        rot_quaternion_x = glm.angleAxis(glm.radians(angle_x), glm.vec3(1.0, 0.0, 0.0))
        rot_quaternion_y = glm.angleAxis(glm.radians(angle_y), glm.vec3(0.0, 1.0, 0.0))
        rot_quaternion_z = glm.angleAxis(glm.radians(angle_z), glm.vec3(0.0, 0.0, 1.0))
        rot_quaternion = rot_quaternion_x * rot_quaternion_y * rot_quaternion_z
        light_rot_mat = glm.mat4_cast(rot_quaternion)

    recorder.finalize()

    print(f'Recorded video saved to: {out_file}')

    # --------------------------------------------------------------------------------------------
    # Cleanup OpenGL resources and EGL context

    cube_obj.dispose()

    egl.eglMakeCurrent(display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT)
    egl.eglDestroyContext(display, context)
    egl.eglDestroySurface(display, surface)
    egl.eglTerminate(display)

    print(f'Bye!')


if __name__ == "__main__":
    main()
