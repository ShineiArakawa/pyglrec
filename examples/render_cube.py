"""
render_cube
===========

This example demonstrates how to render a rotating colored cube using OpenGL and record the frames to a video file.
"""

import math
import os
import platform
import time

import click
import cube_object
import gl_utils
import glfw
import OpenGL.GL as gl
import OpenGL.platform as gl_platform
import pyglm.glm as glm

import pyglrec

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
# Main function

rot_speed = 60.0  # degrees per second


@click.command()
@click.option("--width", default=2560, help="Window width")
@click.option("--height", default=1440, help="Window height")
@click.option("--out_dir", default="./outputs/render_cube", help="Output directory to save the recorded video")
@click.option("--fps", default=60.0, help="Frame rate limit")
@click.option("--out_fps", default=60.0, help="Output frame rate for recording")
@click.option("--nvenc", is_flag=True, help="Enable NVENC frame recording (requires NVIDIA GPU)")
@click.option("--disabled", is_flag=True, help="Disable frame recording")
def main(**args):
    """Render a rotating colored cube using OpenGL and record the frames to a video file.
    """

    args = pyglrec.EasyDict(args)

    window_width = args.width
    window_height = args.height

    # --------------------------------------------------------------------------------------------
    # Create GLFW window and OpenGL context

    if not glfw.init():
        raise Exception("GLFW can't be initialized")

    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, OPENGL_VERSION_MAJOR)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, OPENGL_VERSION_MINOR)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Disable window resize
    glfw.window_hint(glfw.RESIZABLE, gl.GL_FALSE)

    window = glfw.create_window(window_width, window_height, "Render Cube", None, None)

    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")

    glfw.make_context_current(window)

    # Print version info
    print(f"[OpenGL/GLFW Version Info]")
    print(f'    OpenGL version  : {gl.glGetString(gl.GL_VERSION).decode()}')
    print(f"    OpenGL vendor   : {gl.glGetString(gl.GL_VENDOR).decode()}")
    print(f"    OpenGL renderer : {gl.glGetString(gl.GL_RENDERER).decode()}")
    print(f"    GLSL version    : {gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode()}")
    print(f'    GLFW version    : {glfw.get_version_string().decode()}')
    print(f'')

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
    # Key callback

    def key_callback(window, key, scancode, action, mods):
        global rot_speed

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_UP and (action == glfw.PRESS or action == glfw.REPEAT):
            rot_speed += 1.0
        elif key == glfw.KEY_DOWN and (action == glfw.PRESS or action == glfw.REPEAT):
            rot_speed -= 1.0

    glfw.set_key_callback(window, key_callback)

    # --------------------------------------------------------------------------------------------
    # Create cube object

    cube_obj = cube_object.CubeObject()

    # --------------------------------------------------------------------------------------------
    # Create quad stuffs for rendering FBO texture to default framebuffer

    quad_obj = pyglrec.Quad()

    # --------------------------------------------------------------------------------------------
    # Create recorder

    out_file = os.path.join(args.out_dir, 'output_nvenc.mp4' if args.nvenc else 'output.mp4')
    recorder = (pyglrec.NVENCFrameRecorder if args.nvenc else pyglrec.UncompressedFrameCPURecorder)(window_width, window_height, out_file, fps=args.out_fps)

    # --------------------------------------------------------------------------------------------
    # Setup states

    glfw.swap_interval(1)           # Enable V-Sync
    gl.glEnable(gl.GL_DEPTH_TEST)   # Enable depth testing
    gl.glEnable(gl.GL_MULTISAMPLE)  # Enable multisampling on default framebuffer

    # --------------------------------------------------------------------------------------------
    # Resolve dpi scaling factor

    dpi_scale_x, dpi_scale_y = 1.0, 1.0

    win_x, win_y = glfw.get_window_pos(window)
    for monitor in glfw.get_monitors():
        monitor_x0, monitor_y0, monitor_w, monitor_h = glfw.get_monitor_workarea(monitor)
        monitor_x1 = monitor_x0 + monitor_w
        monitor_y1 = monitor_y0 + monitor_h
        if (monitor_x0 <= win_x < monitor_x1) and (monitor_y0 <= win_y < monitor_y1):
            break

    if monitor is not None:
        (dpi_scale_x, dpi_scale_y) = glfw.get_monitor_content_scale(monitor)
        monitor_name = glfw.get_monitor_name(monitor).decode()
        print(f'You are running on monitor "{monitor_name}" with DPI scale factors ({dpi_scale_x:.2f}, {dpi_scale_y:.2f}). The framebuffer size for rendering is ({window_width}, {window_height}), while the window size is ({int(window_width * dpi_scale_x)}, {int(window_height * dpi_scale_y)}).')

    # --------------------------------------------------------------------------------------------
    # Main loop

    prev_time = time.monotonic()

    animation_time = 0.0
    delta_time = 1.0 / args.out_fps

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Limit frame rate
        cur_time = time.monotonic()
        if (cur_time - prev_time) > (1.0 / args.fps):
            # Compute transformation matrices
            model_mat = model_trans_mat * model_rot_mat * model_scale_mat
            mv_mat = view_mat * model_mat
            mvp_mat = proj_mat * mv_mat
            light_pos_camera_space = glm.vec3(view_mat * light_rot_mat * glm.vec4(light_pos, 1.0))

            # First render pass: Render to FBO
            with recorder.record(enabled=(not args.disabled)):
                gl.glViewport(0, 0, window_width, window_height)

                gl.glClearColor(0.0, 0.0, 0.0, 1.0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

                # Draw cube
                cube_obj.draw(mvp_mat, mv_mat, glm.transpose(glm.inverse(glm.mat3(mv_mat))), light_pos_camera_space)

            # Second render pass: Render FBO texture to default framebuffer
            quad_obj.draw(recorder.texture_id, int(window_width * dpi_scale_x), int(window_height * dpi_scale_y))

            # Animate cube
            angle_x = math.sin(animation_time * 0.6) * rot_speed
            angle_y = math.sin(animation_time * 0.9) * rot_speed
            angle_z = math.sin(animation_time * 1.1) * rot_speed

            rot_quaternion_x = glm.angleAxis(glm.radians(angle_x), glm.vec3(1.0, 0.0, 0.0))
            rot_quaternion_y = glm.angleAxis(glm.radians(angle_y), glm.vec3(0.0, 1.0, 0.0))
            rot_quaternion_z = glm.angleAxis(glm.radians(angle_z), glm.vec3(0.0, 0.0, 1.0))
            rot_quaternion = rot_quaternion_x * rot_quaternion_y * rot_quaternion_z
            model_rot_mat = glm.mat4_cast(rot_quaternion)

            # Also rotate light at different speed
            angle_x = math.sin(animation_time * 1.5) * rot_speed
            angle_y = math.sin(animation_time * 1.8) * rot_speed
            angle_z = math.sin(animation_time * 2.1) * rot_speed

            rot_quaternion_x = glm.angleAxis(glm.radians(angle_x), glm.vec3(1.0, 0.0, 0.0))
            rot_quaternion_y = glm.angleAxis(glm.radians(angle_y), glm.vec3(0.0, 1.0, 0.0))
            rot_quaternion_z = glm.angleAxis(glm.radians(angle_z), glm.vec3(0.0, 0.0, 1.0))
            rot_quaternion = rot_quaternion_x * rot_quaternion_y * rot_quaternion_z
            light_rot_mat = glm.mat4_cast(rot_quaternion)

            # Swap buffers
            glfw.swap_buffers(window)

            prev_time = cur_time
            animation_time += delta_time

    recorder.finalize()
    print(f'Recorded video saved to: {out_file}')

    # --------------------------------------------------------------------------------------------
    # Cleanup OpenGL resources and GLFW window

    cube_obj.dispose()
    quad_obj.dispose()

    glfw.destroy_window(window)
    glfw.terminate()

    print(f'Bye!')


if __name__ == "__main__":
    main()
