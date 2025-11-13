import enum
import math
import os
import platform
import time

import click
import glfw
import numpy as np
import OpenGL.GL as gl
import pyglm.glm as glm

import pyglrec

# --------------------------------------------------------------------------------------------
# OpenGL version and GLSL version settings

if platform.system() == "Darwin":
    # On macOS, OpenGL 4.1 is the highest supported version
    OPENGL_VERSION_MAJOR = 4
    OPENGL_VERSION_MINOR = 1
    GLSL_VERSION = "#version 410 core"
else:
    OPENGL_VERSION_MAJOR = 4
    OPENGL_VERSION_MINOR = 6
    GLSL_VERSION = "#version 460 core"

# --------------------------------------------------------------------------------------------
# Shader sources

VERTEX_SHADER_SRC = GLSL_VERSION + """\n\n
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

out vec3 frag_color;

uniform mat4 u_mvp_mat;

void main() {
    gl_Position = u_mvp_mat * vec4(in_position, 1.0);
    frag_color = in_color;
}
"""

FRAGMENT_SHADER_SRC = GLSL_VERSION + """\n\n
in vec3 frag_color;

out vec4 out_color;

void main() {
    out_color = vec4(frag_color, 1.0);
}
"""

QUAD_VERTEX_SHADER_SRC = GLSL_VERSION + """\n\n
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texcoord;

out vec2 frag_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    frag_texcoord = in_texcoord;
}
"""

QUAD_FRAGMENT_SHADER_SRC = GLSL_VERSION + """\n\n
in vec2 frag_texcoord;

out vec4 out_color;

uniform sampler2D u_texture;

void main() {
    out_color = texture(u_texture, frag_texcoord);
}
"""


# --------------------------------------------------------------------------------------------
# Utility functions


def perspective(aspect: float, fov_deg: float = 45.0,  near: float = 0.1, far: float = 100.0) -> glm.mat4:
    """Create a perspective projection matrix. This method creates the matrix based on the viewport size.

    Parameters
    ----------
    aspect : float
        Aspect ratio (width / height).
    fov_deg : float
        Field of view in degrees. Default is 45.0 degrees.
    near : float
        Near clipping plane. Default is 0.1.
    far : float
        Far clipping plane. Default is 100.0.

    Returns
    -------
    glm.mat4
        The perspective projection matrix.
    """

    angle = glm.radians(fov_deg)
    width, height = None, None
    if aspect >= 1.0:
        height = 2.0 * near * math.tan(angle / 2.0)
        width = height * aspect
    else:
        width = 2.0 * near * math.tan(angle / 2.0)
        height = width / aspect

    depth = far - near

    return glm.mat4(
        # autopep8: off
        2.0 * near / width,                       0.0,                                   0.0,   0.0,
                       0.0,       2.0 * near / height,                                   0.0,   0.0,
                       0.0,                       0.0,                 -(far + near) / depth,  -1.0,
                       0.0,                       0.0,             -2.0 * far * near / depth,   0.0,
        # autopep8: on
    )


class ArcBallModel(enum.Enum):
    """Arcball interaction modes."""

    ROTATE = enum.auto()
    TRANSLATE = enum.auto()
    SCALE = enum.auto()
    NONE = enum.auto()

# --------------------------------------------------------------------------------------------
# Main function


rot_speed = 60.0  # degrees per second


@click.command()
@click.option("--offscreen", type=int, default=None, help="The number of offscreen rendered frames. If not specified, onscreen rendering is performed.")
@click.option("--width", default=2560, help="Window width")
@click.option("--height", default=1440, help="Window height")
@click.option("--out_dir", default="./outputs/render_cube", help="Output directory for offscreen rendering")
@click.option("--fps_limit", default=30.0, help="Frame rate limit for offscreen rendering")
@click.option("--nvenc", is_flag=True, help="Enable NVENC frame recording (requires NVIDIA GPU)")
def main(**args):
    """Render a rotating colored cube using OpenGL and record the frames to a video file.
    """

    args = pyglrec.EasyDict(args)

    global window_width, window_height
    window_width = args.width
    window_height = args.height

    # --------------------------------------------------------------------------------------------
    # Create GLFW window and OpenGL context

    if not glfw.init():
        raise Exception("GLFW can't be initialized")

    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.VISIBLE, args.offscreen is None)
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

    # Print OpenGL and GLFW versions
    print(f'OpenGL version: {gl.glGetString(gl.GL_VERSION).decode()}')
    print(f'GLFW version: {glfw.get_version_string().decode()}')

    # --------------------------------------------------------------------------------------------
    # Camera setup

    camera_pos = glm.vec3(5.0, 0.0, 0.0)
    camera_lookat = glm.vec3(0.0, 0.0, 0.0)
    camera_up = glm.vec3(0.0, 1.0, 0.0)
    view_mat = glm.lookAt(camera_pos, camera_lookat, camera_up)

    model_rot_mat = glm.mat4(1.0)
    model_trans_mat = glm.mat4(1.0)
    model_scale_mat = glm.mat4(1.0)

    proj_mat = perspective(window_width / window_height)

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
    # Build shader program

    shader_program = pyglrec.shader.build_shader_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)

    # --------------------------------------------------------------------------------------------
    # Create cube geometry

    vertex_coords = np.array([
        # Front face
        -0.5, -0.5,  0.5,
        0.5, -0.5,  0.5,
        0.5,  0.5,  0.5,
        -0.5,  0.5,  0.5,
        # Back face
        -0.5, -0.5, -0.5,
        0.5, -0.5, -0.5,
        0.5,  0.5, -0.5,
        -0.5,  0.5, -0.5,
    ], dtype=np.float32)

    vertex_colors = np.array([
        # Front face (red)
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        # Back face (green)
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
    ], dtype=np.float32)

    # Create interleaved vertex data
    vertex_data = np.hstack((vertex_coords.reshape(-1, 3), vertex_colors.reshape(-1, 3))).flatten()

    indices = np.array([
        0, 1, 2, 2, 3, 0,       # Front face
        4, 5, 6, 6, 7, 4,       # Back face
        0, 4, 7, 7, 3, 0,       # Left face
        1, 5, 6, 6, 2, 1,       # Right face
        3, 2, 6, 6, 7, 3,       # Top face
        0, 1, 5, 5, 4, 0        # Bottom face
    ], dtype=np.uint32)

    # Allocate VAO and EBO

    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)

    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW)

    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, (3 + 3) * vertex_data.itemsize, gl.ctypes.c_void_p(0))

    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, (3 + 3) * vertex_data.itemsize, gl.ctypes.c_void_p(3 * vertex_data.itemsize))

    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)

    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindVertexArray(0)

    num_indices = len(indices)

    # --------------------------------------------------------------------------------------------
    # Create quad stuffs (only for onscreen rendering)

    quad_obj = None

    if args.offscreen is None:
        quad_obj = pyglrec.Quad()

    # --------------------------------------------------------------------------------------------
    # Create frame buffer object
    # 1. Onscreen mode
    #     [OpenGL] => [Default FBO] => [FBO] => [NVENC]
    # 2. Offscreen mode
    #     [OpenGL] => [FBO] => [NVENC]

    # Generate framebuffer object
    recorder = (pyglrec.NVENCFrameRecorder if args.nvenc else pyglrec.UncompressedFrameCPURecorder)(window_width, window_height)

    # --------------------------------------------------------------------------------------------
    # Setup states

    glfw.swap_interval(1)
    gl.glEnable(gl.GL_DEPTH_TEST)
    if args.offscreen is None:
        # Enable multisampling on default framebuffer
        gl.glEnable(gl.GL_MULTISAMPLE)

    # --------------------------------------------------------------------------------------------
    # Main loop

    mvp_mat_loc = gl.glGetUniformLocation(shader_program, "u_mvp_mat")

    prev_time = time.monotonic()

    cur_frame = 0
    max_frames = args.offscreen if args.offscreen is not None else float('inf')

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Limit frame rate
        cur_time = time.monotonic()
        if (cur_time - prev_time) > (1.0 / args.fps_limit):
            # First render pass: Render to FBO
            with recorder.record():
                gl.glViewport(0, 0, window_width, window_height)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

                gl.glUseProgram(shader_program)

                # Compute MVP matrix
                model_mat = model_trans_mat * model_rot_mat * model_scale_mat
                mvp_mat = proj_mat * view_mat * model_mat

                # Set uniform variables
                gl.glUniformMatrix4fv(mvp_mat_loc, 1, gl.GL_FALSE, glm.value_ptr(mvp_mat))

                # Draw cube
                gl.glBindVertexArray(vao)
                gl.glDrawElements(gl.GL_TRIANGLES, num_indices, gl.GL_UNSIGNED_INT, None)
                gl.glBindVertexArray(0)

                gl.glUseProgram(0)

            # Second render pass: Render FBO texture to default framebuffer
            if args.offscreen is None:
                quad_obj.draw(recorder.texture_id, window_width, window_height)

            # Animate
            delta_angle = rot_speed * (cur_time - prev_time)
            axis_in_world = glm.normalize(camera_up)
            quad = glm.angleAxis(glm.radians(delta_angle), axis_in_world)
            delta_rot_mat = glm.mat4_cast(quad)
            model_rot_mat = delta_rot_mat * model_rot_mat

            # Swap buffers
            if args.offscreen is None:
                glfw.swap_buffers(window)
            else:
                gl.glFinish()  # Synchronize

            prev_time = cur_time
            cur_frame += 1

            if cur_frame >= max_frames:
                break

    glfw.destroy_window(window)
    glfw.terminate()

    out_file = os.path.join(args.out_dir, 'output_nvenc.mp4' if args.nvenc else 'output.mp4')
    recorder.finalize(out_file)

    print(f'Bye!')


if __name__ == "__main__":
    main()
