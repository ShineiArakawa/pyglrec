import enum
import math
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


window_width, window_height = 0, 0
proj_mat, view_mat = None, None
model_trans_mat, model_rot_mat, model_scale_mat = None, None, None
arc_scale = 1.0
arc_ball_mode = ArcBallModel.NONE
is_dragging = False
last_mouse_pos = glm.vec2(0.0, 0.0)
camera_pos = None
camera_lookat = None
camera_up = None
last_mouse_clicked_time = 0.0


@click.command()
@click.option("--offscreen", is_flag=True, help="Run in offscreen mode")
@click.option("--width", default=640, help="Window width")
@click.option("--height", default=480, help="Window height")
def main(**args):
    args = pyglrec.EasyDict(args)

    global window_width, window_height
    window_width = args.width
    window_height = args.height

    # --------------------------------------------------------------------------------------------
    # Create GLFW window and OpenGL context

    if not glfw.init():
        raise Exception("GLFW can't be initialized")

    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.VISIBLE, not args.offscreen)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, OPENGL_VERSION_MAJOR)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, OPENGL_VERSION_MINOR)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

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

    def reset_camera_pose():
        global camera_pos, camera_lookat, camera_up
        camera_pos = glm.vec3(5.0, 0.0, 0.0)
        camera_lookat = glm.vec3(0.0, 0.0, 0.0)
        camera_up = glm.vec3(0.0, 1.0, 0.0)
        global view_mat
        view_mat = glm.lookAt(camera_pos, camera_lookat, camera_up)

        global model_rot_mat, model_trans_mat, model_scale_mat
        model_rot_mat = glm.mat4(1.0)
        model_trans_mat = glm.mat4(1.0)
        model_scale_mat = glm.mat4(1.0)

        global proj_mat
        proj_mat = perspective(window_width / window_height)

    reset_camera_pose()

    # --------------------------------------------------------------------------------------------
    # Set callback functions

    def window_resize_callback(window: glfw._GLFWwindow, width: int, height: int):
        if width <= 0 or height <= 0:
            return  # minimized window

        global window_width, window_height
        window_width = width
        window_height = height

        # Update the projection matrix
        global proj_mat
        proj_mat = perspective(window_width / window_height)

    glfw.set_window_size_callback(window, window_resize_callback)

    def scroll_callback(window: glfw._GLFWwindow, x_offset: float, y_offset: float):
        global window_width, window_height, arc_scale
        global model_scale_mat

        speed = 0.035 if platform.system() == 'Darwin' else 0.15
        scale_max = 0.5 * window_height  # canvas ~ half of the window height
        scale_min = 50 / min(window_width, window_height)  # canvas ~50x50 pixels

        arc_scale = min(scale_max, max(scale_min, arc_scale * (1.0 + y_offset * speed)))
        model_scale_mat = glm.scale(glm.vec3(arc_scale, arc_scale, arc_scale))

    glfw.set_scroll_callback(window, scroll_callback)

    def mouse_button_callback(window: glfw._GLFWwindow, button: int, action: int, mods: int):
        global arc_ball_mode, is_dragging, last_mouse_pos, last_mouse_clicked_time

        if action == glfw.PRESS:
            # Detect double click for resetting camera pose
            cur_time = time.monotonic()
            is_double_clicked = (cur_time - last_mouse_clicked_time) < 0.3  # in seconds
            last_mouse_clicked_time = cur_time

            if is_double_clicked:
                reset_camera_pose()
                return

            # Arcball mode selection
            if button == glfw.MOUSE_BUTTON_LEFT:
                arc_ball_mode = ArcBallModel.ROTATE
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                arc_ball_mode = ArcBallModel.SCALE
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                arc_ball_mode = ArcBallModel.TRANSLATE

            if arc_ball_mode != ArcBallModel.NONE and not is_dragging:
                is_dragging = True
                last_mouse_pos = glm.vec2(*glfw.get_cursor_pos(window))
        else:
            is_dragging = False
            arc_ball_mode = ArcBallModel.NONE

    glfw.set_mouse_button_callback(window, mouse_button_callback)

    def cursor_pos_callback(window: glfw._GLFWwindow, x_pos: float, y_pos: float):
        global arc_ball_mode, window_width, window_height, arc_scale, is_dragging, last_mouse_pos
        global view_mat, proj_mat, model_rot_mat, model_trans_mat, model_scale_mat

        if not is_dragging:
            return

        cur_mouse_pos = glm.vec2(x_pos, y_pos)

        delta = cur_mouse_pos - last_mouse_pos
        length = delta.x * delta.x + delta.y * delta.y
        pix_width = min(1.0 / window_width, 1.0 / window_height)
        if length < pix_width * pix_width:
            return  # ignore small movements

        if arc_ball_mode == ArcBallModel.ROTATE:
            cur_pos_norm = 2.0 * cur_mouse_pos / glm.vec2(window_width, window_height) - 1.0
            last_pos_norm = 2.0 * last_mouse_pos / glm.vec2(window_width, window_height) - 1.0

            c2w_mat = glm.inverse(view_mat)

            dx = cur_pos_norm.x - last_pos_norm.x
            dy = cur_pos_norm.y - last_pos_norm.y

            horizontal_axis_in_world = glm.vec3(c2w_mat[1])
            vertical_axis_in_world = glm.vec3(c2w_mat[0])

            horizontal_angle = 2.0 * math.pi * dx
            vertical_angle = 2.0 * math.pi * dy

            horizontal_quad = glm.angleAxis(horizontal_angle, glm.normalize(horizontal_axis_in_world))
            vertical_quad = glm.angleAxis(vertical_angle, glm.normalize(vertical_axis_in_world))

            rot_mat = glm.mat4_cast(horizontal_quad * vertical_quad)

            # Update model rotation matrix
            model_rot_mat = rot_mat * model_rot_mat
        elif arc_ball_mode == ArcBallModel.TRANSLATE:
            cur_pos_norm = cur_mouse_pos / glm.vec2(window_width, window_height)
            last_pos_norm = last_mouse_pos / glm.vec2(window_width, window_height)

            origin_on_screen = proj_mat * view_mat * glm.vec4(0.0, 0.0, 0.0, 1.0)
            origin_on_screen = origin_on_screen / origin_on_screen.w

            new_pos_on_screen = glm.vec4(
                2.0 * cur_pos_norm.x - 1.0,
                1.0 - 2.0 * cur_pos_norm.y,
                origin_on_screen.z,
                1.0
            )
            old_pos_on_screen = glm.vec4(
                2.0 * last_pos_norm.x - 1.0,
                1.0 - 2.0 * last_pos_norm.y,
                origin_on_screen.z,
                1.0
            )

            inv_mvp_mat = glm.inverse(proj_mat * view_mat)

            new_pos_in_obj_space = inv_mvp_mat * new_pos_on_screen
            new_pos_in_obj_space /= new_pos_in_obj_space.w
            old_pos_in_obj_space = inv_mvp_mat * old_pos_on_screen
            old_pos_in_obj_space /= old_pos_in_obj_space.w

            trans_world = glm.vec3(new_pos_in_obj_space - old_pos_in_obj_space)

            # Update translation matrix
            model_trans_mat = glm.translate(trans_world) * model_trans_mat
        elif arc_ball_mode == ArcBallModel.SCALE:
            scale_max = 0.5 * window_height  # canvas ~ half of the window height
            scale_min = 50 / min(window_width, window_height)  # canvas ~50x50 pixels
            arc_scale = min(scale_max, max(scale_min, arc_scale * (1.0 + 10.0 * delta.y)))
            model_scale_mat = glm.scale(glm.vec3(arc_scale, arc_scale, arc_scale))

        last_mouse_pos = cur_mouse_pos

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)

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
    # Setup states

    glfw.swap_interval(1)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # --------------------------------------------------------------------------------------------
    # Main loop

    while not glfw.window_should_close(window):
        glfw.poll_events()

        gl.glViewport(0, 0, window_width, window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glUseProgram(shader_program)

        # Compute MVP matrix
        model_mat = model_trans_mat * model_rot_mat * model_scale_mat
        mvp_mat = proj_mat * view_mat * model_mat

        # Set uniform variables
        mvp_mat_loc = gl.glGetUniformLocation(shader_program, "u_mvp_mat")
        gl.glUniformMatrix4fv(mvp_mat_loc, 1, gl.GL_FALSE, glm.value_ptr(mvp_mat))

        # Draw cube
        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES, num_indices, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        gl.glUseProgram(0)

        glfw.swap_buffers(window)

    glfw.destroy_window(window)
    glfw.terminate()
    print(f'Bye!')


if __name__ == "__main__":
    main()
