import platform

import numpy as np
import OpenGL.GL as gl
import pyglm.glm as glm

import pyglrec

# --------------------------------------------------------------------------------------------
# OpenGL version and GLSL version settings

if platform.system() == "Darwin":
    GLSL_VERSION = "#version 410 core"
else:
    GLSL_VERSION = "#version 460 core"

# --------------------------------------------------------------------------------------------
# Cube object class


class CubeObject:
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

    def __init__(self):
        """Initialize the cube object by setting up shaders and geometry. This function must be called within a valid OpenGL context."""

        # --------------------------------------------------------------------------------------------
        # Build shader program

        self.shader_program = pyglrec.shader.build_shader_program(self.VERTEX_SHADER_SRC, self.FRAGMENT_SHADER_SRC)
        self.mvp_mat_loc = gl.glGetUniformLocation(self.shader_program, "u_mvp_mat")

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

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

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

        gl.glBindVertexArray(0)

        self.num_indices = len(indices)

    def dispose(self) -> None:
        """Gracefully delete OpenGL resources."""
        gl.glDeleteProgram(self.shader_program)
        gl.glDeleteVertexArrays(1, [self.vao])

    def draw(self, mvp_mat: glm.mat4) -> None:
        """Draw the cube with the given MVP matrix.

        Parameters
        ----------
        mvp_mat : glm.mat4
            The Model-View-Projection matrix to use for rendering.
        """

        gl.glUseProgram(self.shader_program)

        gl.glUniformMatrix4fv(self.mvp_mat_loc, 1, gl.GL_FALSE, glm.value_ptr(mvp_mat))

        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.num_indices, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        gl.glUseProgram(0)
