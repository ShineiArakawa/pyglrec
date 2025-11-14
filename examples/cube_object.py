import platform

import numpy as np
import OpenGL.GL as gl
import pyglm.glm as glm
import shader

# --------------------------------------------------------------------------------------------
# OpenGL version and GLSL version settings

if platform.system() == "Darwin":
    GLSL_VERSION = "#version 410 core"
else:
    GLSL_VERSION = "#version 460 core"

# --------------------------------------------------------------------------------------------
# Cube object class


class CubeObject:
    def __init__(self):
        """Initialize the cube object by setting up shaders and geometry. This function must be called within a valid OpenGL context."""

        # --------------------------------------------------------------------------------------------
        # Build shader program

        self.shader = shader.ObjectShader()

        # --------------------------------------------------------------------------------------------
        # Create cube geometry

        positions = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32) * 0.5
        colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ], dtype=np.float32)
        normals = np.array([
            [1.0, 0.0, 0.0],   # 0
            [0.0, 1.0, 0.0],   # 1
            [0.0, 0.0, 1.0],   # 2
            [0.0, 0.0, -1.0],  # 3
            [0.0, -1.0, 0.0],  # 4
            [-1.0, 0.0, 0.0],  # 5
        ])
        faces = np.array([
            [7, 4, 1],
            [7, 1, 6],
            [2, 4, 7],
            [2, 7, 5],
            [5, 7, 6],
            [5, 6, 3],
            [4, 2, 0],
            [4, 0, 1],
            [3, 6, 1],
            [3, 1, 0],
            [2, 5, 3],
            [2, 3, 0]
        ], dtype=np.uint32)

        # Create interleaved vertex data
        vertex_data = np.zeros((faces.shape[0] * 3, 3 + 3 + 3), dtype=np.float32)
        indices = np.zeros((faces.shape[0] * 3,), dtype=np.uint32)

        for i in range(6):
            for j in range(3):
                pos = positions[faces[i * 2 + 0][j]]
                vertex_data[i * 6 + j, 0:3] = pos
                vertex_data[i * 6 + j, 3:6] = colors[i]
                vertex_data[i * 6 + j, 6:9] = normals[i]
                indices[i * 6 + j] = i * 6 + j
            for j in range(3):
                pos = positions[faces[i * 2 + 1][j]]
                vertex_data[i * 6 + 3 + j, 0:3] = pos
                vertex_data[i * 6 + 3 + j, 3:6] = colors[i]
                vertex_data[i * 6 + 3 + j, 6:9] = normals[i]
                indices[i * 6 + 3 + j] = i * 6 + 3 + j

        index_data = np.array(indices, dtype=np.uint32)

        # Allocate VAO and EBO

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)

        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, (3 + 3 + 3) * vertex_data.itemsize, gl.ctypes.c_void_p(0))

        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, (3 + 3 + 3) * vertex_data.itemsize, gl.ctypes.c_void_p(3 * vertex_data.itemsize))

        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, (3 + 3 + 3) * vertex_data.itemsize, gl.ctypes.c_void_p((3 + 3) * vertex_data.itemsize))

        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)

        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, gl.GL_STATIC_DRAW)

        gl.glBindVertexArray(0)

        self.num_indices = len(index_data)

    def dispose(self) -> None:
        """Gracefully delete OpenGL resources."""

        self.shader.dispose()
        gl.glDeleteVertexArrays(1, [self.vao])

    def draw(
        self,
        mvp_mat: glm.mat4,
        mv_mat: glm.mat4,
        normal_mat: glm.mat3,
        light_pos_camera_space: glm.vec3,
    ) -> None:
        """Draw the cube with the given MVP matrix.

        Parameters
        ----------
        mvp_mat : glm.mat4
            The Model-View-Projection matrix to use for rendering.
        mv_mat : glm.mat4
            The Model-View matrix to use for rendering.
        normal_mat : glm.mat3
            The Normal matrix to use for rendering.
        light_pos_camera_space : glm.vec3
            The light position in camera space.
        """

        with self.shader.ctx(
            rendering_type=shader.ObjectShader.RenderingType.SHADING_COLOR,
            mvp_mat=mvp_mat,
            mv_mat=mv_mat,
            normal_mat=normal_mat,
            light_pos_camera_space=light_pos_camera_space,
        ):
            gl.glBindVertexArray(self.vao)
            gl.glDrawElements(gl.GL_TRIANGLES, self.num_indices, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
