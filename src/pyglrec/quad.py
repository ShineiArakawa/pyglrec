"""
quad
====

Quad rendering utilities for OpenGL.
"""

import numpy as np
import OpenGL.GL as gl

import pyglrec.shader as shader


class Quad:
    """A simple screen-aligned quad for rendering textures."""

    QUAD_VERTEX_SHADER_SRC = """
    # version 330 core

    layout(location = 0) in vec2 in_position;
    layout(location = 1) in vec2 in_texcoord;

    out vec2 frag_texcoord;

    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        frag_texcoord = in_texcoord;
    }
    """

    QUAD_FRAGMENT_SHADER_SRC = """
    # version 330 core

    in vec2 frag_texcoord;

    out vec4 out_color;

    uniform sampler2D u_texture;

    void main() {
        out_color = texture(u_texture, frag_texcoord);
    }
    """

    def __init__(self):
        # Create shader program for rendering quad
        self._quad_shader = shader.build_shader_program(self.QUAD_VERTEX_SHADER_SRC, self.QUAD_FRAGMENT_SHADER_SRC)

        # Create quad geometry
        quad_vertex_data = np.array([
            # positions   # texcoords
            -1.0, -1.0,   0.0, 0.0,
            1.0, -1.0,   1.0, 0.0,
            1.0,  1.0,   1.0, 1.0,
            -1.0,  1.0,   0.0, 1.0,
        ], dtype=np.float32)

        quad_indices = np.array([
            0, 1, 2,
            2, 3, 0,
        ], dtype=np.uint32)

        self._quad_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._quad_vao)

        quad_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, quad_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad_vertex_data.nbytes, quad_vertex_data, gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * quad_vertex_data.itemsize, gl.ctypes.c_void_p(0))

        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * quad_vertex_data.itemsize, gl.ctypes.c_void_p(2 * quad_vertex_data.itemsize))

        quad_ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, quad_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        self._tex_loc = gl.glGetUniformLocation(self._quad_shader, "u_texture")

    def draw(self, tex_id: int, width: int, height: int) -> None:
        """Draw the quad with the given texture.

        Parameters
        ----------
        tex_id : int
            The OpenGL texture ID to bind to the quad.
        width : int
            The width of the viewport.
        height : int
            The height of the viewport.
        """

        gl.glViewport(0, 0, width, height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glUseProgram(self._quad_shader)

        # Bind FBO texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glUniform1i(self._tex_loc, 0)

        # Draw quad
        gl.glBindVertexArray(self._quad_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        # Unbind texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        gl.glUseProgram(0)
