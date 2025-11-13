"""Shader compiler utility functions."""

import textwrap

import OpenGL.GL as gl

# --------------------------------------------------------------------------------------------------------------------------------
# Shader compiler


def build_shader_program(vert_src: str, frag_src: str) -> int:
    """Build a shader program from vertex and fragment shader source code.

    Parameters
    ----------
    vert_src : str
        Vertex shader source code.
    frag_src : str
        Fragment shader source code.

    Returns
    -------
    int
        The ID of the created shader program.
    """

    is_OK = True
    error_msg = ''

    program = gl.glCreateProgram()
    vert_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    frag_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(vert_shader, textwrap.dedent(vert_src))
    gl.glShaderSource(frag_shader, textwrap.dedent(frag_src))
    gl.glCompileShader(vert_shader)
    log = gl.glGetShaderInfoLog(vert_shader)
    if log:
        error_msg += f"Vertex shader compilation failed:\n{log}\n"
        is_OK = False
    gl.glCompileShader(frag_shader)
    log = gl.glGetShaderInfoLog(frag_shader)
    if log:
        error_msg += f"Fragment shader compilation failed:\n{log}\n"
        is_OK = False

    gl.glAttachShader(program, vert_shader)
    gl.glAttachShader(program, frag_shader)

    gl.glLinkProgram(program)
    gl.glDeleteShader(vert_shader)
    gl.glDeleteShader(frag_shader)

    if not is_OK:
        raise RuntimeError(f"Shader program linking failed:\n{error_msg}")

    return program

# --------------------------------------------------------------------------------------------------------------------------------
