"""Shader program for rendering 3D objects with shading."""

import contextlib
import enum

import OpenGL.GL as gl
import pyglm.glm as glm

import pyglrec

# --------------------------------------------------------------------------------------------------------------------------------
# Shaders


class ObjectShader:
    """A shader program for rendering 3D objects with various rendering modes including shading."""

    VERTEX_SHADER = """
    # version 330 core
    
    layout(location = 0) in vec3 in_position;
    layout(location = 1) in vec3 in_color;
    layout(location = 2) in vec3 in_normal;

    uniform mat4 u_mvp_mat;
    uniform mat4 u_mv_mat;
    uniform mat4 u_normal_mat;
    uniform float u_point_size;

    out vec3 f_color;
    out vec3 f_normal_camera_space;
    out vec3 f_pos_camera_space;

    void main() {
        gl_Position = u_mvp_mat * vec4(in_position, 1.0);
        gl_PointSize = u_point_size;
        
        f_color = in_color;
        f_normal_camera_space = (u_normal_mat * vec4(in_normal, 0.0)).xyz;
        f_pos_camera_space = (u_mv_mat * vec4(in_position, 1.0)).xyz;
    }
    
    """

    FRAGMENT_SHADER = """
    # version 330 core
    
    in vec3 f_color;
    in vec3 f_normal_camera_space;
    in vec3 f_pos_camera_space;
    
    uniform vec3 u_color;
    uniform float u_rendering_type;
    uniform float u_shiness;
    uniform float u_ambient_intensity;
    uniform vec3 u_light_pos_camera_space;
    
    out vec4 out_color;
    
    vec4 shade(vec3 diffuse_color, vec3 specular_color, vec3 ambient_color, vec3 N) {
        vec3 V = normalize(-f_pos_camera_space);                           // View direction
        vec3 L = normalize(u_light_pos_camera_space - f_pos_camera_space); // Light direction
        vec3 H = normalize(L + V);                                         // Halfway vector
        
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        
        vec3 diffuse = diffuse_color * NdotL;
        vec3 specular = specular_color * pow(NdotH, u_shiness);
        vec3 ambient = ambient_color * u_ambient_intensity;
        
        return vec4(diffuse + specular + ambient, 1.0);
    }
    
    void main() {
        // Compute normal
        vec3 N = normalize(f_normal_camera_space);
        
        // Compute color
        if (-0.5 < u_rendering_type && u_rendering_type < 0.5) {
            // uniform color
            out_color = vec4(u_color, 1.0);
        } else if (0.5 < u_rendering_type && u_rendering_type < 1.5) {
            // color
            out_color = vec4(f_color, 1.0);
        } else if (1.5 < u_rendering_type && u_rendering_type < 2.5) {
            // SHADING with uniform color
            out_color = shade(u_color, vec3(1.0), u_color, N);
        } else if (2.5 < u_rendering_type && u_rendering_type < 3.5) {
            // SHADING with color
            out_color = shade(f_color, vec3(1.0), f_color, N);
        } else {
            out_color = vec4(1.0, 0.0, 1.0, 1.0); // Magenta for error
        }
    }
    """

    class RenderingType(enum.IntEnum):
        # autopep8: off
        UNIFORM_COLOR         = 0
        COLOR                 = 1
        SHADING_UNIFORM_COLOR = 2
        SHADING_COLOR         = 3
        # autopep8: on


    # autopep8: off
    U_MVP_MAT                                       = 'mvp_mat'
    U_MV_MAT                                        = 'mv_mat'
    U_NORMAL_MAT                                    = 'normal_mat'
    U_RENDERING_TYPE                                = 'rendering_type'
    U_SHINESS                                       = 'shiness'
    U_AMBIENT_INTENSITY                             = 'ambient_intensity'
    U_LIGHT_POS_CAMERA_SPACE                        = 'light_pos_camera_space'
    U_POINT_SIZE                                    = 'point_size'
    U_COLOR                                         = 'color'
    # autopep8: on

    UNIFORM_VAR_NAMES = [
        U_MVP_MAT,
        U_MV_MAT,
        U_NORMAL_MAT,
        U_RENDERING_TYPE,
        U_SHINESS,
        U_AMBIENT_INTENSITY,
        U_LIGHT_POS_CAMERA_SPACE,
        U_POINT_SIZE,
        U_COLOR,
    ]

    def __init__(self) -> None:
        """Initialize the shader program. This function must be called within a valid OpenGL context."""

        self._program = pyglrec.build_shader_program(self.VERTEX_SHADER, self.FRAGMENT_SHADER)

        # Get uniform locations
        self._uniform_locs = {name: gl.glGetUniformLocation(self._program, f'u_{name}') for name in self.UNIFORM_VAR_NAMES}

    @contextlib.contextmanager
    def ctx(
        self,
        rendering_type: RenderingType,
        mvp_mat: glm.mat4,
        mv_mat: glm.mat4,
        normal_mat: glm.mat4 = glm.mat4(1.0),
        light_pos_camera_space: glm.vec3 = glm.vec3(0.0, 10.0, 0.0),
        color: glm.vec3 = glm.vec3(1.0),
        shiness: float = 32.0,
        ambient_intensity: float = 0.3,
        point_size: float = 1.0,
    ):
        """Context manager for using the shader program.

        Parameters
        ----------
        rendering_type : RenderingType
            Rendering type.
        mvp_mat : glm.mat4
            Model-View-Projection matrix.
        mv_mat : glm.mat4
            Model-View matrix.
        normal_mat : glm.mat4
            Normal matrix.
        light_pos_camera_space : glm.vec3
            Light position in camera space.
        color : glm.vec3, optional
            Object color, by default glm.vec3(1.0).
        shiness : float, optional
            Shininess factor for specular highlights, by default 32.0.
        ambient_intensity : float, optional
            Ambient light intensity for shading, by default 0.3.
        point_size : float, optional
            Point size for point rendering, by default 1.0.
        """

        try:
            gl.glUseProgram(self._program)

            gl.glUniformMatrix4fv(self._uniform_locs[self.U_MVP_MAT], 1, gl.GL_FALSE, glm.value_ptr(mvp_mat))
            gl.glUniformMatrix4fv(self._uniform_locs[self.U_MV_MAT], 1, gl.GL_FALSE, glm.value_ptr(mv_mat))
            gl.glUniformMatrix4fv(self._uniform_locs[self.U_NORMAL_MAT], 1, gl.GL_FALSE, glm.value_ptr(normal_mat))
            gl.glUniform1f(self._uniform_locs[self.U_RENDERING_TYPE], float(rendering_type.value))
            gl.glUniform1f(self._uniform_locs[self.U_SHINESS], shiness)
            gl.glUniform1f(self._uniform_locs[self.U_AMBIENT_INTENSITY], ambient_intensity)
            gl.glUniform3fv(self._uniform_locs[self.U_LIGHT_POS_CAMERA_SPACE], 1, glm.value_ptr(light_pos_camera_space))
            gl.glUniform1f(self._uniform_locs[self.U_POINT_SIZE], point_size)
            gl.glUniform3fv(self._uniform_locs[self.U_COLOR], 1, glm.value_ptr(color))

            yield

            # Drawing done
        finally:
            gl.glUseProgram(0)

    def dispose(self) -> None:
        """Gracefully delete the shader program."""

        gl.glDeleteProgram(self._program)
        self._program = 0

# --------------------------------------------------------------------------------------------------------------------------------
