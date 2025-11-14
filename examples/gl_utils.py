import math

import pyglm.glm as glm

# --------------------------------------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------------------------------------
