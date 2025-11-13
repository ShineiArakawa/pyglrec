"""
frame_buffer
------------

Easy management of OpenGL framebuffer objects.
"""

import contextlib

import OpenGL.GL as gl

# --------------------------------------------------------------------------------------------------------------------------------
# Framebuffer object


class FrameBuffer:
    MSAA_FACTOR = 4

    def __init__(self, width: int, height: int):
        """Create a framebuffer object with a texture and a renderbuffer.

        Parameters
        ----------
        width : int
            The width of the framebuffer.
        height : int
            The height of the framebuffer.

        Raises
        ------
        RuntimeError
            If the framebuffer is not complete.
        """

        self._prev_fbo = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Create MSAA FBO for drawing

        # Generate framebuffer object
        self._fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)

        # Generate framebuffer texture for color attachment
        self._texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, self._texture)
        gl.glTexImage2DMultisample(gl.GL_TEXTURE_2D_MULTISAMPLE, self.MSAA_FACTOR, gl.GL_RGBA8, width, height, True)

        # Attach texture to the FBO as color attachment
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D_MULTISAMPLE, self._texture, 0)

        # Generate renderbuffer for depth and stencil attachment
        self._rbo = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._rbo)
        gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, self.MSAA_FACTOR, gl.GL_DEPTH24_STENCIL8, width, height)

        # Attach renderbuffer to the FBO as depth and stencil attachment
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self._rbo)

        # Check if framebuffer is complete
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Framebuffer is not complete')

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._prev_fbo)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Create a single color FBO for texture readback

        self._resolve_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._resolve_fbo)

        self._resolve_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._resolve_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._resolve_texture, 0)

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Resolve Framebuffer is not complete')

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._prev_fbo)

        # ----------------------------------------------------------------------------------------------------------------------------

        self._width = width
        self._height = height

        # ----------------------------------------------------------------------------------------------------------------------------

    @property
    def texture_id(self) -> int:
        """The OpenGL texture ID of the framebuffer's color attachment.
        """
        return self._resolve_texture

    @property
    def width(self) -> int:
        """The current width of the framebuffer."""
        return self._width

    @property
    def height(self) -> int:
        """The current height of the framebuffer."""
        return self._height

    def bind(self):
        """Bind the framebuffer.
        """

        self._prev_fbo = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)

    def unbind(self, resolve: bool = True):
        """Unbind the framebuffer and switch back to the previous framebuffer.

        Parameters
        ----------
        resolve : bool, optional
            Whether to resolve the MSAA framebuffer to the single-sample framebuffer. Default is True.
        """

        if resolve:
            # Resolve MSAA framebuffer to single-sample framebuffer
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._fbo)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._resolve_fbo)

            gl.glBlitFramebuffer(
                0, 0, self._width, self._height,
                0, 0, self._width, self._height,
                gl.GL_COLOR_BUFFER_BIT,
                gl.GL_NEAREST
            )

        # Switch back to the previous framebuffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._prev_fbo)

    @contextlib.contextmanager
    def ctx(self, resolve: bool = True):
        """Context manager for binding and unbinding the framebuffer.

        Parameters
        ----------
        resolve : bool, optional
            Whether to resolve the MSAA framebuffer to the single-sample framebuffer upon exiting the context. Default is True.

        Example
        -------
        ```python
        fbo = FrameBuffer(800, 600)

        with fbo.ctx():
            # Render to the framebuffer
            ...
            gl.glDrawElements(...)
            ...

        # Automatically unbind the framebuffer here and switch back to the previous framebuffer
        """

        self.bind()

        try:
            # Within this block, the framebuffer is bound
            yield
        finally:
            self.unbind(resolve=resolve)

    @contextlib.contextmanager
    def ctx_resolved_buffer(self):
        """Context manager for binding the framebuffer and yielding the resolved texture ID.

        Example
        -------
        ```python
        fbo = FrameBuffer(800, 600)

        with fbo.ctx_resolved_texture() as tex_id:
            # Render to the framebuffer
            ...
            gl.glDrawElements(...)
            ...
            # Use tex_id as the texture ID of the resolved framebuffer

        # Automatically unbind the framebuffer here and switch back to the previous framebuffer
        ```
        """

        self._prev_fbo = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._resolve_fbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._resolve_texture)

        try:
            yield self._resolve_texture
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._prev_fbo)

    def rescale(self, width: int, height: int):
        """Rescale the framebuffer to a new width and height.

        Parameters
        ----------
        width : int
            New width.
        height : int
            New height.

        Raises
        ------
        RuntimeError
            If the framebuffer is not complete.
        """

        self._width = width
        self._height = height

        cur_fbo = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Update MSAA FBO

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)

        # Update color attachment texture
        gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, self._texture)
        gl.glTexImage2DMultisample(gl.GL_TEXTURE_2D_MULTISAMPLE, self.MSAA_FACTOR, gl.GL_RGBA8, width, height, True)

        # Update renderbuffer
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._rbo)
        gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, self.MSAA_FACTOR, gl.GL_DEPTH24_STENCIL8, width, height)

        # Check if framebuffer is complete
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Framebuffer is not complete after rescaling')

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, cur_fbo)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Update single color FBO for texture readback

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._resolve_fbo)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._resolve_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Resolve Framebuffer is not complete after rescaling')

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, cur_fbo)

    def dispose(self):
        """Gracefully delete the framebuffer and its attachments.
        """

        if hasattr(self, '_texture'):
            gl.glDeleteTextures([self._texture])
            del self._texture
        if hasattr(self, '_rbo'):
            gl.glDeleteRenderbuffers([self._rbo])
            del self._rbo
        if hasattr(self, '_fbo'):
            gl.glDeleteFramebuffers([self._fbo])
            del self._fbo

        if hasattr(self, '_resolve_texture'):
            gl.glDeleteTextures([self._resolve_texture])
            del self._resolve_texture
        if hasattr(self, '_resolve_fbo'):
            gl.glDeleteFramebuffers([self._resolve_fbo])
            del self._resolve_fbo


# --------------------------------------------------------------------------------------------------------------------------------
