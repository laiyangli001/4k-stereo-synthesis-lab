from __future__ import annotations

import ctypes

from OpenGL.GL import (
    glBindBuffer,
    glBindTexture,
    glBufferData,
    glDeleteBuffers,
    glGenBuffers,
    glGenerateMipmap,
    glTexSubImage2D,
    GL_DYNAMIC_DRAW,
    GL_PIXEL_UNPACK_BUFFER,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_UNSIGNED_BYTE,
)


class CudaGlTextureUploader:
    """Upload CUDA uint8 HWC RGBA tensors into existing GL textures."""

    def __init__(self, cuda_gl, *, backend: str | None, debug=False, log_prefix="GLTextureUploader"):
        self.cuda_gl = cuda_gl
        self.backend = backend
        self.debug = bool(debug)
        self.log_prefix = str(log_prefix)
        self.image_enabled = True
        self.image_failed_reason = None
        self._image_resources = []
        self._image_key = None
        self._pbos = []
        self._pbo_resources = []
        self._pbo_key = None

    def release(self):
        self.release_images()
        self.release_pbos()

    def release_images(self):
        if self.cuda_gl:
            for resource in self._image_resources:
                try:
                    self.cuda_gl.unregister_resource(resource)
                except Exception:
                    pass
        self._image_resources = []
        self._image_key = None

    def release_pbos(self):
        if self.cuda_gl:
            for resource in self._pbo_resources:
                try:
                    self.cuda_gl.unregister_resource(resource)
                except Exception:
                    pass
        if self._pbos:
            try:
                glDeleteBuffers(len(self._pbos), [int(pbo) for pbo in self._pbos])
            except Exception:
                pass
        self._pbos = []
        self._pbo_resources = []
        self._pbo_key = None

    def upload_rgba(self, textures, tensors, width: int, height: int, *, prefer_image=True, build_mipmaps=True) -> str:
        if prefer_image and self.image_enabled:
            try:
                self._upload_image(textures, tensors, width, height, build_mipmaps=build_mipmaps)
                return "image"
            except Exception as exc:
                self.release_images()
                self.image_enabled = False
                self.image_failed_reason = str(exc)
                print(f"[{self.log_prefix}] CUDA/GL image texture upload failed: {exc}; using PBO fallback", flush=True)
        self._upload_pbo(textures, tensors, width, height, build_mipmaps=build_mipmaps)
        return "pbo"

    def _validate_rgba(self, tensor, width: int, height: int):
        if not (hasattr(tensor, "is_cuda") and tensor.is_cuda):
            raise RuntimeError("upload tensor is not CUDA")
        if tuple(tensor.shape) != (int(height), int(width), 4):
            raise RuntimeError(f"expected HWC RGBA {(height, width, 4)}, got {tuple(tensor.shape)}")
        if str(getattr(tensor, "dtype", "")) != "torch.uint8":
            raise RuntimeError(f"expected uint8 tensor, got {getattr(tensor, 'dtype', None)}")
        return tensor.contiguous()

    def _ensure_images(self, textures, width: int, height: int):
        if self.backend != "CUDA":
            raise RuntimeError(f"CUDA/GL image upload requires CUDA backend, got {self.backend}")
        for name in ("register_image", "map_graphics_resource", "mapped_array", "memcpy_2d_to_array"):
            if not hasattr(self.cuda_gl, name):
                raise RuntimeError(f"CUDA/GL image upload requires {name} support")
        key = (int(width), int(height), tuple(int(texture.glo) for texture in textures))
        if self._image_key == key and len(self._image_resources) == len(textures):
            return
        self.release_images()
        self._image_resources = [self.cuda_gl.register_image(texture.glo, GL_TEXTURE_2D) for texture in textures]
        self._image_key = key
        if self.debug:
            print(f"[{self.log_prefix}] CUDA/GL image resources registered {width}x{height}", flush=True)

    def _upload_image(self, textures, tensors, width: int, height: int, *, build_mipmaps: bool):
        self._ensure_images(textures, width, height)
        row_bytes = int(width) * 4
        for texture, resource, tensor in zip(textures, self._image_resources, tensors):
            tensor = self._validate_rgba(tensor, width, height)
            self.cuda_gl.map_graphics_resource(resource)
            try:
                array = self.cuda_gl.mapped_array(resource)
                self.cuda_gl.memcpy_2d_to_array(array, tensor.data_ptr(), row_bytes, row_bytes, int(height))
            finally:
                self.cuda_gl.unmap_resource(resource)
            if build_mipmaps:
                glBindTexture(GL_TEXTURE_2D, texture.glo)
                glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _ensure_pbos(self, count: int, width: int, height: int):
        key = (int(count), int(width), int(height))
        if self._pbo_key == key and len(self._pbos) == count:
            return
        self.release_pbos()
        ids = glGenBuffers(count)
        if count == 1:
            ids = [ids]
        self._pbos = [int(pbo) for pbo in ids]
        nbytes = int(width) * int(height) * 4
        for pbo in self._pbos:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self._pbo_resources = [self.cuda_gl.register_buffer(pbo) for pbo in self._pbos]
        self._pbo_key = key
        if self.debug:
            print(f"[{self.log_prefix}] CUDA/GL PBOs created {width}x{height}", flush=True)

    def _upload_pbo(self, textures, tensors, width: int, height: int, *, build_mipmaps: bool):
        self._ensure_pbos(len(textures), width, height)
        for texture, pbo, resource, tensor in zip(textures, self._pbos, self._pbo_resources, tensors):
            tensor = self._validate_rgba(tensor, width, height)
            ptr = self.cuda_gl.map_resource(resource)
            try:
                self.cuda_gl.memcpy_d2d(ptr, tensor.data_ptr(), tensor.nbytes)
            finally:
                self.cuda_gl.unmap_resource(resource)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
            glBindTexture(GL_TEXTURE_2D, texture.glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            if build_mipmaps:
                glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
