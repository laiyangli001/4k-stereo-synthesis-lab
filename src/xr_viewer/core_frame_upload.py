import ctypes
import time

import moderngl
import numpy as np
from OpenGL.GL import (
    glBindBuffer,
    glBindTexture,
    glBufferData,
    glBufferSubData,
    glDeleteBuffers,
    glGenBuffers,
    glGenerateMipmap,
    glTexParameterf,
    glTexSubImage2D,
    GL_DYNAMIC_DRAW,
    GL_FLOAT,
    GL_PIXEL_UNPACK_BUFFER,
    GL_RED,
    GL_RGB,
    GL_STREAM_DRAW,
    GL_TEXTURE_2D,
    GL_TEXTURE_LOD_BIAS,
    GL_UNSIGNED_BYTE,
)
from viewer.viewer import BACKEND

try:
    from viewer.viewer import CUDART_GL
except ImportError:
    CUDART_GL = None

from .constants import _GLOW_GRID_COLS, _GLOW_GRID_ROWS


class CoreFrameUploadMixin:
    def _init_textures(self, w, h):
        if self.color_tex:
            self.color_tex.release()
        if self.depth_tex:
            self.depth_tex.release()
        self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
        self.color_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.color_tex.build_mipmaps()
        try:
            self.color_tex.anisotropy = 16.0
        except Exception:
            pass
        # Negative LOD bias: bias the sampler toward sharper (higher-res) mip levels.
        # -0.5 = use a mip level 0.5 finer than the GPU would naturally pick,
        # preserving anti-aliasing while recovering perceived sharpness.
        glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.5)
        glBindTexture(GL_TEXTURE_2D, 0)
        # Depth texture: plain LINEAR -NO mipmaps.
        # Mipmapping depth averages foreground+background values at edges,
        # producing wrong depth that breaks the DIBR shift formula and
        # disocclusion detection. viewer.py (FullSBS reference) also uses
        # default LINEAR; this keeps openxr_viewer DIBR output numerically
        # consistent with viewer.py for the same RGB+depth input.
        self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
        self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture_size = (w, h)

    def _init_cuda_pbos(self, w, h):
        """Create or recreate PBOs and register them with CUDA/HIP."""
        if not self._cuda_gl or BACKEND not in ("CUDA", "HIP"):
            return
        # Unregister old resources before deleting PBOs
        if self._pbo_color is not None:
            try:
                self._cuda_gl.unregister_resource(self._cuda_res_color)
                self._cuda_gl.unregister_resource(self._cuda_res_depth)
                glDeleteBuffers(2, [self._pbo_color, self._pbo_depth])
            except Exception:
                pass

        ids = glGenBuffers(2)
        self._pbo_color = int(ids[0])
        self._pbo_depth = int(ids[1])

        for pbo_id, nbytes in [
            (self._pbo_color, w * h * 3),   # RGB uint8
            (self._pbo_depth, w * h * 4),   # float32
        ]:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        self._cuda_res_color = self._cuda_gl.register_buffer(self._pbo_color)
        self._cuda_res_depth = self._cuda_gl.register_buffer(self._pbo_depth)
        self._pbo_texture_size = (w, h)
        print(f"[OpenXRViewer] GPU interop PBOs created ({BACKEND}) {w}x{h}")

    def _init_cpu_pbos(self, w, h):
        """Create unpack PBOs for CPU-path texture upload."""
        if getattr(self, '_cpu_pbo_color', None) is not None:
            try:
                glDeleteBuffers(2, [self._cpu_pbo_color, self._cpu_pbo_depth])
            except Exception:
                pass
        try:
            ids = glGenBuffers(2)
            self._cpu_pbo_color = int(ids[0])
            self._cpu_pbo_depth = int(ids[1])
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._cpu_pbo_color)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 3, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._cpu_pbo_depth)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self._cpu_pbo_size = (w, h)
            print(f"[OpenXRViewer] CPU-path PBOs created {w}x{h}")
        except Exception as exc:
            print(f"[OpenXRViewer] CPU PBO init failed, using direct upload: {exc}")
            self._cpu_pbo_color = None
            self._cpu_pbo_depth = None
            self._cpu_pbo_size = (0, 0)
            try:
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            except Exception:
                pass

    def _sample_glow_target_color(self, rgb, is_tensor):
        """Update glow target from a thin frame border with minimal CPU work."""
        try:
            if is_tensor:
                rgb_t = rgb.detach()
                if rgb_t.ndim == 4:
                    rgb_t = rgb_t[0]
                if rgb_t.ndim == 3 and rgb_t.shape[0] in (3, 4):
                    channels_first = True
                    rgb_t = rgb_t[:3]
                    h, w = int(rgb_t.shape[1]), int(rgb_t.shape[2])
                elif rgb_t.ndim == 3 and rgb_t.shape[-1] >= 3:
                    channels_first = False
                    rgb_t = rgb_t[..., :3]
                    h, w = int(rgb_t.shape[0]), int(rgb_t.shape[1])
                else:
                    return
                bt = max(1, int(min(h, w) * 0.08))

                top_h = min(bt, h)
                bot_h = min(bt, h)
                if channels_first:
                    total = rgb_t[:, :top_h, :].float().sum(dim=(1, 2))
                    total = total + rgb_t[:, max(0, h - bot_h):, :].float().sum(dim=(1, 2))
                else:
                    total = rgb_t[:top_h, :, :].float().sum(dim=(0, 1))
                    total = total + rgb_t[max(0, h - bot_h):, :, :].float().sum(dim=(0, 1))
                count = (top_h * w) + (bot_h * w)

                mid_h = max(0, h - top_h - bot_h)
                side_w = min(bt, w)
                if mid_h > 0 and side_w > 0:
                    y0 = top_h
                    y1 = h - bot_h
                    if channels_first:
                        total = total + rgb_t[:, y0:y1, :side_w].float().sum(dim=(1, 2))
                        total = total + rgb_t[:, y0:y1, max(0, w - side_w):].float().sum(dim=(1, 2))
                    else:
                        total = total + rgb_t[y0:y1, :side_w, :].float().sum(dim=(0, 1))
                        total = total + rgb_t[y0:y1, max(0, w - side_w):, :].float().sum(dim=(0, 1))
                    count += mid_h * side_w * 2

                avg_t = (total / max(1, count)).float()
                if avg_t.numel() and float(avg_t.detach().max().item()) <= 1.0:
                    avg = avg_t.clamp(0.0, 1.0).detach().cpu().numpy()
                    scale = 1.0
                else:
                    avg = avg_t.clamp(0.0, 255.0).detach().cpu().numpy()
                    scale = 255.0
                self._glow_target_color = (
                    float(avg[0]) / scale,
                    float(avg[1]) / scale,
                    float(avg[2]) / scale,
                )
                stride = 8
                grid = []
                x_edges = tuple((w * i) // _GLOW_GRID_COLS for i in range(_GLOW_GRID_COLS + 1))
                y_edges = tuple((h * i) // _GLOW_GRID_ROWS for i in range(_GLOW_GRID_ROWS + 1))
                for row in range(_GLOW_GRID_ROWS):
                    y0, y1 = y_edges[row], y_edges[row + 1]
                    for col in range(_GLOW_GRID_COLS):
                        x0, x1 = x_edges[col], x_edges[col + 1]
                        if x1 <= x0 or y1 <= y0:
                            grid.append(self._glow_target_color)
                            continue
                        if channels_first:
                            region = rgb_t[:, y0:y1:stride, x0:x1:stride].float()
                            avg_t = region.mean(dim=(1, 2))
                        else:
                            region = rgb_t[y0:y1:stride, x0:x1:stride, :].float()
                            avg_t = region.mean(dim=(0, 1))
                        if avg_t.numel() and float(avg_t.detach().max().item()) <= 1.0:
                            avg3 = avg_t.clamp(0.0, 1.0).detach().cpu().numpy()
                            scale3 = 1.0
                        else:
                            avg3 = avg_t.clamp(0.0, 255.0).detach().cpu().numpy()
                            scale3 = 255.0
                        grid.append((float(avg3[0]) / scale3, float(avg3[1]) / scale3, float(avg3[2]) / scale3))
                self._screen_light_target_colors = tuple(grid)
                return

            rgb_np = np.asarray(rgb, dtype=np.uint8)
            h, w = rgb_np.shape[:2]
            bt = max(1, int(min(h, w) * 0.08))
            top_h = min(bt, h)
            bot_h = min(bt, h)
            step = 4

            total = rgb_np[:top_h:step, ::step, :].sum(axis=(0, 1), dtype=np.float64)
            total += rgb_np[max(0, h - bot_h)::step, ::step, :].sum(axis=(0, 1), dtype=np.float64)
            count = (len(range(0, top_h, step)) + len(range(0, bot_h, step))) * len(range(0, w, step))

            mid_h = max(0, h - top_h - bot_h)
            side_w = min(bt, w)
            if mid_h > 0 and side_w > 0:
                y0 = top_h
                y1 = h - bot_h
                total += rgb_np[y0:y1:step, :side_w:step, :].sum(axis=(0, 1), dtype=np.float64)
                total += rgb_np[y0:y1:step, max(0, w - side_w)::step, :].sum(axis=(0, 1), dtype=np.float64)
                count += len(range(y0, y1, step)) * len(range(0, side_w, step)) * 2

            avg = total / max(1, count)
            self._glow_target_color = (
                float(avg[0]) / 255.0,
                float(avg[1]) / 255.0,
                float(avg[2]) / 255.0,
            )
            stride = 8
            grid = []
            x_edges = tuple((w * i) // _GLOW_GRID_COLS for i in range(_GLOW_GRID_COLS + 1))
            y_edges = tuple((h * i) // _GLOW_GRID_ROWS for i in range(_GLOW_GRID_ROWS + 1))
            for row in range(_GLOW_GRID_ROWS):
                y0, y1 = y_edges[row], y_edges[row + 1]
                for col in range(_GLOW_GRID_COLS):
                    x0, x1 = x_edges[col], x_edges[col + 1]
                    if x1 <= x0 or y1 <= y0:
                        grid.append(self._glow_target_color)
                        continue
                    avg3 = rgb_np[y0:y1:stride, x0:x1:stride, :].mean(axis=(0, 1))
                    grid.append((float(avg3[0]) / 255.0, float(avg3[1]) / 255.0, float(avg3[2]) / 255.0))
            self._screen_light_target_colors = tuple(grid)
        except Exception:
            pass

    # Per-frame helpers
    def _update_frame(self, rgb, depth):
        """Upload RGB and depth to GL textures -GPU path when available, CPU fallback."""
        import torch

        perf_enabled = bool(getattr(self, '_openxr_perf_log', False))
        perf_t0 = time.perf_counter() if perf_enabled else 0.0
        perf_last = perf_t0
        perf_marks = []

        def _mark_upload(label):
            nonlocal perf_last
            if not perf_enabled:
                return
            now = time.perf_counter()
            perf_marks.append((label, (now - perf_last) * 1000.0))
            perf_last = now

        self._runtime_direct_source = False
        is_tensor = hasattr(rgb, 'data_ptr')

        # Resolve depth shape and GPU tensor
        if hasattr(depth, 'detach'):
            depth_gpu = depth.detach().contiguous().float()
            h, w = depth_gpu.shape[0], depth_gpu.shape[1]
            depth_np = None
        else:
            depth_gpu = None
            depth_np = np.asarray(depth, dtype=np.float32)
            h, w = depth_np.shape[0], depth_np.shape[1]
        if perf_enabled:
            _mark_upload('shape')

        if self._use_d3d11 and self._d3d11_native_renderer is not None:
            try:
                self._d3d11_native_renderer.update_frame(rgb, depth)
                if perf_enabled:
                    _mark_upload('d3d11_update_frame')
                self.frame_size = (w, h)
                self.screen_height = None
                self._maybe_sample_glow_target_color(rgb, is_tensor)
                if perf_enabled:
                    _mark_upload('sample_glow')
                if perf_enabled:
                    self._log_upload_perf_if_slow(perf_t0, perf_marks, w, h, 'd3d11')
                return
            except Exception as e:
                print(f"[OpenXRViewer] D3D11 native frame upload failed: {e}")
                try:
                    self._d3d11_native_renderer.cleanup()
                except Exception:
                    pass
                self._d3d11_native_renderer = None
                self._texture_size = None

        if self._texture_size != (w, h):
            self._init_textures(w, h)
            self.frame_size = (w, h)
            self.screen_height = None
            if perf_enabled:
                _mark_upload('init_textures')

        # Lazy GPU interop init (includes PBO registration to verify interop)
        if self._cuda_gl is None and CUDART_GL is not None and BACKEND in ("CUDA", "HIP"):
            try:
                self._cuda_gl = CUDART_GL()
                self._init_cuda_pbos(w, h)   # create PBOs + register with HIP
                print(f"[OpenXRViewer] GPU interop active ({BACKEND})")
                if perf_enabled:
                    _mark_upload('init_cuda_gl')
            except Exception as e:
                print(f"[OpenXRViewer] GPU interop unavailable: {e}")
                self._cuda_gl = False   # sentinel: don't retry
                if perf_enabled:
                    _mark_upload('init_cuda_gl_failed')

        gpu_ok = bool(self._cuda_gl) and is_tensor and BACKEND in ("CUDA", "HIP")

        if gpu_ok:
            if self._pbo_texture_size != (w, h):
                self._init_cuda_pbos(w, h)
                if perf_enabled:
                    _mark_upload('resize_cuda_pbos')

            # Color: CHW tensor ->HWC contiguous uint8 on GPU, DMA into PBO
            rgb_gpu = rgb.permute(1, 2, 0).contiguous().clamp(0, 255).to(torch.uint8)
            if perf_enabled:
                _mark_upload('rgb_prepare_gpu')
            ptr = self._cuda_gl.map_resource(self._cuda_res_color)
            self._cuda_gl.memcpy_d2d(ptr, rgb_gpu.data_ptr(), rgb_gpu.nbytes)
            self._cuda_gl.unmap_resource(self._cuda_res_color)
            if perf_enabled:
                _mark_upload('rgb_cuda_copy')
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_color)
            glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            if perf_enabled:
                _mark_upload('rgb_tex_sub_image')
            glGenerateMipmap(GL_TEXTURE_2D)
            if perf_enabled:
                _mark_upload('rgb_mipmap')
            glBindTexture(GL_TEXTURE_2D, 0)

            ptr = self._cuda_gl.map_resource(self._cuda_res_depth)
            self._cuda_gl.memcpy_d2d(ptr, depth_gpu.contiguous().data_ptr(), depth_gpu.nbytes)
            self._cuda_gl.unmap_resource(self._cuda_res_depth)
            if perf_enabled:
                _mark_upload('depth_cuda_copy')
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_depth)
            glBindTexture(GL_TEXTURE_2D, self.depth_tex.glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, ctypes.c_void_p(0))
            if perf_enabled:
                _mark_upload('depth_tex_sub_image')
            # No glGenerateMipmap for depth: keep DIBR sampling at full-res
            # to match viewer.py FullSBS numerics.
            glBindTexture(GL_TEXTURE_2D, 0)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # CPU fallback - use PBO for async DMA when available.
            if hasattr(rgb, 'detach'):
                rgb_np = (
                    rgb.permute(1, 2, 0).detach().contiguous()
                    .clamp(0, 255).to(torch.uint8).cpu().numpy()
                )
            else:
                rgb_np = np.asarray(rgb, dtype=np.uint8)
            if perf_enabled:
                _mark_upload('rgb_to_cpu')
            if depth_np is None:
                depth_np = depth_gpu.cpu().numpy()
            if perf_enabled:
                _mark_upload('depth_to_cpu')
            rgb_bytes = rgb_np.astype('uint8', copy=False).tobytes()
            depth_bytes = depth_np.tobytes()
            if perf_enabled:
                _mark_upload('bytes')
            cpu_pbo = getattr(self, '_cpu_pbo_color', None)
            if cpu_pbo is not None and self._cpu_pbo_size == (w, h):
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._cpu_pbo_color)
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, len(rgb_bytes), rgb_bytes)
                if perf_enabled:
                    _mark_upload('cpu_rgb_buffer')
                glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
                if perf_enabled:
                    _mark_upload('cpu_rgb_tex')
                glGenerateMipmap(GL_TEXTURE_2D)
                if perf_enabled:
                    _mark_upload('cpu_rgb_mipmap')
                glBindTexture(GL_TEXTURE_2D, 0)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._cpu_pbo_depth)
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, len(depth_bytes), depth_bytes)
                if perf_enabled:
                    _mark_upload('cpu_depth_buffer')
                glBindTexture(GL_TEXTURE_2D, self.depth_tex.glo)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, ctypes.c_void_p(0))
                if perf_enabled:
                    _mark_upload('cpu_depth_tex')
                glBindTexture(GL_TEXTURE_2D, 0)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            else:
                if cpu_pbo is None or self._cpu_pbo_size != (w, h):
                    self._init_cpu_pbos(w, h)
                    if perf_enabled:
                        _mark_upload('init_cpu_pbos')
                self.color_tex.write(rgb_bytes)
                if perf_enabled:
                    _mark_upload('cpu_color_write')
                glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
                glGenerateMipmap(GL_TEXTURE_2D)
                if perf_enabled:
                    _mark_upload('cpu_color_mipmap')
                glBindTexture(GL_TEXTURE_2D, 0)
                self.depth_tex.write(depth_bytes)
                if perf_enabled:
                    _mark_upload('cpu_depth_write')
            # No glGenerateMipmap for depth: keep DIBR sampling at full-res
            # to match viewer.py FullSBS numerics.

        self._maybe_sample_glow_target_color(rgb, is_tensor)
        if perf_enabled:
            _mark_upload('sample_glow')
        if perf_enabled:
            self._log_upload_perf_if_slow(perf_t0, perf_marks, w, h, 'gpu' if gpu_ok else 'cpu')

    def _log_upload_perf_if_slow(self, perf_t0, perf_marks, w, h, path):
        total_ms = (time.perf_counter() - perf_t0) * 1000.0
        if total_ms < 20.0:
            return
        parts = ' '.join(f'{label}={ms:.1f}' for label, ms in perf_marks if ms >= 0.05)
        print(f"[OpenXRViewer] upload segments path={path} size={w}x{h} total_ms={total_ms:.1f} {parts}")

    def _maybe_sample_glow_target_color(self, rgb, is_tensor):
        """Sample frame color only when glow or cinema spill lighting consumes it."""
        glow_active = False
        env_spill_active = False
        env_static_spill_active = (
            getattr(self, '_bg_color_idx', 0) != 1
            and bool(getattr(self, '_env_model_visible', False))
            and bool(getattr(self, '_env_model_prims', []))
            and float(getattr(self, '_screen_light_intensity', 0.0)) > 0.0
        )
        if glow_active or env_spill_active:
            self._glow_color_counter = int(getattr(self, '_glow_color_counter', 0)) + 1
            interval = max(1, int(getattr(self, '_screen_light_sample_interval', 15)))
            if self._glow_color_counter >= interval:
                self._glow_color_counter = 0
                self._sample_glow_target_color(rgb, is_tensor)
        elif env_static_spill_active:
            self._glow_color_counter = 0
