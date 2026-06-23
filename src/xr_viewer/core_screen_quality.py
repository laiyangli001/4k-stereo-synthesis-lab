import math

import moderngl
import numpy as np


class CoreScreenQualityMixin:
    def _screen_footprint_px(self, mvp, swapchain_size):
        sc_w, sc_h = int(swapchain_size[0]), int(swapchain_size[1])
        corners = np.array([
            [-1.0, -1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0, 1.0],
        ], dtype=np.float32)
        clip = (mvp @ corners.T).T
        valid = np.abs(clip[:, 3]) > 1e-6
        if not np.any(valid):
            return None
        ndc = clip[valid, :3] / clip[valid, 3:4]
        px = (ndc[:, 0] * 0.5 + 0.5) * sc_w
        py = (ndc[:, 1] * 0.5 + 0.5) * sc_h
        min_x, max_x = float(np.min(px)), float(np.max(px))
        min_y, max_y = float(np.min(py)), float(np.max(py))
        footprint_w = max(0.0, min(max_x, sc_w) - max(min_x, 0.0))
        footprint_h = max(0.0, min(max_y, sc_h) - max(min_y, 0.0))
        return footprint_w, footprint_h

    def _log_screen_footprint_once(self, eye_index, mvp, swapchain_size):
        key = (
            int(eye_index),
            tuple(swapchain_size),
            tuple(self._texture_size or (0, 0)),
            round(float(self.screen_width), 3),
            round(float(self.screen_height or 0.0), 3),
            round(float(self.screen_distance), 3),
        )
        if key in self._screen_footprint_logged:
            return
        self._screen_footprint_logged.add(key)
        try:
            sc_w, sc_h = int(swapchain_size[0]), int(swapchain_size[1])
            footprint_px = self._screen_footprint_px(mvp, swapchain_size)
            if footprint_px is None:
                footprint = "unknown"
            else:
                footprint_w, footprint_h = footprint_px
                footprint = f"{int(round(footprint_w))}x{int(round(footprint_h))}"
            tex_w, tex_h = self._texture_size or (0, 0)
            print(
                f"[OpenXRViewer] screen footprint eye={eye_index} approx={footprint} "
                f"swapchain={sc_w}x{sc_h} texture={int(tex_w)}x{int(tex_h)} "
                f"screen_m={self.screen_width:.3f}x{self.screen_height:.3f} "
                f"distance_m={self.screen_distance:.3f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[OpenXRViewer] screen footprint unavailable: {type(exc).__name__}: {exc}", flush=True)

    def _screen_quality_target_size(self, mvp, swapchain_size, source_size):
        if not self._screen_quality_filter:
            return None
        footprint = self._screen_footprint_px(mvp, swapchain_size)
        if footprint is None:
            return None
        src_w, src_h = int(source_size[0]), int(source_size[1])
        if src_w <= 0 or src_h <= 0:
            return None
        scale = float(self._screen_quality_oversample)
        target_w = int(math.ceil(max(16.0, footprint[0] * scale) / 16.0) * 16)
        target_h = int(math.ceil(max(16.0, footprint[1] * scale) / 16.0) * 16)
        target_w = min(src_w, max(16, target_w)) & ~1
        target_h = min(src_h, max(16, target_h)) & ~1
        return target_w, target_h

    def _release_screen_quality_resources(self):
        for name in (
            '_screen_ds_fbo',
            '_screen_rcas_fbo',
            '_screen_ds_tex',
            '_screen_rcas_tex',
            '_glow_ds_fbo',
            '_glow_ds_tex',
        ):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
                setattr(self, name, None)
        self._screen_quality_size = None
        self._glow_ds_size = None

    def _ensure_screen_quality_resources(self, size):
        if self._screen_quality_size == tuple(size) and self._screen_ds_tex is not None and self._screen_rcas_tex is not None:
            return
        self._release_screen_quality_resources()
        w, h = int(size[0]), int(size[1])
        self._screen_ds_tex = self.ctx.texture((w, h), 4, dtype='f1')
        self._screen_ds_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._screen_rcas_tex = self.ctx.texture((w, h), 4, dtype='f1')
        self._screen_rcas_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._screen_ds_fbo = self.ctx.framebuffer(color_attachments=[self._screen_ds_tex])
        self._screen_rcas_fbo = self.ctx.framebuffer(color_attachments=[self._screen_rcas_tex])
        self._screen_quality_size = (w, h)
        if self._screen_quality_logged_size != (w, h):
            print(f"[OpenXRViewer] screen quality filter active: {w}x{h}, RCAS={self._screen_quality_sharpness:.2f}")
            self._screen_quality_logged_size = (w, h)

    def _prepare_screen_quality_texture(self, source_tex, source_size, mvp, swapchain_size, source_label='color'):
        if source_tex is None or source_size is None:
            return None
        target_size = self._screen_quality_target_size(mvp, swapchain_size, source_size)
        if target_size is None:
            return None
        log_key = (source_label, target_size)
        if log_key not in self._screen_quality_logged_sources:
            print(f"[OpenXRViewer] screen quality source={source_label} target={target_size[0]}x{target_size[1]}")
            self._screen_quality_logged_sources.add(log_key)
        self._ensure_screen_quality_resources(target_size)
        src_w, src_h = int(source_size[0]), int(source_size[1])
        out_w, out_h = target_size

        prev_viewport = self.ctx.viewport
        prev_depth_mask = self.ctx.depth_mask
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = False

        self._screen_ds_fbo.use()
        self.ctx.viewport = (0, 0, out_w, out_h)
        source_tex.use(location=0)
        self._screen_ds_prog['u_input_size'].value = (float(src_w), float(src_h))
        self._screen_ds_vao.render(moderngl.TRIANGLE_STRIP)

        self._screen_rcas_fbo.use()
        self.ctx.viewport = (0, 0, out_w, out_h)
        self._screen_ds_tex.use(location=0)
        self._screen_rcas_prog['u_output_size'].value = (float(out_w), float(out_h))
        self._screen_rcas_prog['u_sharpness'].value = float(self._screen_quality_sharpness)
        self._screen_rcas_vao.render(moderngl.TRIANGLE_STRIP)

        self.ctx.viewport = prev_viewport
        self.ctx.depth_mask = prev_depth_mask
        self.ctx.enable(moderngl.DEPTH_TEST)
        return self._screen_rcas_tex

    def _ensure_glow_downsample_resources(self, size):
        if self._glow_ds_size == tuple(size) and self._glow_ds_tex is not None and self._glow_ds_fbo is not None:
            return
        for name in ('_glow_ds_fbo', '_glow_ds_tex'):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
                setattr(self, name, None)
        w, h = int(size[0]), int(size[1])
        self._glow_ds_tex = self.ctx.texture((w, h), 4, dtype='f1')
        self._glow_ds_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._glow_ds_fbo = self.ctx.framebuffer(color_attachments=[self._glow_ds_tex])
        self._glow_ds_size = (w, h)

    def _prepare_glow_downsample_texture(self, source_tex, source_size):
        if source_tex is None or source_size is None:
            return None
        src_w, src_h = int(source_size[0]), int(source_size[1])
        if src_w <= 0 or src_h <= 0:
            return None
        out_w = max(32, min(192, src_w // 20))
        out_h = max(18, min(108, src_h // 20))
        out_w = max(2, out_w & ~1)
        out_h = max(2, out_h & ~1)
        self._ensure_glow_downsample_resources((out_w, out_h))
        cache_key = (
            int(getattr(self, '_frame_count', 0)),
            int(getattr(self, '_current_eye_index', 0) or 0),
            int(getattr(source_tex, 'glo', 0) or 0),
            src_w,
            src_h,
            out_w,
            out_h,
        )
        if getattr(self, '_glow_ds_cache_key', None) == cache_key:
            return self._glow_ds_tex

        prev_viewport = self.ctx.viewport
        prev_depth_mask = self.ctx.depth_mask
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = False

        self._glow_ds_fbo.use()
        self.ctx.viewport = (0, 0, out_w, out_h)
        source_tex.use(location=0)
        self._glow_ds_prog['u_input_size'].value = (float(src_w), float(src_h))
        self._glow_ds_vao.render(moderngl.TRIANGLE_STRIP)
        self._glow_ds_cache_key = cache_key

        self.ctx.viewport = prev_viewport
        self.ctx.depth_mask = prev_depth_mask
        self.ctx.enable(moderngl.DEPTH_TEST)
        return self._glow_ds_tex
