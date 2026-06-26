import math
import os

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils import ENV_ROWS, ROWS


class CoreOverlayPanelsMixin:
    def _render_fps_overlay(self, eye_index, mgl_fbo, vp_mat):
        """Render the FPS/latency text quad (head-relative or left-controller-attached)."""
        if self.screen_height is None:
            return

        now = self._frame_now

        # Update cached values once per second
        if now - self._last_overlay_update >= 1.0:
            self._cached_actual_fps = self.actual_fps
            self._cached_sbs_fps = self.sbs_fps
            self._cached_latency = self.total_latency
            self._cached_screen_width = self.screen_width
            self._cached_screen_height = self.screen_width * 9.0 / 16.0
            self._cached_screen_dist = self.screen_distance
            self._cached_depth_strength = float(getattr(self, 'depth_strength', 0.0) or 0.0)
            self._cached_vr_res = self._swapchain_sizes.get(0, (0, 0))
            self._cached_sbs_res = self.frame_size
            self._last_overlay_update = now

            if eye_index == 0 and self.font is not None:
                ow, oh = self._overlay_tex_size
                img = Image.new('RGBA', (ow, oh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rounded_rectangle(
                    [0, 0, ow - 1, oh - 1],
                    radius=14,
                    fill=(32, 32, 36, 210),
                )

                c_label = (150, 158, 185, 255)
                c_green = (0, 230, 90, 255)
                c_cyan = (0, 210, 230, 255)
                c_amber = (255, 190, 40, 255)
                bfont = self.bold_font or self.font

                pad = 14
                row0 = 22
                row1 = 56
                row2 = 90
                row3 = 124
                row4 = 158

                labels = ["[Performance]", "[3D Display]", "[Resolution]", "[Controller]", "[Environment]"]
                try:
                    max_lw = max(int(draw.textlength(label, font=bfont)) for label in labels)
                except AttributeError:
                    max_lw = max(
                        (int(bfont.getsize(label)[0]) if hasattr(bfont, 'getsize') else 190)
                        for label in labels
                    )
                val_x = pad + max_lw + 10

                def _draw_row(y, label, label_color, value, value_color):
                    draw.text((pad, y), label, font=bfont, fill=label_color)
                    draw.text((val_x, y), value, font=self.font, fill=value_color)

                lat_str = f"{self._cached_latency:.0f}ms" if self._cached_latency > 0 else "N/A"
                fps_str = (
                    f"XR {self._cached_actual_fps:.0f} FPS"
                    f"   SBS {self._cached_sbs_fps:.0f} FPS"
                    f"   Latency {lat_str}"
                )
                _draw_row(row0, "[Performance]", c_label, fps_str, c_green)
                if self._current_brand:
                    brand_str = f"Model: {self._current_brand}"
                    _draw_row(row3, "[Controller]", c_label, brand_str, c_cyan)
                scr_str = (
                    f"{self._cached_screen_width:.2f}"
                    f" x {self._cached_screen_height:.2f} m"
                    f"  @  {self._cached_screen_dist:.2f} m"
                    f"   Depth Strength {self._cached_depth_strength:.2f}"
                )
                _draw_row(row1, "[3D Display]", c_label, scr_str, c_cyan)

                vw, vh = self._cached_vr_res
                sw, sh = self._cached_sbs_res
                res_str = f"XR {vw}x{vh}/eye   Screen {sw}x{sh}"
                _draw_row(row2, "[Resolution]", c_label, res_str, c_amber)
                env_str = "ON" if self._env_model_visible else "OFF"
                _draw_row(row4, "[Environment]", c_label, env_str, c_cyan)

                data = np.flipud(np.array(img, dtype=np.uint8))
                self._overlay_tex.write(data.tobytes())

        overlay_h = 0.075
        ow, oh = self._overlay_tex_size
        overlay_w = overlay_h * (ow / oh)

        panel_pos = None
        panel_fwd = None
        panel_up = None

        if self._grip_mat_l is not None and self._aim_mat_l is not None:
            grip_right = self._grip_mat_l[:3, 0].astype('f8')
            grip_up = self._grip_mat_l[:3, 1].astype('f8')
            grip_fwd = self._grip_mat_l[:3, 2].astype('f8')
            grip_right /= np.linalg.norm(grip_right) + 1e-10
            grip_up /= np.linalg.norm(grip_up) + 1e-10
            grip_fwd /= np.linalg.norm(grip_fwd) + 1e-10

            fwd_w = -self._aim_mat_l[:3, 2].astype('f8')
            right_w = self._aim_mat_l[:3, 0].astype('f8')
            ang = math.radians(12)
            ca, sa = math.cos(ang), math.sin(ang)
            axis = right_w / (np.linalg.norm(right_w) + 1e-10)
            laser_fwd = fwd_w * ca + np.cross(axis, fwd_w) * sa + axis * np.dot(axis, fwd_w) * (1 - ca)
            laser_fwd /= np.linalg.norm(laser_fwd) + 1e-10

            grip_pos = self._grip_mat_l[:3, 3].astype('f8')
            laser_origin = grip_pos + grip_up * 0.020 + laser_fwd * 0.11

            toward_user = (-laser_fwd).astype('f8')
            panel_fwd = grip_up + toward_user
            panel_fwd /= np.linalg.norm(panel_fwd) + 1e-10
            panel_up = grip_up.copy()

            panel_right = np.cross(panel_up, panel_fwd)
            panel_right /= np.linalg.norm(panel_right) + 1e-10
            panel_up2 = np.cross(panel_fwd, panel_right)
            panel_up2 /= np.linalg.norm(panel_up2) + 1e-10

            panel_offset = 0.05
            top_ref = 0.10
            panel_pos = laser_origin + panel_fwd * panel_offset + panel_up2 * (top_ref - overlay_h / 2.0)

        if panel_pos is None and self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            panel_pos = np.array([hx + fx * 1.0, hy + fy * 1.0 - 0.15, hz + fz * 1.0], dtype='f8')
            panel_fwd = np.array([-fx, -fy, -fz], dtype='f8')
            panel_up = np.array([0.0, 1.0, 0.0], dtype='f8')

        if panel_pos is not None:
            scale = np.diag([overlay_w / 2.0, overlay_h / 2.0, 1.0, 1.0]).astype(np.float32)
            panel_right = np.cross(panel_up, panel_fwd)
            panel_right /= np.linalg.norm(panel_right) + 1e-10
            panel_up2 = np.cross(panel_fwd, panel_right)
            panel_up2 /= np.linalg.norm(panel_up2) + 1e-10
            rot = np.eye(4, dtype=np.float32)
            rot[:3, 0] = panel_right.astype(np.float32)
            rot[:3, 1] = panel_up2.astype(np.float32)
            rot[:3, 2] = panel_fwd.astype(np.float32)
            trans = np.eye(4, dtype=np.float32)
            trans[0, 3] = panel_pos[0]
            trans[1, 3] = panel_pos[1]
            trans[2, 3] = panel_pos[2]
            mvp = vp_mat @ trans @ rot @ scale
        else:
            mvp = vp_mat

        mgl_fbo.use()
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._overlay_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = 1.0
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

    def _render_brand_osd(self, eye_index, mgl_fbo, vp_mat):
        """Render controller brand indicator attached to the right controller."""
        if self._brand_osd_tex is None or self._current_brand is None:
            return

        now = self._frame_now

        if eye_index == 0 and self.font is not None:
            cur_name = self._current_brand
            if cur_name != self._brand_osd_last_name:
                self._brand_osd_last_name = cur_name
                self._brand_osd_show_t = now
                self._brand_osd_alpha = 1.0

                bw, bh = self._brand_osd_tex_size
                img = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rounded_rectangle(
                    [0, 0, bw - 1, bh - 1],
                    radius=12,
                    fill=(32, 32, 36, 210),
                )
                bfont = self.bold_font or self.font
                c_label = (150, 158, 185, 255)
                c_value = (0, 210, 230, 255)
                pad = 12
                cy = (bh - 32) // 2
                label = "Controller"
                draw.text((pad, cy), label, font=bfont, fill=c_label)
                try:
                    lw = int(draw.textlength(label, font=bfont))
                except AttributeError:
                    lw = int(bfont.getsize(label)[0]) if hasattr(bfont, 'getsize') else 60
                draw.text((pad + lw + 8, cy), cur_name, font=self.font, fill=c_value)
                data = np.flipud(np.array(img, dtype=np.uint8))
                self._brand_osd_tex.write(data.tobytes())

        hold = 1.5
        decay = 0.8
        elapsed = now - self._brand_osd_show_t
        if elapsed < hold:
            alpha = 1.0
        elif elapsed < hold + decay:
            alpha = 1.0 - (elapsed - hold) / decay
        else:
            alpha = 0.0
        self._brand_osd_alpha = alpha

        if alpha <= 0.0 or self._grip_mat_r is None:
            return

        osd_h = self.screen_width * 0.03
        bw, bh = self._brand_osd_tex_size
        osd_w = osd_h * (bw / bh)

        grip_pos = self._grip_mat_r[:3, 3].astype('f8')
        sh = self.screen_height
        if sh is None:
            fw, fh = self.frame_size
            if fh > fw:
                sh = self.screen_width
            else:
                sh = self.screen_width * 9.0 / 16.0
        bottom_edge = np.array(
            [self.screen_pan_x, self.screen_pan_y - sh / 2.0, -self.screen_distance],
            dtype='f8',
        )
        panel_pos = (bottom_edge + grip_pos) * 0.5

        if self._head_pos_w is not None:
            toward = np.array(self._head_pos_w, dtype='f8') - panel_pos
            toward /= np.linalg.norm(toward) + 1e-10
            panel_fwd = toward
        else:
            panel_fwd = np.array([0.0, 0.0, -1.0], dtype='f8')
        panel_up = np.array([0.0, 1.0, 0.0], dtype='f8')

        scale = np.diag([osd_w / 2.0, osd_h / 2.0, 1.0, 1.0]).astype(np.float32)
        panel_right = np.cross(panel_up, panel_fwd)
        panel_right /= np.linalg.norm(panel_right) + 1e-10
        panel_up2 = np.cross(panel_fwd, panel_right)
        panel_up2 /= np.linalg.norm(panel_up2) + 1e-10
        rot = np.eye(4, dtype=np.float32)
        rot[:3, 0] = panel_right.astype(np.float32)
        rot[:3, 1] = panel_up2.astype(np.float32)
        rot[:3, 2] = panel_fwd.astype(np.float32)
        trans = np.eye(4, dtype=np.float32)
        trans[0, 3] = panel_pos[0]
        trans[1, 3] = panel_pos[1]
        trans[2, 3] = panel_pos[2]
        mvp = vp_mat @ trans @ rot @ scale

        mgl_fbo.use()
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._brand_osd_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._brand_osd_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

    def _render_calibration_panel(self, mgl_fbo, vp_mat):
        """Draw semi-transparent calibration panel in VR (reuses FPS overlay shader)."""
        if self._overlay_prog is None:
            return
        now = self._frame_now
        # stale data check: only update texture every 0.5s to avoid excessive PIL overhead when using sticks to adjust values
        if not hasattr(self, '_calib_last_update'):
            self._calib_last_update = 0.0
        if now - self._calib_last_update < 0.5:
            if self._calib_tex is not None:
                self._render_overlay_quad(mgl_fbo, vp_mat, self._calib_tex,
                                        self._calib_tex_size, 0.6)
            return
        self._calib_last_update = now

        # Generate panel content with PIL: controller model + current temp offsets/rotation + instructions.
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return
        off = self._calibration_temp_offset
        rot = self._calibration_temp_rot
        lines = [
            f"Calibration: {self._current_brand or self._controller_model}",
            f"Y offset: {off[1]:.3f} m",
            f"Z offset: {off[2]:.3f} m",
            f"Rotation: {rot:.1f} deg",
            "",
            "L-stick U/D: Y  |  R-stick U/D: Z",
            "R-stick L/R: Rotation",
            "B: Save  Menu+A+B: Quit",
        ]
        fw, fh = 420, 200
        img = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, fw - 1, fh - 1], radius=12,
                            fill=(20, 20, 28, 200))
        # Cache the consola font once -re-loading from disk every 0.5s
        # (the panel's rebuild cadence) is wasted I/O.
        if not hasattr(self, '_calib_font') or self._calib_font is None:
            try:
                self._calib_font = ImageFont.truetype("consola.ttf", 20)
            except Exception:
                self._calib_font = ImageFont.load_default()
        font = self._calib_font
        y = 16
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text((16, y), line, font=font, fill=(200, 220, 255, 255))
            y += bbox[3] - bbox[1] + 4

        data = np.flipud(np.array(img, dtype=np.uint8))
        if self._calib_tex is None:
            self._calib_tex = self.ctx.texture((fw, fh), 4, dtype='f1')
            self._calib_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._calib_tex.write(data.tobytes())
        self._calib_tex_size = (fw, fh)

        self._render_overlay_quad(mgl_fbo, vp_mat, self._calib_tex,
                                (fw, fh), 0.6)

    def _render_overlay_quad(self, mgl_fbo, vp_mat, tex, tex_size, alpha):
        """Generate MVP for a quad in front of the user's view, with size based on tex_size and distance based on typical arm's length, then render with the given texture and alpha."""
        tex_w, tex_h = tex_size
        OVERLAY_H = 0.10
        OVERLAY_W = OVERLAY_H * (tex_w / tex_h)

        # Head-Locked Overlay: position the quad in front of the user's head, slightly below eye level, and oriented to always face the user. This is more comfortable for reading text and ensures the overlay is always visible regardless of head orientation. The quad's forward direction is aligned with the headset's forward direction projected onto the horizontal plane, so it remains readable even when looking down or up.
        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            wx = hx + fx * 1.2
            wy = hy + fy * 1.2 - 0.1
            wz = hz + fz * 1.2
            V3 = vp_mat[:3, :3]
            right = V3[0, :].copy(); right = right / (np.linalg.norm(right) + 1e-10)
            up    = V3[1, :].copy(); up    = up    / (np.linalg.norm(up)    + 1e-10)
            fwd   = -V3[2,:].copy(); fwd  = fwd  / (np.linalg.norm(fwd)   + 1e-10)
            S = np.diag([OVERLAY_W/2.0, OVERLAY_H/2.0, 1.0, 1.0]).astype(np.float32)
            R = np.eye(4, dtype=np.float32)
            R[:3, 0] = right; R[:3, 1] = up; R[:3, 2] = fwd
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = wx; T[1, 3] = wy; T[2, 3] = wz
            mvp = vp_mat @ T @ R @ S
        else:
            return
        mgl_fbo.use()
        # depth_mask = False  # do not write semi-transparent pixels to depth
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

    # Shortcuts help panel (right controller attachment)
    def _controller_guide_rows(self):
        return ENV_ROWS if getattr(self, 'ENVIRONMENT_MODE', False) else ROWS

    def _build_help_texture(self):
        """Generate help panel texture (3-column layout, adaptive size, Chinese font priority)."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return

        # font loading with fallback: first look for "font.ttf" in project root, then common system fonts (Windows/macOS/Linux), finally default PIL font.
        _font_paths = [
            # internal project font (supports Chinese, bundled with the app)
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "font.ttf"),
            # Windows system fonts
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            # macOS system fonts
            "/System/Library/Fonts/PingFang.ttc",
            # Linux common Chinese fonts
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        _font_size = 16
        _title_size = 18
        font = None
        for fp in _font_paths:
            if os.path.isfile(fp):
                try:
                    font  = ImageFont.truetype(fp, _font_size)
                    bfont = ImageFont.truetype(fp, _title_size)
                    break
                except Exception:
                    continue
        if font is None:
            try:
                font  = ImageFont.truetype("consola.ttf", _font_size)
                bfont = ImageFont.truetype("consolab.ttf", _title_size)
            except Exception:
                font  = ImageFont.load_default()
                bfont = font

        draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        # Measure column widths for 3-column layout
        col_w = [0, 0, 0]
        rows = self._controller_guide_rows()
        for r in rows:
            if len(r) >= 4:
                is_title = r[3]
                fields = list(r[:3])
            else:
                is_title = False
                fields = list(r[:3])
            for ci in range(3):
                b = draw.textbbox((0, 0), fields[ci], font=bfont if is_title else font)
                w = b[2] - b[0]
                if w > col_w[ci]:
                    col_w[ci] = w

        GAP = 20       # Column spacing
        MID_GAP = 50   # Gap between left and right section groups
        PAD_X = 30
        PAD_Y = 20
        LINE_H = (_font_size + 6) if font else 20

        # Split guide rows into left and right groups at the 4th section title (index 3).
        title_indices = [i for i, r in enumerate(rows) if len(r) >= 4 and r[3]]
        mid_idx = title_indices[4] if len(title_indices) > 4 else len(rows)

        left_rows = rows[:mid_idx]
        right_rows = rows[mid_idx:]

        left_h = len(left_rows) * LINE_H
        right_h = len(right_rows) * LINE_H
        th = max(left_h, right_h) + PAD_Y * 2

        inner_w = col_w[0] + GAP + col_w[1] + GAP + col_w[2]
        tw = inner_w * 2 + MID_GAP + PAD_X * 2

        if hasattr(self, '_help_tex') and self._help_tex is not None:
            try:
                self._help_tex.release()
            except Exception:
                pass
        self._help_tex = self.ctx.texture((tw, th), 4, dtype='f1')
        self._help_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._help_tex_size = (tw, th)

        img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, tw - 1, th - 1], radius=14,
                            fill=(18, 18, 28, 210))

        col_x = [PAD_X, PAD_X + col_w[0] + GAP,
                 PAD_X + col_w[0] + GAP + col_w[1] + GAP]
        col_x2 = [PAD_X + inner_w + MID_GAP,
                  PAD_X + inner_w + MID_GAP + col_w[0] + GAP,
                  PAD_X + inner_w + MID_GAP + col_w[0] + GAP + col_w[1] + GAP]

        # Render left column (sections 0-2)
        for ri, r in enumerate(left_rows):
            if len(r) >= 4:
                c1, c2, c3, is_title = r[0], r[1], r[2], r[3]
            else:
                c1, c2, c3, is_title = r[0], r[1], r[2], False
            y = PAD_Y + ri * LINE_H
            f = bfont if is_title else font
            color = (90, 190, 255, 255) if is_title else (200, 210, 235, 255)
            for ci, txt in enumerate([c1, c2, c3]):
                if txt:
                    draw.text((col_x[ci], y), txt, font=f, fill=color)

        # Render right column (sections 3+)
        for ri, r in enumerate(right_rows):
            if len(r) >= 4:
                c1, c2, c3, is_title = r[0], r[1], r[2], r[3]
            else:
                c1, c2, c3, is_title = r[0], r[1], r[2], False
            y = PAD_Y + ri * LINE_H
            f = bfont if is_title else font
            color = (90, 190, 255, 255) if is_title else (200, 210, 235, 255)
            for ci, txt in enumerate([c1, c2, c3]):
                if txt:
                    draw.text((col_x2[ci], y), txt, font=f, fill=color)

        data = np.flipud(np.array(img, dtype=np.uint8))
        self._help_tex.write(data.tobytes())

    def _build_team_help_texture(self):
        """Generate teammate single-column shortcuts texture."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return

        _font_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "font.ttf"),
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        _font_size = 21
        _title_size = 21
        font = None
        for fp in _font_paths:
            if os.path.isfile(fp):
                try:
                    font = ImageFont.truetype(fp, _font_size)
                    bfont = ImageFont.truetype(fp, _title_size)
                    break
                except Exception:
                    continue
        if font is None:
            try:
                font = ImageFont.truetype("consola.ttf", _font_size)
                bfont = ImageFont.truetype("consolab.ttf", _title_size)
            except Exception:
                font = ImageFont.load_default()
                bfont = font

        rows = ROWS
        draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        col_w = [0, 0, 0]
        for r in rows:
            for ci in range(3):
                b = draw.textbbox((0, 0), r[ci], font=bfont if r[3] else font)
                w = b[2] - b[0]
                if w > col_w[ci]:
                    col_w[ci] = w

        GAP = 20
        PAD_X = 30
        PAD_Y = 20
        LINE_H = (_font_size + 6) if font else 20
        tw = col_w[0] + GAP + col_w[1] + GAP + col_w[2] + PAD_X * 2
        th = len(rows) * LINE_H + PAD_Y * 2

        if self._team_help_tex is not None:
            try:
                self._team_help_tex.release()
            except Exception:
                pass
        self._team_help_tex = self.ctx.texture((tw, th), 4, dtype='f1')
        self._team_help_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._team_help_tex_size = (tw, th)

        img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, tw - 1, th - 1], radius=14,
                            fill=(18, 18, 28, 210))

        col_x = [PAD_X, PAD_X + col_w[0] + GAP,
                PAD_X + col_w[0] + GAP + col_w[1] + GAP]
        for ri, (c1, c2, c3, is_title) in enumerate(rows):
            y = PAD_Y + ri * LINE_H
            f = bfont if is_title else font
            color = (90, 190, 255, 255) if is_title else (200, 210, 235, 255)
            for ci, txt in enumerate([c1, c2, c3]):
                if txt:
                    draw.text((col_x[ci], y), txt, font=f, fill=color)

        data = np.flipud(np.array(img, dtype=np.uint8))
        self._team_help_tex.write(data.tobytes())

    def _render_team_status_overlay(self, eye_index, mgl_fbo, vp_mat):
        """Render teammate FPS/status panel below the screen bottom edge."""
        if self.screen_height is None or self._team_status_tex is None:
            return

        now = self._frame_now
        if now - self._team_last_overlay_update >= 1.0:
            if eye_index == 0 and self.font is not None:
                ow, oh = self._team_status_tex_size
                img = Image.new('RGBA', (ow, oh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rounded_rectangle([0, 0, ow - 1, oh - 1], radius=14, fill=(32, 32, 36, 210))

                C_LABEL = (150, 158, 185, 255)
                C_GREEN = (0, 230, 90, 255)
                C_CYAN = (0, 210, 230, 255)
                C_AMBER = (255, 190, 40, 255)
                label_font = self.bold_font or self.font
                value_font = self.font
                PAD = 14
                rows_y = [22, 56, 90, 124, 158]

                def _ascent(f):
                    try:
                        return f.getmetrics()[0]
                    except Exception:
                        return 0
                lbl_asc = _ascent(label_font)
                val_asc = _ascent(value_font)
                lbl_dy = max(0, val_asc - lbl_asc)
                val_dy = max(0, lbl_asc - val_asc)

                labels = ["[Performance]", "[3D Display]", "[Resolution]", "[Show Shortcuts]", "[Models]"]
                try:
                    max_lw = max(int(draw.textlength(l, font=label_font)) for l in labels)
                except AttributeError:
                    max_lw = max((int(label_font.getsize(l)[0]) if hasattr(label_font, 'getsize') else 190) for l in labels)
                VAL_X = PAD + max_lw + 10

                def _draw_row(y, label, value, value_color):
                    draw.text((PAD, y + lbl_dy), label, font=label_font, fill=C_LABEL)
                    draw.text((VAL_X, y + val_dy), value, font=value_font, fill=value_color)

                lat_str = f"{self.total_latency:.0f}ms" if self.total_latency > 0 else "--"
                fps_str = f"XR {self.actual_fps:.0f} FPS   SBS {self.sbs_fps:.0f} FPS   Latency {lat_str}"
                _draw_row(rows_y[0], "[Performance]", fps_str, C_GREEN)
                display_height_m = self.screen_width * 9.0 / 16.0
                display_depth_strength = float(getattr(self, 'depth_strength', 0.0) or 0.0)
                scr_str = (f"{self.screen_width:.2f} x {display_height_m:.2f} m"
                           f"  @  {self.screen_distance:.2f} m   Depth Strength {display_depth_strength:.2f}")
                _draw_row(rows_y[1], "[3D Display]", scr_str, C_CYAN)
                vw, vh = self._swapchain_sizes.get(0, (0, 0))
                sw, sh = self.frame_size
                _draw_row(rows_y[2], "[Resolution]", f"XR {vw}x{vh}/eye   Screen {sw}x{sh}", C_AMBER)
                env_str = (self._active_environment or self._environment_model or 'Default').strip() or 'Default'
                model_str = f"Environment: {env_str}"
                if self._current_brand:
                    model_str += f"   Controller: {self._current_brand}"
                _draw_row(rows_y[3], "[Models]", model_str, C_CYAN)

                draw.text((PAD, rows_y[4] + lbl_dy), "[Show Shortcuts]", font=label_font, fill=C_LABEL)
                SW_W, SW_H = 52, 26
                SW_X = VAL_X
                SW_Y = rows_y[4] + (34 - SW_H) // 2
                on = self._team_help_visible
                track_col = (0, 200, 80, 255) if on else (80, 84, 100, 255)
                draw.rounded_rectangle([SW_X, SW_Y, SW_X + SW_W, SW_Y + SW_H], radius=SW_H // 2, fill=track_col)
                KR = SW_H // 2 - 2
                KX = (SW_X + SW_W - KR - 3) if on else (SW_X + KR + 3)
                KY = SW_Y + SW_H // 2
                draw.ellipse([KX - KR, KY - KR, KX + KR, KY + KR], fill=(255, 255, 255, 255))

                data = np.flipud(np.array(img, dtype=np.uint8))
                self._team_status_tex.write(data.tobytes())
            self._team_last_overlay_update = now

        sh = self.screen_height
        sx = self.screen_width / 2.0
        sy = sh / 2.0
        GAP, OVERLAY_H = self._team_status_panel_metrics()
        ow, oh = self._team_status_tex_size
        OVERLAY_W = OVERLAY_H * (ow / oh)
        local_cx = -sx + OVERLAY_W / 2.0
        local_cy = -sy - GAP - OVERLAY_H / 2.0

        cy_s = math.cos(self.screen_yaw); sy_s = math.sin(self.screen_yaw)
        cp_s = math.cos(self.screen_pitch); sp_s = math.sin(self.screen_pitch)
        R = (np.array([[cy_s, 0, sy_s, 0], [0, 1, 0, 0], [-sy_s, 0, cy_s, 0], [0, 0, 0, 1]], dtype=np.float32) @
             np.array([[1, 0, 0, 0], [0, cp_s, -sp_s, 0], [0, sp_s, cp_s, 0], [0, 0, 0, 1]], dtype=np.float32))
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self.screen_pan_x
        T[1, 3] = self.screen_pan_y
        T[2, 3] = -self.screen_distance
        S_ov = np.diag([OVERLAY_W / 2.0, OVERLAY_H / 2.0, 1.0, 1.0]).astype(np.float32)
        T_local = np.eye(4, dtype=np.float32)
        T_local[0, 3] = local_cx
        T_local[1, 3] = local_cy
        model = T @ R @ T_local @ S_ov

        trigger_held = self._team_ov_ltrig_held or self._team_ov_rtrig_held
        if trigger_held:
            fade_speed = 8.0
            dt = max(0.001, getattr(self, '_last_frame_dt', 0.016))
            self._team_status_alpha = max(0.15, self._team_status_alpha - fade_speed * dt)
        else:
            self._team_status_alpha = 1.0

        mvp = vp_mat @ model
        mgl_fbo.use()
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._team_status_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = self._team_status_alpha
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

    def _render_team_help_panel(self, mgl_fbo, vp_mat):
        """Render teammate shortcuts panel hinged to the left side of the screen."""
        if self._team_help_tex is None or self.screen_height is None:
            return

        tex_w, tex_h = self._team_help_tex_size
        sh = self.screen_height
        sx = self.screen_width / 2.0
        GAP = sh * 0.02
        PANEL_H = sh
        PANEL_W = PANEL_H * (tex_w / tex_h)

        cy_s = math.cos(self.screen_yaw); sy_s = math.sin(self.screen_yaw)
        cp_s = math.cos(self.screen_pitch); sp_s = math.sin(self.screen_pitch)
        R_yaw = np.array([[cy_s, 0, sy_s, 0], [0, 1, 0, 0], [-sy_s, 0, cy_s, 0], [0, 0, 0, 1]], dtype=np.float32)
        R_pitch = np.array([[1, 0, 0, 0], [0, cp_s, -sp_s, 0], [0, sp_s, cp_s, 0], [0, 0, 0, 1]], dtype=np.float32)
        R = R_yaw @ R_pitch
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self.screen_pan_x
        T[1, 3] = self.screen_pan_y
        T[2, 3] = -self.screen_distance

        head_w = np.array(self._head_pos_w, dtype=np.float32) if self._head_pos_w is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        screen_c_w = np.array([self.screen_pan_x, self.screen_pan_y, -self.screen_distance], dtype=np.float32)
        R3 = R[:3, :3].astype(np.float32)
        head_local = R3.T @ (head_w - screen_c_w)
        right_edge_local = np.array([-sx - GAP, 0.0, 0.0], dtype=np.float32)
        to_user = head_local - right_edge_local
        to_user /= np.linalg.norm(to_user) + 1e-10
        theta = math.atan2(float(to_user[0]), float(to_user[2]))
        ct = math.cos(theta); st = math.sin(theta)

        T_right_edge = np.eye(4, dtype=np.float32)
        T_right_edge[0, 3] = -sx - GAP
        T_offset = np.eye(4, dtype=np.float32)
        T_offset[0, 3] = -PANEL_W / 2.0
        Ry = np.eye(4, dtype=np.float32)
        Ry[0, 0] = ct; Ry[0, 2] = st
        Ry[2, 0] = -st; Ry[2, 2] = ct
        S_panel = np.diag([PANEL_W / 2.0, PANEL_H / 2.0, 1.0, 1.0]).astype(np.float32)
        model = T @ R @ T_right_edge @ Ry @ T_offset @ S_panel

        mvp = vp_mat @ model
        mgl_fbo.use()
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._team_help_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = 0.75
        self._help_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

    def _render_help_panel(self, mgl_fbo, vp_mat):
        """Render the help/shortcut panel attached to the right controller.

        Attach exactly like the left-controller status panel: compute a controller-
        relative origin and basis so the panel moves/rotates with the right grip
        (no head-facing billboard behaviour).
        """
        if self._help_tex is None:
            return

        # Default panel size (world metres)
        tex_w, tex_h = self._help_tex_size
        PANEL_H = 0.2
        PANEL_W = PANEL_H * (tex_w / tex_h)

        panel_pos = None
        panel_fwd = None
        panel_up = None

        # Prefer right-controller attachment; fall back to head-relative placement
        if self._grip_mat_r is not None and self._aim_mat_r is not None:
            # Grip axes in world space (columns of grip_mat)
            grip_right = self._grip_mat_r[:3, 0].astype('f8')
            grip_up    = self._grip_mat_r[:3, 1].astype('f8')
            grip_fwd   = self._grip_mat_r[:3, 2].astype('f8')
            grip_right /= np.linalg.norm(grip_right) + 1e-10
            grip_up    /= np.linalg.norm(grip_up) + 1e-10
            grip_fwd   /= np.linalg.norm(grip_fwd) + 1e-10

            # Laser forward (same offset used for laser beam)
            fwd_w = -self._aim_mat_r[:3, 2].astype('f8')
            right_w = self._aim_mat_r[:3, 0].astype('f8')
            _ang = math.radians(12); _ca, _sa = math.cos(_ang), math.sin(_ang)
            _k = right_w / (np.linalg.norm(right_w) + 1e-10)
            laser_fwd = fwd_w * _ca + np.cross(_k, fwd_w) * _sa + _k * np.dot(_k, fwd_w) * (1 - _ca)
            laser_fwd /= np.linalg.norm(laser_fwd) + 1e-10

            # Laser origin similar to status panel
            grip_pos = self._grip_mat_r[:3, 3].astype('f8')
            laser_origin = grip_pos + grip_up * 0.020 + laser_fwd * 0.11

            # Panel forward: blend grip_up (button surface normal) with toward_user
            toward_user = (-laser_fwd).astype('f8')
            panel_fwd = grip_up + toward_user
            panel_fwd /= np.linalg.norm(panel_fwd) + 1e-10
            panel_up = grip_up.copy()

            # Pre-compute orthonormal basis
            _pr = np.cross(panel_up, panel_fwd)
            _pr /= np.linalg.norm(_pr) + 1e-10
            _pu2 = np.cross(panel_fwd, _pr)
            _pu2 /= np.linalg.norm(_pu2) + 1e-10

            PANEL_OFFSET = 0.05
            _top_ref = PANEL_H + 0.025  # bottom edge gap matches status panel (0.025)
            panel_pos = laser_origin + panel_fwd * PANEL_OFFSET + _pu2 * (_top_ref - PANEL_H / 2.0)

        if panel_pos is None and self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            panel_pos = np.array([hx + fx * 1.2, hy + fy * 1.2 - 0.3, hz + fz * 1.2], dtype='f8')
            panel_fwd = np.array([-fx, -fy, -fz], dtype='f8')
            panel_up  = np.array([0.0, 1.0, 0.0], dtype='f8')

        # Head-facing mode: reorient panel toward user's head (keep position attached to controller)
        if self._panel_mode == 0 and panel_pos is not None and self._head_pos_w is not None:
            toward = np.array(self._head_pos_w, dtype='f8') - panel_pos
            panel_fwd = toward / (np.linalg.norm(toward) + 1e-10)
            panel_up  = np.array([0.0, 1.0, 0.0], dtype='f8')

        if panel_pos is not None:

            S = np.diag([PANEL_W/2.0, PANEL_H/2.0, 1.0, 1.0]).astype(np.float32)
            panel_right = np.cross(panel_up, panel_fwd)
            panel_right /= np.linalg.norm(panel_right) + 1e-10
            panel_up2 = np.cross(panel_fwd, panel_right)
            panel_up2 /= np.linalg.norm(panel_up2) + 1e-10
            R = np.eye(4, dtype=np.float32)
            R[:3, 0] = panel_right.astype(np.float32)
            R[:3, 1] = panel_up2.astype(np.float32)
            R[:3, 2] = panel_fwd.astype(np.float32)
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = panel_pos[0]; T[1, 3] = panel_pos[1]; T[2, 3] = panel_pos[2]
            mvp = vp_mat @ T @ R @ S
        else:
            return

        mgl_fbo.use()
        # depth_mask = False  # do not write semi-transparent pixels to depth
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._help_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = 0.75
        self._help_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
