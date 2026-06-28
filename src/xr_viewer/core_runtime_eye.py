import ctypes

import moderngl
import numpy as np
from OpenGL.GL import (
    glBindBuffer,
    glBindTexture,
    glBufferData,
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
    GL_TEXTURE_2D,
    GL_TEXTURE_LOD_BIAS,
    GL_UNSIGNED_BYTE,
)
from viewer.viewer import BACKEND

try:
    from viewer.viewer import CUDART_GL
except ImportError:
    CUDART_GL = None


def _runtime_shader_render_width(value) -> int:
    if isinstance(value, (tuple, list)) and len(value) >= 1:
        try:
            return max(0, int(value[0]))
        except (TypeError, ValueError):
            return 0
    if isinstance(value, str) and "x" in value:
        try:
            return max(0, int(value.split("x", 1)[0]))
        except (TypeError, ValueError):
            return 0
    return 0


class CoreRuntimeEyeMixin:
    def _runtime_eye_source_mean(self, frame):
        try:
            if hasattr(frame, 'detach'):
                tensor = frame.detach()
                mean = float(tensor.float().mean().item())
                if not tensor.is_floating_point():
                    mean /= 255.0
                return mean
            arr = np.asarray(frame).astype(np.float32, copy=False)
            if np.issubdtype(np.asarray(frame).dtype, np.integer):
                arr = arr / 255.0
            return float(arr.mean())
        except Exception:
            return None

    def _log_runtime_eye_stats_once(self, runtime_result, *, upload_path):
        if getattr(self, '_runtime_eye_stats_logged', False):
            return
        self._runtime_eye_stats_logged = True

        def _stats(label, frame):
            try:
                if hasattr(frame, 'detach'):
                    tensor = frame.detach().float()
                    return (
                        f"{label}: shape={tuple(frame.shape)} dtype={getattr(frame, 'dtype', 'unknown')} "
                        f"device={getattr(frame, 'device', 'unknown')} "
                        f"min={float(tensor.amin().item()):.6f} "
                        f"max={float(tensor.amax().item()):.6f} "
                        f"mean={float(tensor.mean().item()):.6f}"
                    )
                arr = np.asarray(frame)
                arr_f = arr.astype(np.float32, copy=False)
                return (
                    f"{label}: shape={arr.shape} dtype={arr.dtype} device=cpu "
                    f"min={float(arr_f.min()):.6f} "
                    f"max={float(arr_f.max()):.6f} "
                    f"mean={float(arr_f.mean()):.6f}"
                )
            except Exception as exc:
                return f"{label}: stats unavailable {type(exc).__name__}: {exc}"

        debug = getattr(runtime_result, 'debug_info', {}) or {}
        output_format = getattr(runtime_result, 'output_format', None) or debug.get('runtime_output_format', 'unknown')
        output_dtype = getattr(runtime_result, 'output_dtype', None) or debug.get('runtime_output_dtype', 'unknown')
        output_eye_size = getattr(runtime_result, 'output_eye_size', None) or debug.get('runtime_output_eye_size', 'unknown')
        output_pack_backend = getattr(runtime_result, 'output_pack_backend', None) or debug.get('runtime_output_pack_backend', 'unknown')
        print(
            "[OpenXRViewer] runtime eye stats:"
            f" upload={upload_path}"
            f" format={output_format}"
            f" runtime_dtype={output_dtype}"
            f" eye_size={output_eye_size}"
            f" pack={output_pack_backend}"
            f" left=({_stats('left', runtime_result.left_eye)})"
            f" right=({_stats('right', runtime_result.right_eye)})",
            flush=True,
        )

    def _verify_runtime_eye_gpu_upload_once(self, eyes, w, h):
        if getattr(self, '_runtime_eye_gpu_verify_done', False):
            return True
        self._runtime_eye_gpu_verify_done = True
        try:
            source_mean = self._runtime_eye_source_mean(eyes[0])
            tex = self._runtime_eye_textures[0]
            if tex is None:
                return True
            data = tex.read()
            if not data:
                return True
            arr = np.frombuffer(data, dtype=np.uint8)
            if arr.size <= 0:
                return True
            texture_mean = float(arr.mean()) / 255.0
            texture_max = int(arr.max())
            print(
                "[OpenXRViewer] runtime eye GPU upload verify:"
                f" source_mean={source_mean if source_mean is not None else 'unknown'}"
                f" texture_mean={texture_mean:.6f}"
                f" texture_max={texture_max}"
                f" size={w}x{h}",
                flush=True,
            )
            if source_mean is not None and source_mean > 0.02 and texture_mean < 0.002 and texture_max <= 2:
                self._runtime_eye_gpu_enabled = False
                self._runtime_eye_gpu_disabled_reason = (
                    f"runtime eye GPU upload produced black GL texture "
                    f"(source_mean={source_mean:.6f}, texture_mean={texture_mean:.6f}, texture_max={texture_max})"
                )
                try:
                    self._release_runtime_eye_texture_resources()
                    self._release_runtime_eye_pbos()
                except Exception:
                    pass
                print(
                    "[OpenXRViewer] runtime OpenGL GPU upload disabled:"
                    f" {self._runtime_eye_gpu_disabled_reason}; falling back to CPU GL upload",
                    flush=True,
                )
                return False
        except Exception as exc:
            print(
                f"[OpenXRViewer] runtime eye GPU upload verify skipped: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return True

    def _log_runtime_eye_difference_once(self, left_eye, right_eye):
        if getattr(self, '_runtime_eye_diff_logged', False):
            return
        try:
            import torch
            if hasattr(left_eye, 'detach') and hasattr(right_eye, 'detach'):
                left = left_eye.detach().float()
                right = right_eye.detach().float()
                if left.shape != right.shape:
                    print(f"[OpenXRViewer] runtime eye diff unavailable: shape mismatch left={tuple(left.shape)} right={tuple(right.shape)}")
                    self._runtime_eye_diff_logged = True
                    return
                if left.numel() == 0:
                    return
                diff = (left - right).abs()
                mean_diff = float(diff.mean().item())
                max_diff = float(diff.max().item())
                scale = 255.0 if max(float(left.max().item()), float(right.max().item())) <= 1.0 else 1.0
                print(
                    f"[OpenXRViewer] runtime eye diff mean={mean_diff * scale:.3f}/255 "
                    f"max={max_diff * scale:.3f}/255 shape={tuple(left.shape)}"
                )
                self._runtime_eye_diff_logged = True
                return
        except Exception as exc:
            print(f"[OpenXRViewer] runtime eye diff unavailable: {type(exc).__name__}: {exc}")
            self._runtime_eye_diff_logged = True
            return

        try:
            left = self._runtime_eye_to_numpy(left_eye).astype(np.int16)
            right = self._runtime_eye_to_numpy(right_eye).astype(np.int16)
            if left.shape != right.shape:
                print(f"[OpenXRViewer] runtime eye diff unavailable: shape mismatch left={left.shape} right={right.shape}")
            else:
                diff = np.abs(left - right)
                print(
                    f"[OpenXRViewer] runtime eye diff mean={float(diff.mean()):.3f}/255 "
                    f"max={int(diff.max())}/255 shape={left.shape}"
                )
        except Exception as exc:
            print(f"[OpenXRViewer] runtime eye diff unavailable: {type(exc).__name__}: {exc}")
        self._runtime_eye_diff_logged = True

    def _runtime_eye_to_numpy(self, frame):
        import torch

        if hasattr(frame, "detach"):
            tensor = frame.detach()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
                tensor = tensor[:3].permute(1, 2, 0)
            elif tensor.ndim == 3 and tensor.shape[-1] >= 3:
                tensor = tensor[..., :3]
            else:
                raise RuntimeError(f"Unsupported OpenXR runtime eye shape: {tuple(tensor.shape)}")
            if tensor.is_floating_point():
                tensor = tensor.clamp(0.0, 1.0).mul(255.0)
            return tensor.contiguous().to(torch.uint8).cpu().numpy()

        arr = np.asarray(frame)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.transpose(arr[:3], (1, 2, 0))
        elif arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            raise RuntimeError(f"Unsupported OpenXR runtime eye shape: {tuple(arr.shape)}")
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        return arr.astype(np.uint8, copy=False)

    def _release_runtime_eye_textures(self):
        self._release_runtime_eye_texture_resources()
        for idx, tex in enumerate(self._runtime_eye_textures):
            if tex is not None:
                try:
                    tex.release()
                except Exception:
                    pass
                self._runtime_eye_textures[idx] = None
        if self._runtime_depth_texture is not None:
            try:
                self._runtime_depth_texture.release()
            except Exception:
                pass
            self._runtime_depth_texture = None
        self._runtime_eye_texture_size = None

    def _release_runtime_eye_texture_resources(self):
        if any(self._runtime_eye_texture_resources) and self._cuda_gl:
            for resource in self._runtime_eye_texture_resources:
                if resource is None:
                    continue
                try:
                    self._cuda_gl.unregister_resource(resource)
                except Exception:
                    pass
        self._runtime_eye_texture_resources = [None, None]
        self._runtime_eye_texture_resource_size = None

    def _ensure_runtime_eye_texture_resources(self, w, h):
        if not self._cuda_gl or BACKEND != "CUDA":
            return False
        if not hasattr(self._cuda_gl, "register_image"):
            return False
        if self._runtime_eye_texture_resource_size == (w, h) and all(self._runtime_eye_texture_resources):
            return True
        self._release_runtime_eye_texture_resources()
        self._runtime_eye_texture_resources = [
            self._cuda_gl.register_image(self._runtime_eye_textures[0].glo, GL_TEXTURE_2D),
            self._cuda_gl.register_image(self._runtime_eye_textures[1].glo, GL_TEXTURE_2D),
        ]
        self._runtime_eye_texture_resource_size = (w, h)
        if self._openxr_debug:
            print(f"[OpenXRViewer] runtime eye CUDA/GL texture resources registered {w}x{h}")
        return True

    def _release_runtime_eye_pbos(self):
        if any(self._runtime_eye_cuda_resources) and self._cuda_gl:
            for resource in self._runtime_eye_cuda_resources:
                if resource is None:
                    continue
                try:
                    self._cuda_gl.unregister_resource(resource)
                except Exception:
                    pass
        ids = [int(pbo) for pbo in self._runtime_eye_pbos if pbo is not None]
        if ids:
            try:
                glDeleteBuffers(len(ids), ids)
            except Exception:
                pass
        self._runtime_eye_pbos = [None, None]
        self._runtime_eye_cuda_resources = [None, None]
        self._runtime_eye_pbo_size = None
        self._runtime_eye_pbo_nbytes = 0

    def _ensure_runtime_eye_pbos(self, w, h):
        if not self._cuda_gl or BACKEND not in ("CUDA", "HIP"):
            return False
        nbytes = int(w) * int(h) * 3
        if self._runtime_eye_pbo_size == (w, h) and all(self._runtime_eye_pbos):
            return True
        self._release_runtime_eye_pbos()
        ids = glGenBuffers(2)
        self._runtime_eye_pbos = [int(ids[0]), int(ids[1])]
        for pbo_id in self._runtime_eye_pbos:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self._runtime_eye_cuda_resources = [
            self._cuda_gl.register_buffer(self._runtime_eye_pbos[0]),
            self._cuda_gl.register_buffer(self._runtime_eye_pbos[1]),
        ]
        self._runtime_eye_pbo_size = (w, h)
        self._runtime_eye_pbo_nbytes = nbytes
        if self._openxr_debug:
            print(f"[OpenXRViewer] runtime eye GPU PBOs created ({BACKEND}) {w}x{h}")
        return True

    def _ensure_runtime_eye_textures(self, w, h):
        if (
            self._runtime_eye_texture_size == (w, h)
            and all(self._runtime_eye_textures)
            and self._runtime_depth_texture is not None
        ):
            return
        self._release_runtime_eye_textures()
        for idx in range(2):
            tex = self.ctx.texture((w, h), 3, dtype='f1')
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.build_mipmaps()
            glBindTexture(GL_TEXTURE_2D, tex.glo)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.35)
            glBindTexture(GL_TEXTURE_2D, 0)
            self._runtime_eye_textures[idx] = tex
        self._runtime_depth_texture = self.ctx.texture((w, h), 1, dtype='f4')
        self._runtime_depth_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._runtime_depth_texture.write(np.zeros((h, w), dtype=np.float32).tobytes())
        self._runtime_eye_texture_size = (w, h)

    def _runtime_eye_tensor_hwc_u8(self, torch_module, frame):
        tensor = frame.detach()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise RuntimeError(f"Unsupported OpenXR runtime eye batch: {tuple(tensor.shape)}")
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
            tensor = tensor[:3].permute(1, 2, 0)
        elif tensor.ndim == 3 and tensor.shape[-1] >= 3:
            tensor = tensor[..., :3]
        else:
            raise RuntimeError(f"Unsupported OpenXR runtime eye shape: {tuple(tensor.shape)}")
        if tensor.is_floating_point():
            max_value = float(tensor.detach().amax().item()) if tensor.numel() else 0.0
            if max_value <= 1.5:
                tensor = tensor * 255.0
        return tensor.contiguous().clamp(0, 255).to(torch_module.uint8)

    def _runtime_eye_shape_hw(self, frame):
        shape = tuple(frame.shape) if hasattr(frame, 'shape') else tuple(np.asarray(frame).shape)
        if len(shape) == 4:
            if shape[0] != 1:
                raise RuntimeError(f"Unsupported OpenXR runtime eye batch: {shape}")
            shape = shape[1:]
        if len(shape) == 3 and shape[0] in (3, 4):
            return int(shape[1]), int(shape[2])
        if len(shape) == 3 and shape[-1] >= 3:
            return int(shape[0]), int(shape[1])
        raise RuntimeError(f"Unsupported OpenXR runtime eye shape: {shape}")

    def _try_update_runtime_frame_gpu(self, runtime_result, w, h):
        if not self._runtime_eye_gpu_enabled:
            return False
        if self._cuda_gl is False or CUDART_GL is None or BACKEND not in ("CUDA", "HIP"):
            return False
        left_src = runtime_result.left_eye
        right_src = runtime_result.right_eye
        if not (
            hasattr(left_src, 'is_cuda') and left_src.is_cuda and
            hasattr(right_src, 'is_cuda') and right_src.is_cuda
        ):
            return False
        try:
            import torch
            if self._cuda_gl is None:
                self._cuda_gl = CUDART_GL()
            left = self._runtime_eye_tensor_hwc_u8(torch, left_src)
            right = self._runtime_eye_tensor_hwc_u8(torch, right_src)
            if left.shape[:2] != (h, w) or right.shape[:2] != (h, w):
                raise RuntimeError(f"Runtime eye tensor size changed during upload: left={tuple(left.shape)} right={tuple(right.shape)}")
            device_index = left.device.index if left.device.index is not None else 0
            torch.cuda.current_stream(device_index).synchronize()
            texture_gpu_was_enabled = self._runtime_eye_texture_gpu_enabled
            if self._try_update_runtime_frame_texture_gpu((left, right), w, h):
                return True
            if texture_gpu_was_enabled and not self._runtime_eye_texture_gpu_enabled:
                return False
            if not self._update_runtime_frame_pbo_gpu((left, right), w, h):
                return False
            return self._verify_runtime_eye_gpu_upload_once((left, right), w, h)
        except Exception as e:
            self._runtime_eye_gpu_enabled = False
            self._runtime_eye_gpu_disabled_reason = str(e)
            try:
                self._release_runtime_eye_texture_resources()
                self._release_runtime_eye_pbos()
            except Exception:
                pass
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            print(f"[OpenXRViewer] runtime OpenGL GPU upload unavailable: {e}; falling back to CPU GL upload")
            return False

    def _try_update_runtime_frame_texture_gpu(self, eyes, w, h):
        if not self._runtime_eye_texture_gpu_enabled:
            return False
        if BACKEND != "CUDA" or not hasattr(self._cuda_gl, "register_image"):
            return False
        try:
            self._ensure_runtime_eye_texture_resources(w, h)
            for idx, eye in enumerate(eyes):
                resource = self._runtime_eye_texture_resources[idx]
                # Image-registered resources (register_image) expose a CUDA
                # array, not a linear pointer.  map_resource() also calls
                # cudaGraphicsResourceGetMappedPointer, which fails with
                # cudaErrorNotMappedAsPointer ("resource not mapped as
                # pointer") on image resources.  Use the map-only helper and
                # fetch the array via cudaGraphicsSubResourceGetMappedArray.
                self._cuda_gl.map_graphics_resource(resource)
                try:
                    array = self._cuda_gl.mapped_array(resource)
                    self._cuda_gl.memcpy_2d_to_array(array, eye.data_ptr(), w * 3, w * 3, h)
                finally:
                    self._cuda_gl.unmap_resource(resource)
                glBindTexture(GL_TEXTURE_2D, self._runtime_eye_textures[idx].glo)
                glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
            if not self._runtime_eye_texture_logged:
                print(f"[OpenXRViewer] runtime_direct_opengl_texture active (CUDA/GL image) {w}x{h}")
                self._runtime_eye_texture_logged = True
            return True
        except Exception as e:
            self._release_runtime_eye_texture_resources()
            self._runtime_eye_texture_gpu_enabled = False
            self._runtime_eye_gpu_disabled_reason = str(e)
            print(f"[OpenXRViewer] runtime_direct_opengl_texture unavailable: {e}; using PBO fallback")
            return False

    def _update_runtime_frame_pbo_gpu(self, eyes, w, h):
        self._ensure_runtime_eye_pbos(w, h)
        for idx, eye in enumerate(eyes):
            ptr = self._cuda_gl.map_resource(self._runtime_eye_cuda_resources[idx])
            try:
                self._cuda_gl.memcpy_d2d(ptr, eye.data_ptr(), eye.nbytes)
            finally:
                self._cuda_gl.unmap_resource(self._runtime_eye_cuda_resources[idx])
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._runtime_eye_pbos[idx])
            glBindTexture(GL_TEXTURE_2D, self._runtime_eye_textures[idx].glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        if not self._runtime_eye_gpu_logged:
            print(f"[OpenXRViewer] runtime_direct_opengl_pbo active ({BACKEND}) {w}x{h}")
            self._runtime_eye_gpu_logged = True
        return True

    def _update_runtime_frame(self, runtime_result):
        debug_info = getattr(runtime_result, 'debug_info', {}) or {}
        output_format = getattr(runtime_result, 'output_format', None) or debug_info.get('runtime_output_format')
        if output_format == 'openxr_rgb_depth':
            self._apply_runtime_rgb_depth_config(
                debug_info,
                shader_uniforms=getattr(runtime_result, 'shader_uniforms', None),
                output_eye_size=getattr(runtime_result, 'output_eye_size', None),
            )
            source_rgb = getattr(runtime_result, 'source_rgb', None)
            if source_rgb is None:
                source_rgb = runtime_result.left_eye
            source_rgb, source_depth = self._normalize_rgb_depth_runtime_source(source_rgb, runtime_result.depth)
            self._runtime_direct_source = False
            self._update_frame(source_rgb, source_depth)
            return
        if not self._runtime_direct_enabled:
            self._runtime_direct_source = False
            return
        left_hw = self._runtime_eye_shape_hw(runtime_result.left_eye)
        right_hw = self._runtime_eye_shape_hw(runtime_result.right_eye)
        if left_hw != right_hw:
            raise RuntimeError(f"OpenXR runtime eye size mismatch: left={left_hw} right={right_hw}")
        self._log_runtime_eye_difference_once(runtime_result.left_eye, runtime_result.right_eye)
        h, w = left_hw
        if self._use_d3d11 and self._d3d11_native_renderer is not None:
            try:
                result = self._d3d11_native_renderer.update_runtime_eyes(
                    runtime_result.left_eye,
                    runtime_result.right_eye,
                )
                if result:
                    self._runtime_direct_source = True
                    self._texture_size = (w, h)
                    self.frame_size = (w, h)
                    self.screen_height = None
                    self._maybe_sample_glow_target_color(runtime_result.left_eye, hasattr(runtime_result.left_eye, 'detach'))
                    return
            except Exception as e:
                print(f"[OpenXRViewer] D3D11 runtime eye upload failed: {e}")
            try:
                self._d3d11_native_renderer.cleanup()
            except Exception:
                pass
            self._d3d11_native_renderer = None
            self._texture_size = None
        self._ensure_runtime_eye_textures(w, h)
        gpu_uploaded = self._try_update_runtime_frame_gpu(runtime_result, w, h)
        if not gpu_uploaded:
            left = self._runtime_eye_to_numpy(runtime_result.left_eye)
            right = self._runtime_eye_to_numpy(runtime_result.right_eye)
            self._runtime_eye_textures[0].write(np.ascontiguousarray(left[:, :, :3]).tobytes())
            self._runtime_eye_textures[1].write(np.ascontiguousarray(right[:, :, :3]).tobytes())
            for tex in self._runtime_eye_textures:
                glBindTexture(GL_TEXTURE_2D, tex.glo)
                glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
            if not self._runtime_eye_cpu_logged:
                print(f"[OpenXRViewer] runtime_direct_cpu_gl active {w}x{h}")
                self._runtime_eye_cpu_logged = True
            self._log_runtime_eye_stats_once(runtime_result, upload_path='cpu_gl')
        else:
            self._log_runtime_eye_stats_once(runtime_result, upload_path='gpu_gl')
        self._runtime_direct_source = True
        self._texture_size = (w, h)
        self.frame_size = (w, h)
        self.screen_height = None
        self._maybe_sample_glow_target_color(runtime_result.left_eye, hasattr(runtime_result.left_eye, 'detach'))
        if self._d3d11_native_renderer is not None:
            self._d3d11_native_renderer.has_frame = False

    def _apply_runtime_rgb_depth_config(self, debug_info, *, shader_uniforms=None, output_eye_size=None):
        uniforms = shader_uniforms
        if not isinstance(uniforms, dict):
            uniforms = debug_info.get("openxr_shader_uniforms")
        if not isinstance(uniforms, dict):
            uniforms = {}
        if "convergence" in uniforms:
            self.convergence = float(uniforms["convergence"])
        elif "openxr_convergence" in debug_info:
            self.convergence = float(debug_info["openxr_convergence"])
        if "depth_strength" in uniforms:
            self._runtime_rgb_depth_depth_strength = max(0.0, float(uniforms["depth_strength"]))
        else:
            self._runtime_rgb_depth_depth_strength = max(0.0, float(getattr(self, "depth_strength", 1.0) or 0.0))
        max_disparity_px = uniforms.get("max_disparity_px", debug_info.get("resolved_max_disparity_px", 0.0))
        self._runtime_rgb_depth_max_disparity_px = max(0.0, float(max_disparity_px or 0.0))
        render_width = _runtime_shader_render_width(uniforms.get("render_size"))
        if render_width <= 0:
            render_width = _runtime_shader_render_width(output_eye_size)
        if render_width <= 0:
            render_width = _runtime_shader_render_width(debug_info.get("runtime_output_eye_size"))
        self._runtime_rgb_depth_render_width = render_width

    def _normalize_rgb_depth_runtime_source(self, rgb, depth):
        import torch

        if isinstance(rgb, torch.Tensor):
            if rgb.ndim == 4:
                if rgb.shape[0] != 1:
                    raise RuntimeError(f"OpenXR rgb-depth batch size must be 1, got {tuple(rgb.shape)}")
                rgb = rgb[0]
            if rgb.ndim != 3:
                raise RuntimeError(f"OpenXR rgb-depth rgb must be CHW or BCHW, got {tuple(rgb.shape)}")
            if rgb.shape[0] != 3 and rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1).contiguous()
            if rgb.is_floating_point():
                rgb = rgb * 255.0
        if isinstance(depth, torch.Tensor):
            if depth.ndim == 4:
                if depth.shape[0] != 1 or depth.shape[1] != 1:
                    raise RuntimeError(f"OpenXR rgb-depth depth must be B1HW, got {tuple(depth.shape)}")
                depth = depth[0, 0]
            elif depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            if depth.ndim != 2:
                raise RuntimeError(f"OpenXR rgb-depth depth must be HW, got {tuple(depth.shape)}")
        return rgb, depth
