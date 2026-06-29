import ctypes
import os
import time

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
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_LOD_BIAS,
    GL_UNSIGNED_BYTE,
)
from viewer.viewer import BACKEND
from viewer.gl_texture_uploader import CudaGlTextureUploader
from utils.cpu_warnings import describe_tensor, warn_cpu_fallback, warn_cpu_transfer

try:
    from viewer.viewer import CUDART_GL
except ImportError:
    CUDART_GL = None


def _openxr_runtime_eye_sync_enabled() -> bool:
    return str(os.environ.get("D2S_OPENXR_RUNTIME_EYE_SYNC", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
                warn_cpu_fallback(
                    "OpenXR runtime eye GPU verify",
                    "black_gl_texture",
                    detail=self._runtime_eye_gpu_disabled_reason,
                    key="openxr_runtime_eye_verify_black",
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
            warn_cpu_transfer(
                "OpenXR runtime eye",
                ".cpu().numpy()",
                detail=describe_tensor(tensor),
                key="openxr_runtime_eye_to_numpy",
            )
            return tensor.contiguous().to(torch.uint8).cpu().numpy()

        warn_cpu_transfer(
            "OpenXR runtime eye",
            "numpy input path",
            detail=f"type={type(frame).__name__}",
            key="openxr_runtime_eye_numpy_input",
        )
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
        uploader = getattr(self, "_runtime_eye_texture_uploader", None)
        if uploader is not None:
            uploader.release_images()
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
        uploader = getattr(self, "_runtime_eye_texture_uploader", None)
        if uploader is not None:
            uploader.release_pbos()
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
        components = int(getattr(self, '_runtime_eye_texture_components', 3) or 3)
        for idx in range(2):
            tex = self.ctx.texture((w, h), components, dtype='f1')
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

    def _runtime_eye_tensor_rgba_u8(self, torch_module, frame):
        tensor = frame.detach()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise RuntimeError(f"Unsupported OpenXR runtime eye batch: {tuple(tensor.shape)}")
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] not in (3, 4) and tensor.shape[-1] == 4 and tensor.dtype == torch_module.uint8:
            return tensor.contiguous()
        tensor = self._runtime_eye_tensor_hwc_u8(torch_module, frame)
        h, w = tensor.shape[:2]
        rgba = torch_module.empty((h, w, 4), device=tensor.device, dtype=torch_module.uint8)
        rgba[..., :3].copy_(tensor[..., :3])
        rgba[..., 3].fill_(255)
        return rgba

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
            warn_cpu_fallback(
                "OpenXR runtime eye GPU upload",
                "disabled",
                detail=getattr(self, '_runtime_eye_gpu_disabled_reason', None),
                key="openxr_runtime_eye_gpu_disabled",
            )
            return False
        if self._cuda_gl is False:
            warn_cpu_fallback(
                "OpenXR runtime eye GPU upload",
                "cuda_gl_disabled",
                detail=getattr(self, '_runtime_eye_gpu_disabled_reason', None),
                key="openxr_runtime_eye_cuda_gl_disabled",
            )
            return False
        if CUDART_GL is None:
            warn_cpu_fallback(
                "OpenXR runtime eye GPU upload",
                "cudart_gl_missing",
                key="openxr_runtime_eye_cudart_gl_missing",
            )
            return False
        if BACKEND not in ("CUDA", "HIP"):
            warn_cpu_fallback(
                "OpenXR runtime eye GPU upload",
                "backend_not_cuda_or_hip",
                detail=f"backend={BACKEND}",
                key="openxr_runtime_eye_backend_not_gpu",
            )
            return False
        left_src = runtime_result.left_eye
        right_src = runtime_result.right_eye
        if not (
            hasattr(left_src, 'is_cuda') and left_src.is_cuda and
            hasattr(right_src, 'is_cuda') and right_src.is_cuda
        ):
            warn_cpu_fallback(
                "OpenXR runtime eye GPU upload",
                "runtime_eye_not_cuda",
                detail=f"left=({describe_tensor(left_src)}) right=({describe_tensor(right_src)})",
                key="openxr_runtime_eye_not_cuda",
            )
            return False
        try:
            total_start = time.perf_counter()
            import torch
            if self._cuda_gl is None:
                self._cuda_gl = CUDART_GL()
            if getattr(self, "_runtime_eye_texture_uploader", None) is None:
                self._runtime_eye_texture_uploader = CudaGlTextureUploader(
                    self._cuda_gl,
                    backend=BACKEND,
                    debug=getattr(self, "_openxr_debug", False),
                    log_prefix="OpenXRViewer",
                )
            tensor_start = time.perf_counter()
            left_rgba = self._runtime_eye_tensor_rgba_u8(torch, left_src)
            right_rgba = self._runtime_eye_tensor_rgba_u8(torch, right_src)
            if left_rgba.shape[:2] != (h, w) or right_rgba.shape[:2] != (h, w):
                raise RuntimeError(f"Runtime eye tensor size changed during upload: left={tuple(left_rgba.shape)} right={tuple(right_rgba.shape)}")
            device_index = left_rgba.device.index if left_rgba.device.index is not None else 0
            self._breakdown_add_time("runtime_eye_tensor", time.perf_counter() - tensor_start)
            sync_start = time.perf_counter()
            if _openxr_runtime_eye_sync_enabled():
                torch.cuda.current_stream(device_index).synchronize()
            else:
                ready_event = getattr(runtime_result, "cuda_ready_event", None)
                if ready_event is not None:
                    ready_event.synchronize()
            self._breakdown_add_time("runtime_eye_sync", time.perf_counter() - sync_start)
            image_start = time.perf_counter()
            upload_ok = self._try_update_runtime_frame_texture_gpu((left_rgba, right_rgba), w, h)
            self._breakdown_add_time("runtime_eye_image", time.perf_counter() - image_start)
            if not upload_ok:
                warn_cpu_fallback(
                    "OpenXR runtime eye GPU upload",
                    "pbo_upload_unavailable",
                    detail=getattr(self, '_runtime_eye_gpu_disabled_reason', None),
                    key="openxr_runtime_eye_pbo_unavailable",
                )
                return False
            verify_start = time.perf_counter()
            verified = self._verify_runtime_eye_gpu_upload_once((left_rgba, right_rgba), w, h)
            self._breakdown_add_time("runtime_eye_verify", time.perf_counter() - verify_start)
            self._breakdown_add_time("runtime_eye_total", time.perf_counter() - total_start)
            return verified
        except Exception as e:
            self._runtime_eye_gpu_enabled = False
            self._runtime_eye_gpu_disabled_reason = str(e)
            try:
                self._release_runtime_eye_texture_resources()
                self._release_runtime_eye_pbos()
            except Exception:
                pass
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            warn_cpu_fallback(
                "OpenXR runtime eye GPU upload",
                "exception",
                detail=str(e),
                key="openxr_runtime_eye_gpu_exception",
            )
            return False

    def _try_update_runtime_frame_texture_gpu(self, eyes, w, h):
        try:
            uploader = getattr(self, "_runtime_eye_texture_uploader", None)
            if uploader is None:
                raise RuntimeError("runtime eye texture uploader not initialized")
            upload_path = uploader.upload_rgba(
                self._runtime_eye_textures,
                eyes,
                w,
                h,
                prefer_image=self._runtime_eye_texture_gpu_enabled,
            )
            if upload_path == "image" and not self._runtime_eye_texture_logged:
                print(f"[OpenXRViewer] runtime_direct_opengl_texture active (CUDA/GL image RGBA) {w}x{h}")
                self._runtime_eye_texture_logged = True
            if upload_path == "pbo":
                self._runtime_eye_gpu_disabled_reason = uploader.image_failed_reason
                if not self._runtime_eye_gpu_logged:
                    print(
                        f"[OpenXRViewer] runtime_direct_opengl_pbo active ({BACKEND}) {w}x{h} texture_image=fallback",
                        flush=True,
                    )
                    self._runtime_eye_gpu_logged = True
            return True
        except Exception as e:
            self._release_runtime_eye_texture_resources()
            self._runtime_eye_texture_gpu_enabled = False
            self._runtime_eye_gpu_disabled_reason = str(e)
            print(f"[OpenXRViewer] runtime_direct_opengl_texture unavailable: {e}; using PBO fallback")
            return False

    def _update_runtime_frame_pbo_gpu(self, eyes, w, h):
        if not self._ensure_runtime_eye_pbos(w, h):
            return False
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
            texture_image_status = "fallback" if getattr(self, '_runtime_eye_gpu_disabled_reason', None) else "disabled"
            print(
                f"[OpenXRViewer] runtime_direct_opengl_pbo active ({BACKEND}) {w}x{h} "
                f"texture_image={texture_image_status}",
                flush=True,
            )
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
            warn_cpu_fallback(
                "OpenXR runtime eye upload",
                "gpu_upload_returned_false",
                detail=f"size={w}x{h}",
                key="openxr_runtime_eye_cpu_gl",
            )
            left = self._runtime_eye_to_numpy(runtime_result.left_eye)
            right = self._runtime_eye_to_numpy(runtime_result.right_eye)
            if int(getattr(self, '_runtime_eye_texture_components', 3) or 3) == 4:
                alpha_l = np.full((h, w, 1), 255, dtype=np.uint8)
                alpha_r = np.full((h, w, 1), 255, dtype=np.uint8)
                left_upload = np.concatenate((left[:, :, :3], alpha_l), axis=2)
                right_upload = np.concatenate((right[:, :, :3], alpha_r), axis=2)
            else:
                left_upload = left[:, :, :3]
                right_upload = right[:, :, :3]
            self._runtime_eye_textures[0].write(np.ascontiguousarray(left_upload).tobytes())
            self._runtime_eye_textures[1].write(np.ascontiguousarray(right_upload).tobytes())
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
