# Desktop2Stereo OpenXR viewer: D3D11-backed OpenXR session creation.

import ctypes

import numpy as np

try:
    import xr
except ImportError:
    xr = None

from .d3d_interop import (
    _D3D11_PREFERRED_FORMATS,
    _DXGI_FORMAT_B8G8R8A8_UNORM,
    _DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    _create_d3d11_device,
)
from .implementation_support import (
    _openxr_app_api_version,
    _openxr_optional_extensions,
    _request_openxr_display_refresh_rate,
)


class CoreOpenXRD3D11Mixin:
    """D3D11-backed OpenXR session and swapchain creation."""

    def _init_openxr_d3d11(self, quiet=False):
        """Create or resume an OpenXR session backed by a D3D11 device.

        The native path renders the virtual screen into Projection layer
        D3D11 swapchains. Quad layers are reserved for overlays/effects.
        """
        if self._xr_backend not in (None, 'd3d11'):
            raise RuntimeError(f"OpenXR backend mismatch: {self._xr_backend}")
        if self._xr_instance is None:
            app_info = xr.ApplicationInfo(
                application_name="Desktop2Stereo",
                application_version=1,
                engine_name="D2S",
                engine_version=1,
                api_version=_openxr_app_api_version(),
            )
            enabled_extensions = [xr.KHR_D3D11_ENABLE_EXTENSION_NAME]
            enabled_extensions += _openxr_optional_extensions(
                getattr(xr, 'KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME', None),
                getattr(xr, 'FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME', None),
            )
            self._openxr_equirect_background_supported = (
                getattr(xr, 'KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME', None) in enabled_extensions
                and hasattr(xr, 'CompositionLayerEquirect2KHR')
            )
            create_info = xr.InstanceCreateInfo(
                application_info=app_info,
                enabled_extension_names=enabled_extensions,
            )
            self._xr_instance = xr.create_instance(create_info)
            self._xr_backend = 'd3d11'
            if not quiet:
                print("[OpenXRViewer] XrInstance created (D3D11)")

        if self._xr_system_id is None:
            self._xr_system_id = xr.get_system(
                self._xr_instance,
                xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
            )

        if self._xr_session is not None:
            return

        # 3. Query D3D11 requirements (runtime mandates this call before session creation)
        _pfn = ctypes.cast(
            xr.get_instance_proc_addr(self._xr_instance, "xrGetD3D11GraphicsRequirementsKHR"),
            xr.PFN_xrGetD3D11GraphicsRequirementsKHR,
        )
        # Python 3.12 ctypes rejects int where a Structure field is expected.
        # pyopenxr's GraphicsRequirementsD3D11KHR.__init__ defaults adapter_luid=0
        # which triggers TypeError. Pass an explicit zeroed _LUID() instance instead.
        from xr.platform.windows import _LUID as _XrLUID
        _reqs = xr.GraphicsRequirementsD3D11KHR(adapter_luid=_XrLUID())
        xr.check_result(xr.Result(_pfn(self._xr_instance, self._xr_system_id, ctypes.byref(_reqs))))
        print(f"[OpenXRViewer] D3D11 min feature level: 0x{_reqs.min_feature_level:04x}")

        # 4. Create D3D11 device on the adapter the runtime requires
        device, context, feat = _create_d3d11_device(adapter_luid=_reqs.adapter_luid)
        self._d3d11_device  = device
        self._d3d11_context = context
        self._use_d3d11 = True
        if not quiet:
            print(f"[OpenXRViewer] D3D11 device created (feature level 0x{feat:04x})")

        # 5. Graphics binding
        binding = xr.GraphicsBindingD3D11KHR(
            device=ctypes.cast(device, ctypes.POINTER(ctypes.c_int)),
        )

        # 6. Session
        session_info = xr.SessionCreateInfo(
            system_id=self._xr_system_id,
            next=binding,
        )
        self._xr_session = xr.create_session(self._xr_instance, session_info)
        if not quiet:
            print("[OpenXRViewer] XrSession created (D3D11)")
        _request_openxr_display_refresh_rate(self._xr_session, quiet=quiet)

        # 7. Reference space
        available_spaces = xr.enumerate_reference_spaces(self._xr_session)
        ref_type = (
            xr.ReferenceSpaceType.STAGE
            if xr.ReferenceSpaceType.STAGE in available_spaces
            else xr.ReferenceSpaceType.LOCAL
        )
        self._xr_ref_space_type = ref_type
        self._xr_space = xr.create_reference_space(
            self._xr_session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=ref_type,
                pose_in_reference_space=xr.Posef(),
            ),
        )
        self._xr_space_pose_in_ref = np.eye(4, dtype=np.float32)

        # 8. Swapchains with DXGI format
        view_configs = xr.enumerate_view_configuration_views(
            self._xr_instance,
            self._xr_system_id,
            xr.ViewConfigurationType.PRIMARY_STEREO,
        )
        # Pick the best supported DXGI format
        runtime_fmts = list(xr.enumerate_swapchain_formats(self._xr_session))
        self._projection_view_configs = view_configs
        self._projection_runtime_formats = runtime_fmts
        print(f"[OpenXRViewer] D3D11 runtime swapchain formats: {runtime_fmts}")
        chosen_fmt = None
        for preferred in _D3D11_PREFERRED_FORMATS:
            if preferred in runtime_fmts:
                chosen_fmt = preferred
                break
        if chosen_fmt is None:
            raise RuntimeError(f"No supported D3D11 swapchain format. Runtime offers: {runtime_fmts}")
        self._d3d11_swapchain_fmt = chosen_fmt
        self._swapchain_is_bgra = chosen_fmt in (
            _DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, _DXGI_FORMAT_B8G8R8A8_UNORM,
        )
        print(f"[OpenXRViewer] D3D11 swapchain format: {chosen_fmt}"
            f"{' (BGRA)' if self._swapchain_is_bgra else ''}")

        for eye_index, vcv in enumerate(view_configs):
            rec_w = vcv.recommended_image_rect_width
            rec_h = vcv.recommended_image_rect_height
            scale = float(getattr(self, '_xr_render_scale', 1.0))
            max_w = int(getattr(vcv, 'max_image_rect_width', rec_w) or rec_w)
            max_h = int(getattr(vcv, 'max_image_rect_height', rec_h) or rec_h)
            sc_w  = min(max_w, max(16, int(rec_w * scale))) & ~1
            sc_h  = min(max_h, max(16, int(rec_h * scale))) & ~1
            print(f"[OpenXRViewer] Eye {eye_index} swapchain: {sc_w}x{sc_h} (D3D11, scale={scale:.2f})")

            sc_info = xr.SwapchainCreateInfo(
                usage_flags=(
                    xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT |
                    xr.SwapchainUsageFlags.SAMPLED_BIT
                ),
                format=chosen_fmt,
                sample_count=1,
                width=sc_w,
                height=sc_h,
                face_count=1,
                array_size=1,
                mip_count=1,
            )
            swapchain = xr.create_swapchain(self._xr_session, sc_info)
            images    = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageD3D11KHR)
            self._xr_swapchains[eye_index]    = swapchain
            self._swapchain_images[eye_index] = images
            self._swapchain_sizes[eye_index]  = (sc_w, sc_h)

        # 9. Resolve the future glTF scene-renderer selector without changing
        # the default native path before the Panda3D OpenXR swapchain gate passes.
        try:
            from .panda_runtime.runtime import log_renderer_selection, resolve_gltf_renderer_mode
            self._gltf_renderer_config = resolve_gltf_renderer_mode()
            if self._gltf_renderer_config.requested_mode != 'native' or self._gltf_renderer_config.fallback_reason:
                log_renderer_selection(self._gltf_renderer_config)
        except Exception as e:
            self._gltf_renderer_config = None
            print(f"[OpenXRViewer] glTF renderer selector unavailable: {e}")

        # 10. Prefer native D3D11 for runtime eye/RGB+depth -> Projection layer
        # upload. OpenGL/NV_DX interop remains a compatibility fallback only.
        if self._d3d11_native_requested:
            try:
                from .d3d11_native_renderer import D3D11NativeRenderer
                self._d3d11_native_renderer = D3D11NativeRenderer(
                    self._d3d11_device, self._d3d11_context, self._d3d11_swapchain_fmt
                )
                self._interop_mode = None
                print("[OpenXRViewer] D3D11 native renderer active")
            except Exception as e:
                self._d3d11_native_renderer = None
                print(f"[OpenXRViewer] D3D11 native renderer unavailable: {e}")

        if view_configs and self._d3d11_native_renderer is not None:
            src_w, src_h = self.frame_size
            max_w = max(int(getattr(v, 'max_image_rect_width', src_w) or src_w) for v in view_configs)
            max_h = max(int(getattr(v, 'max_image_rect_height', src_h) or src_h) for v in view_configs)
            self._quad_swapchain_format = chosen_fmt
            self._quad_swapchain_formats = (chosen_fmt,)
            self._quad_swapchain_image_type = xr.SwapchainImageD3D11KHR
            self._quad_swapchain_max_size = (max_w, max_h)
            self._quad_swapchain_presented_eyes = set()
            print(f"[OpenXRViewer] Quad layer D3D11 lazy swapchains armed max={max_w}x{max_h}")

        # 11. Try NV_DX interop for projection overlays when native D3D11
        # rendering is not available. Phase-0 Panda3D migration probes may also
        # request it explicitly to validate real OpenXR swapchain registration.
        if self._panda3d_phase0_swapchain_probe_enabled:
            print("[OpenXRViewer] Panda3D Phase-0 swapchain probe enabled")
            self._setup_gpu_interop_d3d11()
        elif self._d3d11_native_renderer is None:
            self._setup_gpu_interop_d3d11()

        # 12. Controller actions (best-effort)
        try:
            if self._action_set is None:
                self._init_controller_actions()
            else:
                self._attach_controller_actions_to_session()
        except Exception as e:
            print(f"[OpenXRViewer] Controller actions unavailable: {e}")
