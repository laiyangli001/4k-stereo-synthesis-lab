import time

from .projection_layer_presenter import ProjectionLayerPresenter
from .screen_layer_presenter import ScreenLayerPresenter
from .overlay_quad_presenter import QuadOverlayPresenter
from .openxr_panda_frame_state import update_panda_frame_state_from_viewer
from .view_pose_tracker import ViewPoseTracker


class OpenXRFrameRenderer:
    def __init__(self, viewer):
        self.viewer = viewer
        self.view_tracker = ViewPoseTracker(viewer)
        self.screen_presenter = ScreenLayerPresenter(viewer)
        viewer._screen_layer_presenter = self.screen_presenter
        self.projection_presenter = ProjectionLayerPresenter(viewer)
        self.overlay_quad_presenter = QuadOverlayPresenter(viewer)

    def render_frame(self, *, composition_layers, display_time, default_fov, default_proj, default_proj_d3d):
        viewer = self.viewer
        # Active sessions can start directly without the preview-only initialization path.
        ensure_env = getattr(viewer, "_ensure_env_model_initialized", None)
        if callable(ensure_env):
            ensure_env("Active")
        screen_frame_uploaded = self.screen_presenter.poll_screen_frame()
        views, view_pose_adjusted = self.view_tracker.locate_views(display_time=display_time)
        update_panda_frame_state_from_viewer(
            viewer,
            predicted_display_time=display_time,
            views=views,
            screen_frame_uploaded=screen_frame_uploaded,
        )

        quad_update_start = time.perf_counter()
        _quad_layers, quad_layer_headers, updated_quad_eyes, render_projection_layer, background_layer_headers = (
            self.screen_presenter.prepare_frame_layers(screen_frame_uploaded=screen_frame_uploaded)
        )
        viewer._breakdown_add_time('openxr_quad_update', time.perf_counter() - quad_update_start)

        try:
            eye_layer_views = self.projection_presenter.render_projection(
                enabled=render_projection_layer,
                views=views,
                default_fov=default_fov,
                default_proj=default_proj,
                default_proj_d3d=default_proj_d3d,
                updated_quad_eyes=updated_quad_eyes,
            )
        except Exception as exc:
            self._log_projection_render_failed(exc)
            viewer._breakdown_inc('openxr_projection_render_failed')
            eye_layer_views = []
        if eye_layer_views:
            overlay_quad_headers = self.overlay_quad_presenter.prepare_layers()
            if overlay_quad_headers:
                quad_layer_headers = tuple(quad_layer_headers) + tuple(overlay_quad_headers)
        self.screen_presenter.append_frame_layers(
            composition_layers,
            projection_views=eye_layer_views,
            projection_space=viewer._xr_space,
            quad_layer_headers=quad_layer_headers,
            background_layer_headers=background_layer_headers,
        )
        return screen_frame_uploaded, view_pose_adjusted, bool(eye_layer_views)

    def _log_projection_render_failed(self, exc):
        viewer = self.viewer
        now = time.perf_counter()
        key = (type(exc).__name__, str(exc))
        last_key = getattr(viewer, '_projection_render_error_log_key', None)
        next_log = float(getattr(viewer, '_projection_render_error_next_log_time', 0.0) or 0.0)
        if key == last_key and now < next_log:
            return
        viewer._projection_render_error_log_key = key
        viewer._projection_render_error_next_log_time = now + 2.0
        print(f"[OpenXRViewer] Projection layer render failed: {type(exc).__name__}: {exc}", flush=True)
