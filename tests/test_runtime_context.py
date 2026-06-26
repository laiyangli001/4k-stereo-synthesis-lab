import queue
from types import SimpleNamespace

from app_runtime.runtime_context import build_capture_callbacks, build_runtime_pipeline_context, env_flag, initial_stereo_preset_state
from stereo_runtime.render_size import RenderSizeConfig


def test_env_flag_accepts_common_truthy_values(monkeypatch):
    monkeypatch.setenv("D2S_TEST_FLAG", "yes")

    assert env_flag("D2S_TEST_FLAG") is True
    assert env_flag("D2S_MISSING_FLAG", "0") is False


def test_initial_stereo_preset_state_normalizes_auto_preset():
    config = SimpleNamespace(stereo_preset="auto", mode="fast")

    assert initial_stereo_preset_state(config) == (False, "cinema")


def test_initial_stereo_preset_state_falls_back_to_runtime_mode():
    config = SimpleNamespace(stereo_preset=None, mode="cinema")

    assert initial_stereo_preset_state(config) == (False, "cinema")


def test_build_capture_callbacks_wires_raw_queue_clear():
    raw_q = queue.Queue()
    calls = []
    shutdown = SimpleNamespace(is_set=lambda: False)

    callbacks = build_capture_callbacks(
        raw_q=raw_q,
        shutdown_event=shutdown,
        queue_clear=lambda q: calls.append(("clear", q)),
        inc_source_stat=lambda *args, **kwargs: None,
        inc_breakdown=lambda *args, **kwargs: None,
        put_raw_latest=lambda item: None,
        is_paused=lambda: False,
        is_hard_idle=lambda: False,
        on_session_update=lambda session, control: None,
        on_tick=lambda: None,
    )

    callbacks.clear_raw_queue()

    assert calls == [("clear", raw_q)]
    assert callbacks.is_shutdown() is False


def test_build_runtime_pipeline_context_uses_app_context_queues():
    raw_q = queue.Queue()
    runtime_q = queue.Queue()
    settings_update_q = queue.Queue()
    render_size_config = RenderSizeConfig()
    app_context = SimpleNamespace(
        raw_q=raw_q,
        runtime_q=runtime_q,
        settings_update_q=settings_update_q,
        time_sleep=0.01,
        openxr_runtime_direct=False,
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=object(),
        render_size_config=render_size_config,
    )
    shutdown = SimpleNamespace(is_set=lambda: False)

    context = build_runtime_pipeline_context(
        shutdown_event=shutdown,
        app_context=app_context,
        run_mode="Viewer",
        device="cpu",
        capture_frame_to_rgb=lambda *args, **kwargs: None,
        prepare_rgb_for_stereo_runtime=lambda *args, **kwargs: None,
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=lambda *args, **kwargs: None,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda *args, **kwargs: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, item: item,
        queue_put_latest=lambda q, item: None,
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    assert context.raw_q is raw_q
    assert context.runtime_q is runtime_q
    assert context.settings_update_q is settings_update_q
    assert context.render_size_config is render_size_config
    assert context.run_mode == "Viewer"
    assert context.device == "cpu"
