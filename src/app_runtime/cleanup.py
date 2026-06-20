from __future__ import annotations

import subprocess


def stop_processes(global_processes: dict) -> None:
    for proc_name, process in global_processes.items():
        if process and hasattr(process, "poll"):
            try:
                print(f"[Cleanup] Stopping {proc_name}...")
                process.terminate()
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    print(f"[Cleanup] Force killing {proc_name}...")
                    process.kill()
                    process.wait()
                except Exception:
                    pass
            except Exception as exc:
                print(f"[Cleanup] Error stopping {proc_name}: {exc}")
            finally:
                global_processes[proc_name] = None


def clear_queues(queues, *, timeout: float) -> None:
    for q in queues:
        while not q.empty():
            try:
                q.get(timeout=timeout)
            except Exception:
                pass


def cleanup_resources(
    *,
    global_processes: dict,
    stop_capture,
    streamer=None,
    queues=(),
    queue_timeout: float,
    rtmp_thread=None,
) -> None:
    print("[Cleanup] Shutting down all resources...")

    stop_processes(global_processes)

    try:
        if stop_capture():
            print("[Cleanup] Capture stopped")
    except Exception as exc:
        print(f"[Cleanup] Error stopping capture: {exc}")

    try:
        if streamer:
            streamer.stop()
            print("[Cleanup] Streamer stopped")
    except Exception as exc:
        print(f"[Cleanup] Error stopping streamer: {exc}")

    clear_queues(queues, timeout=queue_timeout)

    if rtmp_thread is not None and rtmp_thread.is_alive():
        rtmp_thread.join(timeout=3)

    print("[Cleanup] All resources cleaned up")
