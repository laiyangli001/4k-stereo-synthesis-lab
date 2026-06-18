import os
import sys
import time
import queue
import threading
import statistics
import tkinter as tk
from tkinter import ttk


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

cache_root = os.path.join(ROOT_DIR, "cache", "benchmark")
os.makedirs(cache_root, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(cache_root, "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(cache_root, "triton"))


class BenchmarkWindow:
    def __init__(self):
        self.q = queue.Queue()
        self.root = tk.Tk()
        self.root.title("Desktop2Stereo Inference Benchmark")
        self.root.geometry("760x520")
        self.root.minsize(680, 440)

        self.status = tk.StringVar(value="准备开始...")
        self.progress = tk.DoubleVar(value=0)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="推理极限测试", font=("Microsoft YaHei", 14, "bold")).pack(anchor="w")
        ttk.Label(main, textvariable=self.status).pack(anchor="w", pady=(8, 4))
        self.bar = ttk.Progressbar(main, variable=self.progress, maximum=100)
        self.bar.pack(fill=tk.X, pady=(0, 10))

        self.log = tk.Text(main, height=20, wrap="word")
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state="disabled")

        buttons = ttk.Frame(main)
        buttons.pack(fill=tk.X, pady=(10, 0))
        self.start_btn = ttk.Button(buttons, text="开始测试", command=self.start)
        self.start_btn.pack(side=tk.LEFT)
        ttk.Button(buttons, text="关闭", command=self.root.destroy).pack(side=tk.RIGHT)

        self.root.after(100, self._poll)

    def start(self):
        self.start_btn.configure(state="disabled")
        threading.Thread(target=self._worker, daemon=True).start()

    def _post(self, kind, payload):
        self.q.put((kind, payload))

    def _log(self, text):
        self._post("log", text)

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert(tk.END, payload + "\n")
                    self.log.see(tk.END)
                    self.log.configure(state="disabled")
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "progress":
                    self.progress.set(payload)
                elif kind == "done":
                    self.start_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _report(self, name, vals):
        avg = statistics.mean(vals)
        med = statistics.median(vals)
        p95 = sorted(vals)[int(len(vals) * 0.95) - 1]
        fps_avg = 1000.0 / avg if avg > 0 else 0
        self._log(f"{name}: avg={avg:.2f}ms median={med:.2f}ms p95={p95:.2f}ms fps_avg={fps_avg:.1f}")

    def _worker(self):
        try:
            self._post("status", "加载模型和当前 settings.yaml...")
            self._log(f"工作目录: {ROOT_DIR}")
            self._log(f"TorchInductor cache: {os.environ['TORCHINDUCTOR_CACHE_DIR']}")
            self._log(f"Triton cache: {os.environ['TRITON_CACHE_DIR']}")

            import numpy as np
            import torch
            from capture import capture_frame_to_rgb, prepare_rgb_for_depth_runtime
            from stereo_runtime import DepthRuntime, DepthRuntimeConfig
            from utils import FPS, TARGET_FPS, OUTPUT_RESOLUTION, MODEL, MODEL_ID

            device = "cuda" if torch.cuda.is_available() else "cpu"
            depth_runtime = DepthRuntime(
                DepthRuntimeConfig(
                    model_id=MODEL_ID,
                    cache_dir="./models",
                    depth_backend="auto",
                    device=device,
                )
            )

            self._log(f"model={MODEL} model_id={MODEL_ID}")
            self._log(f"device={device} backend=auto runtime=DepthRuntime")
            self._log(f"target_fps={TARGET_FPS} fps={FPS} output_resolution={OUTPUT_RESOLUTION}")

            width, height = 3840, 2160
            frame = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)

            warmup = 6
            samples = 60
            total_steps = warmup + samples

            self._post("status", "Warmup 中，排除首次加载和引擎初始化...")
            for i in range(warmup):
                rgb = capture_frame_to_rgb(frame, OUTPUT_RESOLUTION)
                runtime_rgb = prepare_rgb_for_depth_runtime(rgb, device=device)
                depth_result = depth_runtime.predict_depth_frame(runtime_rgb)
                depth = depth_result.depth
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._post("progress", (i + 1) / total_steps * 100)
                self._log(f"warmup {i + 1}/{warmup}")

            proc_times = []
            depth_times = []
            total_times = []

            self._post("status", "正式测试 capture RGB prepare + DepthRuntime...")
            for i in range(samples):
                t0 = time.perf_counter()
                rgb = capture_frame_to_rgb(frame, OUTPUT_RESOLUTION)
                runtime_rgb = prepare_rgb_for_depth_runtime(rgb, device=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                depth_result = depth_runtime.predict_depth_frame(runtime_rgb)
                depth = depth_result.depth
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

                proc_times.append((t1 - t0) * 1000)
                depth_times.append((t2 - t1) * 1000)
                total_times.append((t2 - t0) * 1000)

                if (i + 1) % 5 == 0 or i == 0:
                    avg_total = statistics.mean(total_times)
                    self._log(f"sample {i + 1}/{samples}: latest_total={total_times[-1]:.2f}ms avg_total={avg_total:.2f}ms fps={1000.0 / avg_total:.1f}")
                self._post("progress", (warmup + i + 1) / total_steps * 100)

            self._post("status", "测试完成")
            self._log("")
            self._log("=== 结果 ===")
            self._report("process", proc_times)
            self._report("depth", depth_times)
            self._report("process+depth", total_times)
            self._log(f"depth_output_shape={tuple(depth.shape) if hasattr(depth, 'shape') else None}")
            self._log(f"last_timing={depth_runtime.last_timing}")
        except Exception as exc:
            self._post("status", "测试失败")
            self._log(f"ERROR: {type(exc).__name__}: {exc}")
        finally:
            self._post("done", None)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    BenchmarkWindow().run()
