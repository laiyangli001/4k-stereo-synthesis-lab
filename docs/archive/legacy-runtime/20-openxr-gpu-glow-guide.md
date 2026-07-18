# OpenXR GPU Glow 实现指南

本文记录当前 OpenXR Default 房间里的 GPU glow 系统：原理、实现过程、使用方法、调用链，以及后续实现更复杂 glow / 反射效果的扩展方式。

## 目标

GPU glow 的目标是把屏幕内容驱动的辉光、雾化光束、环绕反射等效果放到 GPU 渲染链路内完成，避免 CPU 采样造成同步卡顿。

之前 CPU 采样方案的问题是：从 4K 帧读取颜色统计会触发 CPU/GPU 同步，实测可能出现 `sample_glow=200ms+`，导致 OpenXR 房间模式掉到个位数帧率。当前方案改为：屏幕纹理已经在 GL texture 中，直接通过 shader / 低分辨率中间纹理在 GPU 内采样、模糊和混合。

## 核心原理

### 数据流

```text
OpenXR source frame / runtime eye texture
  -> GL color texture
  -> optional glow downsample texture, about 192x108 max
  -> glow shader samples low-frequency color
  -> render screen glow / frosted beams / surround shell
```

关键点：

- 不把屏幕图像读回 CPU。
- glow 使用低分辨率中间纹理，保留颜色变化，过滤掉图像细节。
- shader 用 `textureLod()`、邻域采样、边缘采样和区域采样生成低频颜色场。
- Default + Glow Off 仍走 blank fast path，不创建/采样 glow 纹理。

## 当前 glow 模式

Default 环境通过 `glow_mode` 区分效果：

| mode | 名称 | 渲染路径 | 用途 |
|---|---|---|---|
| `off` | Glow Off | 不渲染 glow | 默认高性能空房间 |
| `screen` | Screen Glow | `_render_glow()` + `_GLOW_FRAG` | 屏幕周围边缘辉光 |
| `frosted` | Frosted Glow | `_render_frosted_glow()` + `_FROSTED_GLOW_FRAG` | 从屏幕边缘向观众发射的毛玻璃光束 |
| `veil` | Frosted Veil | `_render_frosted_veil()` 复用 frosted shader | 更柔、更雾化的前景光幕 |
| `surround` | Surround Glow | `_render_glow_shell()` + `_GLOW_SHELL_FRAG` | 包裹视角的环绕壳层反射 |

模式切换在 `src/xr_viewer/environment_profiles.py` 的 `_cycle_glow_mode_from_y()` 中处理。它会按目标 `glow_mode` 从 `lighting_presets` 找到对应 preset，并应用该模式的强度、壳层、frosted 参数。

## 使用方法

### 运行时切换

在 OpenXR Default 房间里：

```text
左 grip + Y：循环 glow 模式
```

循环顺序：

```text
Frosted Veil -> Frosted Glow -> Surround Glow -> Screen Glow -> Glow Off
```

控制台会输出：

```text
[OpenXRViewer] Glow mode: surround
```

### 配置入口

Default 配置文件：

```text
src/xr_viewer/environments/Default/profile.json
```

常用字段：

```json
{
  "glow_mode": "off",
  "glow_intensity": 0.5,
  "glow_width": 1.9,
  "glow_surround_margin": 18.0,
  "glow_intensity_multiplier": 0.0,
  "glow_shell_intensity_multiplier": 0.0,
  "glow_shell_radius": 20.0,
  "glow_shell_height": 9.5,
  "frosted_glow_intensity": 3.0,
  "frosted_glow_alpha": 0.55,
  "frosted_glow_lod": 5.4,
  "frosted_glow_blend": 2.25,
  "frosted_glow_thickness": 2.65,
  "frosted_glow_diffuse": 1.45
}
```

默认顶层是 `off`，保证进入房间后走 fast path。每个可见效果的强度放在 `lighting_presets` 中，切换模式时再应用。

## 实现位置

主要文件：

| 文件 | 职责 |
|---|---|
| `src/xr_viewer/environment_effects.py` | 根据 `glow_mode` 调用不同渲染路径 |
| `src/xr_viewer/core_screen_quality.py` | 创建和更新低分辨率 glow texture |
| `src/xr_viewer/glsl.py` | glow / frosted / surround shader 源码 |
| `src/xr_viewer/environment_profiles.py` | glow preset 切换和参数应用 |
| `src/xr_viewer/implementation.py` | shader program、VAO、uniform 初始化 |

### 背景/前景渲染分层

`environment_effects.py` 里分两层调用：

```text
_render_screen_background_effects()
  -> surround: _render_glow_shell()
  -> screen:   _render_glow()

_render_screen_foreground_effects()
  -> veil:     _render_frosted_veil()
  -> frosted:  _render_frosted_glow()
```

这样 screen/surround 可以在屏幕后或背景层渲染，frosted/veil 可以盖在屏幕前方，形成“光从屏幕向观众方向发射”的效果。

## GPU 低分辨率纹理

`core_screen_quality.py::_prepare_glow_downsample_texture()` 负责生成 glow 中间纹理。

当前尺寸规则：

```python
out_w = max(32, min(192, src_w // 20))
out_h = max(18, min(108, src_h // 20))
```

例如 3840x2160 输入会生成约 192x108 的低分辨率纹理。这样保留大色块变化，但不会保留清晰图像细节。

该函数会缓存同一帧、同一 eye、同一 source texture 的结果，避免一个眼睛内重复 downsample。

## 各模式实现说明

### Screen Glow

调用链：

```text
glow_mode = screen
  -> _render_screen_background_effects()
  -> _render_glow()
  -> _GLOW_FRAG
```

特点：

- 用扩大后的屏幕矩形做辉光 plane。
- `u_screen_half` 定义屏幕内部区域，shader 只在屏幕外部绘制。
- 如果有源纹理，则通过 `u_glow_tex` 按最近屏幕区域采样颜色。
- 适合做屏幕四周 Ambilight 风格辉光。

### Frosted Glow / Frosted Veil

调用链：

```text
glow_mode = frosted
  -> _render_screen_foreground_effects()
  -> _render_frosted_glow()
  -> _FROSTED_GLOW_FRAG

glow_mode = veil
  -> _render_frosted_veil()
  -> _render_frosted_glow()
```

特点：

- 几何由屏幕四条边向观众方向拉出。
- shader 从屏幕边缘采样颜色，做多点 blur、亮度阈值、噪声、厚度和漫反射混合。
- 适合做毛玻璃、空气散射、发射光线。

主要实时参数：

| 参数 | 作用 |
|---|---|
| `frosted_glow_intensity` | 光束亮度 |
| `frosted_glow_alpha` | 透明度 |
| `frosted_glow_threshold` | 高亮提取阈值 |
| `frosted_glow_lod` | 采样模糊层级 |
| `frosted_glow_blend` | 毛玻璃混合程度 |
| `frosted_glow_thickness` | 光束体积厚度 |
| `frosted_glow_diffuse` | 漫反射散射强度 |

### Surround Glow

调用链：

```text
glow_mode = surround
  -> _render_screen_background_effects()
  -> _render_glow_shell()
  -> _prepare_glow_downsample_texture()
  -> _GLOW_SHELL_FRAG
```

特点：

- 用半圆柱/壳层包裹观众视角。
- 壳层中心优先使用头显位置 `_head_pos_w`，没有时 fallback 到屏幕附近。
- 半径和高度由 `glow_shell_radius` / `glow_shell_height` 控制。
- 颜色不是直接投影整张图，而是混合两种低频采样：
  - `sample_border_color()`：采屏幕上下左右边缘平均色。
  - `sample_region_reflection()`：16x9 区域化低频反射，5x5 邻域 blur，`textureLod(..., 2.2)`。

这避免了“把画面线条投到壳层上”的问题，同时比单一平均色更有区域层次。

## 如何调用一个新 GPU glow 效果

新增效果建议按以下步骤：

1. 在 `glsl.py` 增加 shader 源码。
2. 在 `implementation.py::_init_moderngl()` 创建 program / VAO，并绑定 sampler uniform。
3. 在 `environment_effects.py` 增加 `_render_xxx_glow()`。
4. 在 `_render_screen_background_effects()` 或 `_render_screen_foreground_effects()` 按 `glow_mode` 分发。
5. 在 `Default/profile.json` 的 `lighting_presets` 里增加 preset。
6. 在 `environment_profiles.py::_cycle_glow_mode_from_y()` 的 `modes` 列表里加入新 mode。
7. 增加测试，至少覆盖 shader 字符串、mode 分发和 fast path 不被破坏。

最小调用形态示例：

```python
mode = str(getattr(self, '_glow_mode', 'screen') or 'screen').strip().lower()
if mode == 'my_new_glow':
    self._render_my_new_glow(mgl_fbo, vp_mat)
```

如果需要屏幕颜色纹理，复用：

```python
glow_tex = self._prepare_glow_downsample_texture(source_tex, source_size)
if mgl_fbo is not None:
    mgl_fbo.use()
```

注意：`_prepare_glow_downsample_texture()` 会切换到自己的 FBO，渲染 glow 前必须切回当前 eye 的 `mgl_fbo`。

## 如何实现复杂 glow

### 1. 多区域采样

适合：区域化反射、环境光、屏幕墙面染色。

方式：

```glsl
vec2 grid = vec2(16.0, 9.0);
vec2 q = (floor(screen_uv * grid) + vec2(0.5)) / grid;
```

然后对 `q` 周围做 3x3 或 5x5 邻域采样。区域越小，层次越细；区域越大，越不容易出现图像感。

推荐起点：

| 用途 | 网格 | 采样 |
|---|---:|---|
| 柔和环境色 | 8x5 | 5x5 blur |
| 屏幕区域反射 | 16x9 | 5x5 blur + LOD 2.0+ |
| 高细节动态光 | 32x18 | 7x7 blur 或 mip chain |

### 2. 边缘采样

适合：屏幕边缘向外发光、Ambilight、frosted beam。

只采屏幕上下左右边缘，避免画面中央细节进入 glow：

```glsl
textureLod(u_glow_tex, vec2(x, 0.045), lod)
textureLod(u_glow_tex, vec2(x, 0.955), lod)
textureLod(u_glow_tex, vec2(0.045, y), lod)
textureLod(u_glow_tex, vec2(0.955, y), lod)
```

### 3. 高亮提取

适合：只让爆炸、灯光、UI 高亮产生明显辉光。

```glsl
float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
float bright = smoothstep(threshold, 1.0, luma);
color *= bright;
```

### 4. 毛玻璃/雾化

适合：光穿过薄雾、毛玻璃、影院空气感。

可组合：

- 多点采样。
- 高 LOD 采样。
- 低频 noise。
- alpha 随距离指数衰减。
- 亮度阈值和色彩保留混合。

### 5. 深度辅助

未来可以把 depth texture 引入复杂 glow：

- 前景避让：人物/物体前方不被雾层盖死。
- 景深式散射：远处光更柔，近处边缘更锐。
- 物体轮廓光：只在深度突变处生成额外辉光。

这类效果适合以后用 Triton 或专用 shader 预处理 mask，再交给 GL shader 渲染。

## 性能原则

1. Glow Off 必须不触发 downsample、不触发 CPU 采样。
2. 不要在 OpenXR 渲染循环里调用 `.cpu()`、`.numpy()`、`glReadPixels()` 做颜色统计。
3. 不要直接把整张画面投影到环境上；必须低频化、区域化或边缘化。
4. 复杂效果优先在低分辨率 glow texture 上做。
5. shader 采样数要和覆盖面积一起评估：fullscreen shell 的 25 次采样比小面片贵很多。
6. 每次加效果后测试 Default + Glow Off，确认仍保持 fast path。

## 验证命令

修改 glow 相关代码后至少运行：

```powershell
src\python3\python.exe -m py_compile src\xr_viewer\glsl.py src\xr_viewer\environment_effects.py src\xr_viewer\implementation.py tests\test_environment_fast_path.py
src\python3\python.exe -m pytest tests\test_environment_fast_path.py
```

OpenXR 实测建议：

1. Default + Glow Off：确认帧率不掉，应该仍走 blank fast path。
2. 左 grip + Y 切到 Screen Glow：确认屏幕边缘颜色跟随。
3. 切到 Frosted Glow / Frosted Veil：确认光束从屏幕边缘向观看者发射。
4. 切到 Surround Glow：确认壳层颜色有区域层次，但没有清晰图像线条。

需要分段性能日志时设置：

```powershell
$env:D2S_OPENXR_PERF_LOG = "1"
```

关注日志中是否出现异常的 `sample_glow` 或 `render_eyes` 耗时。正常 GPU glow 不应出现 200ms 级 CPU 采样等待。

## 常见问题

### 控制台显示切换了 glow，但画面没有 glow

通常是目标 mode 的 multiplier 没有从 `lighting_presets` 应用，导致强度为 0。检查：

- `glow_intensity_multiplier`
- `glow_shell_intensity_multiplier`
- `_cycle_glow_mode_from_y()` 是否按目标 `glow_mode` 找到了对应 preset

### Glow 变成画面投影或出现线条

说明 shader 直接按 UV 采了整张图。应改成：

- 边缘采样。
- 区域量化采样。
- 提高 LOD。
- 增加邻域 blur。
- 降低区域反射混合比例。

### 帧率突然大幅下降

优先检查：

- 是否重新启用了 CPU 采样。
- 是否每帧重复创建纹理/FBO。
- 是否在大面积 shell 上使用了过多采样。
- 是否 Glow Off 仍触发 `_prepare_glow_downsample_texture()`。

## 后续扩展方向

1. Surround shell 参数化：把 16x9、LOD、region mix 做成 profile 字段。
2. 环境模型反射：给墙面/边框材质增加 `u_screen_reflect_tex`，按法线和粗糙度采样。
3. Depth-aware fog：用深度图做前景避让和轮廓光。
4. Triton 预处理：生成高光 mask、运动 mask、区域统计 buffer，再交给 GL shader 使用。
5. Debug overlay：显示 glow downsample texture / 区域色块，方便调参。
