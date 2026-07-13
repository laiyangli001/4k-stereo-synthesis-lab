"""GLSL shader sources for the OpenXR viewer."""

# World-space vertex shader: applies MVP to place the quad in the scene
_WORLD_VERT = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;
uniform mat4 u_mvp;
void main() {
    uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 0.0, 1.0);
}
"""

# World-space overlay fragment shader (plain RGBA texture, no parallax)
# u_alpha scales the output alpha; defaults to 1.0 (fully opaque per texture).
_OVERLAY_FRAG = """
#version 330
uniform sampler2D tex;
uniform float u_alpha;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(tex, uv);
    fragColor = vec4(c.rgb, c.a * u_alpha);
}
"""

_SCREEN_QUALITY_VERT = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;
void main() {
    uv = in_uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_SCREEN_DOWNSAMPLE_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform sampler2D tex_color;
uniform vec2 u_input_size;

float sinc(float x) {
    x = abs(x);
    if (x < 1e-5) {
        return 1.0;
    }
    float pix = 3.141592653589793 * x;
    return sin(pix) / pix;
}

float lanczos2(float x) {
    x = abs(x);
    if (x >= 2.0) {
        return 0.0;
    }
    return sinc(x) * sinc(x * 0.5);
}

void main() {
    vec2 src_pos = uv * u_input_size - vec2(0.5);
    vec2 base = floor(src_pos);
    vec3 accum = vec3(0.0);
    float weight_sum = 0.0;
    for (int y = -1; y <= 2; ++y) {
        for (int x = -1; x <= 2; ++x) {
            vec2 sample_pos = base + vec2(float(x), float(y));
            vec2 delta = src_pos - sample_pos;
            float w = lanczos2(delta.x) * lanczos2(delta.y);
            vec2 sample_uv = (sample_pos + vec2(0.5)) / u_input_size;
            accum += texture(tex_color, clamp(sample_uv, vec2(0.0), vec2(1.0))).rgb * w;
            weight_sum += w;
        }
    }
    frag_color = vec4(clamp(accum / max(weight_sum, 1e-6), 0.0, 1.0), 1.0);
}
"""

_SCREEN_RCAS_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform sampler2D tex_scene;
uniform vec2 u_output_size;
uniform float u_sharpness;

float sat(float x) {
    return clamp(x, 0.0, 1.0);
}

float rcp_safe(float x) {
    return 1.0 / max(abs(x), 1e-6);
}

float luma2(vec3 c) {
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

vec3 sample_scene(vec2 p) {
    return texture(tex_scene, clamp(p, vec2(0.0), vec2(1.0))).rgb;
}

vec3 fsr_rcas(vec2 p) {
    vec2 texel = 1.0 / u_output_size;
    vec3 b = sample_scene(p + vec2(0.0, -texel.y));
    vec3 d = sample_scene(p + vec2(-texel.x, 0.0));
    vec3 e = sample_scene(p);
    vec3 f = sample_scene(p + vec2(texel.x, 0.0));
    vec3 h = sample_scene(p + vec2(0.0, texel.y));

    float bL = luma2(b);
    float dL = luma2(d);
    float eL = luma2(e);
    float fL = luma2(f);
    float hL = luma2(h);
    float nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    float lMax = max(max(max(bL, dL), max(eL, fL)), hL);
    float lMin = min(min(min(bL, dL), min(eL, fL)), hL);
    nz = sat(abs(nz) * rcp_safe(lMax - lMin));
    nz = -0.5 * nz + 1.0;

    vec3 mn4 = min(min(b, d), min(f, h));
    vec3 mx4 = max(max(b, d), max(f, h));
    vec3 hitMin = min(mn4, e) / max(4.0 * mx4, vec3(1e-6));
    vec3 hitMax = (vec3(1.0) - max(mx4, e)) / min(4.0 * mn4 - 4.0, vec3(-1e-6));
    vec3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(max(lobeRGB.r, lobeRGB.g), lobeRGB.b);
    float rcasLimit = 0.25 - (1.0 / 16.0);

    float sharpnessStops = mix(2.0, 0.0, sat(u_sharpness));
    float con = exp2(-sharpnessStops);
    lobe = max(-rcasLimit, min(lobe, 0.0)) * con * nz;
    float rcpL = rcp_safe(4.0 * lobe + 1.0);
    return clamp((lobe * b + lobe * d + lobe * h + lobe * f + e) * rcpL, 0.0, 1.0);
}

void main() {
    frag_color = vec4(fsr_rcas(uv), 1.0);
}
"""

_QUAD_COPY_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform sampler2D tex_source;
uniform int u_flip_y;
uniform int u_linearize_srgb;

vec3 srgb_to_linear(vec3 c) {
    bvec3 cutoff = lessThanEqual(c, vec3(0.04045));
    vec3 low = c / 12.92;
    vec3 high = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, cutoff);
}

void main() {
    vec2 p = (u_flip_y == 1) ? vec2(uv.x, 1.0 - uv.y) : uv;
    vec4 color = texture(tex_source, p);
    if (u_linearize_srgb == 1) {
        color.rgb = srgb_to_linear(color.rgb);
    }
    frag_color = color;
}
"""

_PANORAMA_VERT = """
#version 330
in vec2 in_position;
out vec2 v_ndc;
void main() {
    v_ndc = in_position;
    gl_Position = vec4(in_position, 0.999, 1.0);
}
"""

_PANORAMA_FRAG = """
#version 330
uniform sampler2D u_tex;
uniform sampler2D u_screen_light_tex;
uniform sampler2D u_wall_light_mask_tex;
uniform mat4 u_inv_proj;
uniform mat4 u_inv_view_rot;
uniform float u_yaw_offset;
uniform float u_exposure;
uniform int u_flip_y;
uniform int u_stereo_layout;
uniform int u_eye_index;
uniform int u_screen_light_enabled;
uniform int u_wall_light_mask_enabled;
uniform float u_screen_light_intensity;
uniform vec2 u_screen_light_uv;
uniform vec2 u_screen_light_radius;
in vec2 v_ndc;
out vec4 fragColor;

const float PI = 3.14159265358979323846;

vec3 screen_light_probe_color() {
    vec3 color = vec3(0.0);
    color += textureLod(u_screen_light_tex, vec2(0.25, 0.25), 0.0).rgb;
    color += textureLod(u_screen_light_tex, vec2(0.50, 0.25), 0.0).rgb;
    color += textureLod(u_screen_light_tex, vec2(0.75, 0.25), 0.0).rgb;
    color += textureLod(u_screen_light_tex, vec2(0.25, 0.50), 0.0).rgb;
    color += textureLod(u_screen_light_tex, vec2(0.50, 0.50), 0.0).rgb;
    color += textureLod(u_screen_light_tex, vec2(0.75, 0.50), 0.0).rgb;
    color += textureLod(u_screen_light_tex, vec2(0.25, 0.75), 0.0).rgb;
    color += textureLod(u_screen_light_tex, vec2(0.50, 0.75), 0.0).rgb;
    color += textureLod(u_screen_light_tex, vec2(0.75, 0.75), 0.0).rgb;
    return color * (1.0 / 9.0);
}

void main() {
    vec4 view_h = u_inv_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 view_dir = normalize(view_h.xyz / max(abs(view_h.w), 1e-6));
    vec3 dir = normalize((u_inv_view_rot * vec4(view_dir, 0.0)).xyz);

    float u = atan(dir.x, -dir.z) / (2.0 * PI) + 0.5 + u_yaw_offset;
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
    if (u_flip_y != 0) {
        v = 1.0 - v;
    }

    vec2 pano_uv = vec2(fract(u), clamp(v, 0.0, 1.0));
    vec2 sample_uv = pano_uv;
    if (u_stereo_layout == 1) {
        sample_uv.x = pano_uv.x * 0.5 + (u_eye_index == 1 ? 0.5 : 0.0);
    }
    vec3 color = texture(u_tex, sample_uv).rgb;
    if (u_screen_light_enabled == 1) {
        vec2 d = (pano_uv - u_screen_light_uv) / max(u_screen_light_radius, vec2(0.001));
        float mask = exp(-dot(d, d));
        if (u_wall_light_mask_enabled == 1) {
            mask *= textureLod(u_wall_light_mask_tex, pano_uv, 0.0).r;
        }
        vec3 screen_col = screen_light_probe_color();
        color += screen_col * mask * u_screen_light_intensity;
    }
    fragColor = vec4(color * u_exposure, 1.0);
}
"""

_GLOW_DOWNSAMPLE_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform sampler2D tex_color;
uniform vec2 u_input_size;

vec3 sample_color(vec2 p) {
    return texture(tex_color, clamp(p, vec2(0.0), vec2(1.0))).rgb;
}

void main() {
    vec2 texel = 1.0 / max(u_input_size, vec2(1.0));
    vec3 c = sample_color(uv) * 0.30;
    c += sample_color(uv + texel * vec2(-2.0,  0.0)) * 0.09;
    c += sample_color(uv + texel * vec2( 2.0,  0.0)) * 0.09;
    c += sample_color(uv + texel * vec2( 0.0, -2.0)) * 0.09;
    c += sample_color(uv + texel * vec2( 0.0,  2.0)) * 0.09;
    c += sample_color(uv + texel * vec2(-1.5, -1.5)) * 0.085;
    c += sample_color(uv + texel * vec2( 1.5, -1.5)) * 0.085;
    c += sample_color(uv + texel * vec2(-1.5,  1.5)) * 0.085;
    c += sample_color(uv + texel * vec2( 1.5,  1.5)) * 0.085;
    frag_color = vec4(c, 1.0);
}
"""

_CURVED_COPY_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform sampler2D tex_color;
void main() {
    frag_color = texture(tex_color, vec2(uv.x, 1.0 - uv.y));
}
"""

# Solid-color vertex shader (no UV -avoids GLSL optimizer stripping in_uv)
_SOLID_VERT = """
#version 330
in vec2 in_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(in_position, 0.0, 1.0);
}
"""

# Solid-color fragment shader for the screen border quad
_SOLID_FRAG = """
#version 330
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }
"""

# Brushed-metal border fragment shader
_BORDER_FRAG = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform vec3 u_color;
uniform float u_alpha;
uniform vec2 u_border_uv;    // border half-width in UV (x, y)
void main() {
    vec2 d = min(uv, 1.0 - uv);
    if (d.x > u_border_uv.x && d.y > u_border_uv.y) discard;
    float bx = 1.0 - d.x / u_border_uv.x;
    float by = 1.0 - d.y / u_border_uv.y;
    float bp = clamp(max(bx, by), 0.0, 1.0);
    float bevel = (1.0 - smoothstep(0.0, 0.4, bp)) * 0.25;
    float shade = mix(1.0, 0.5, bp * bp);
    float brush = sin(uv.y * 3000.0 + uv.x * 500.0) * 0.015;
    brush += sin(uv.y * 5000.0) * 0.008;
    vec3 col = u_color * shade + vec3(bevel) + vec3(brush);
    fragColor = vec4(col, u_alpha);
}
"""

# Shadow fragment shader: soft ground shadow beneath the screen
_SHADOW_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform float u_opacity;
void main() {
    vec2 centered = uv - 0.5;
    float dist = length(centered);
    // Gaussian falloff: darkest at center, smooth fade to edges
    float shadow = exp(-dist * dist * 6.0);
    shadow *= u_opacity;
    vec2 shadow_edge = smoothstep(0.0, 0.08, uv) * smoothstep(1.0, 0.92, uv);
    shadow *= min(shadow_edge.x, shadow_edge.y);
    frag_color = vec4(0.0, 0.0, 0.0, shadow);
}
"""

# Ground ambient light fragment shader: colored light pool cast by the screen
_GROUND_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform vec3 u_color;
uniform float u_intensity;
void main() {
    vec2 dir = uv - vec2(0.5, 1.0);
    float dist = length(dir);
    float light = exp(-dist * dist * 3.0);
    light *= u_intensity;
    vec2 edge = smoothstep(0.0, 0.1, uv) * smoothstep(1.0, 0.9, uv);
    light *= min(edge.x, edge.y);
    frag_color = vec4(u_color * light, light);
}
"""

# 3D vertex shader for tapered rainbow beam
_BEAM_VERT = """
#version 330
in vec3 in_position;
in float in_v;
out float v_v;
uniform mat4 u_mvp;
void main() {
    v_v = in_v;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_BEAM_FRAG = """
#version 330
in float v_v;
out vec4 fragColor;
uniform float u_time;
void main() {
    // Rainbow gradient: blue->cyan->green->yellow->red, flowing from root to tip.
    // Keep the time sign aligned with the D3D11 laser shader so both backends
    // show the same forward beam flow for the shared beamV contract.
    float t = fract(v_v - u_time * 0.4);
    vec3 col;
    if (t < 0.167)      col = mix(vec3(0.0,0.4,1.0), vec3(0.0,1.0,1.0), t/0.167);
    else if (t < 0.333) col = mix(vec3(0.0,1.0,1.0), vec3(0.0,1.0,0.0), (t-0.167)/0.166);
    else if (t < 0.5)   col = mix(vec3(0.0,1.0,0.0), vec3(1.0,1.0,0.0), (t-0.333)/0.167);
    else if (t < 0.667) col = mix(vec3(1.0,1.0,0.0), vec3(1.0,0.5,0.0), (t-0.5)/0.167);
    else if (t < 0.833) col = mix(vec3(1.0,0.5,0.0), vec3(1.0,0.0,0.0), (t-0.667)/0.166);
    else                col = mix(vec3(1.0,0.0,0.0), vec3(0.0,0.4,1.0), (t-0.833)/0.167);
    fragColor = vec4(col, 1.0);
}
"""

# Curved-screen vertex shader: in_position is a world-space vec3 arc point (no model matrix).
# UV is passed through normally.  vp_mat is the combined view-projection for the current eye.
_CURVED_VERT = """
#version 330
in vec3 in_position;
in vec2 in_uv;
out vec2 uv;
uniform mat4 u_mvp;
void main() {
    uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

# VR controller model shader (improved: supports Blinn-Phong lighting and texture toggle)
_CTRL_VERT = """
#version 330
in vec3 in_position;
in vec3 in_normal;   // Corresponds to 12 skipped bytes in the data
in vec2 in_uv;
out vec2 v_uv;
out vec3 v_normal;
out vec3 v_position;
uniform mat4 u_mvp;
uniform mat4 u_model; // Used for normal transformation
void main() {
    v_uv = in_uv;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal; // Normal transformation
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_position = world_pos.xyz;
    gl_Position = u_mvp * world_pos;
}
"""

_CTRL_FRAG = """
#version 330
in vec2 v_uv;
in vec3 v_normal;
in vec3 v_position;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform sampler2D u_env_tex;
uniform sampler2D u_screen_light_tex;
uniform vec3 u_base_color_factor; // Base color factor
uniform int u_use_texture;     // 0: use solid color, 1: sample texture
uniform int u_use_env_tex;
uniform int u_env_stereo_layout;
uniform int u_env_eye_index;
uniform int u_screen_light_enabled;
uniform float u_env_intensity;
uniform float u_screen_light_intensity;
uniform float u_top_light_intensity;
uniform float u_env_gamma;
uniform vec3 u_camera_pos;     // Camera world coordinates (= headset position)
uniform vec3 u_screen_light_pos;
uniform vec3 u_screen_light_normal;
uniform vec3 u_screen_light_right;
uniform vec3 u_screen_light_up;
uniform vec2 u_screen_light_half_size;

vec3 gltfSrgbToLinear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + vec3(0.055)) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 gltfToneMap(vec3 linearColor) {
    linearColor = max(linearColor, vec3(0.0));
    return linearColor / (linearColor + vec3(1.0));
}

vec3 gltfLinearToOutput(vec3 linearColor, float gamma) {
    float safeGamma = gamma > 0.0 ? gamma : 2.2;
    return pow(clamp(gltfToneMap(linearColor), 0.0, 1.0), vec3(1.0 / safeGamma));
}

vec2 env_uv(vec3 dir) {
    dir = normalize(dir);
    float u = atan(dir.x, dir.z) / 6.28318530718 + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) / 3.14159265359 + 0.5;
    return vec2(u, 1.0 - v);
}

vec2 env_sample_uv(vec3 dir) {
    vec2 uv = env_uv(dir);
    if (u_env_stereo_layout == 1) {
        uv.x = uv.x * 0.5 + (u_env_eye_index == 1 ? 0.5 : 0.0);
    }
    return uv;
}

void main() {
    vec2 t_uv = v_uv;  // alias (no transform for controllers)
    // Discard back faces (inner walls), keep only front faces (outer shell)
    if (!gl_FrontFacing) discard;
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_camera_pos - v_position);

    vec3 baseColor;
    if (u_use_texture == 1) {
        baseColor = gltfSrgbToLinear(texture(u_tex, t_uv).rgb) * u_base_color_factor;
    } else {
        baseColor = u_base_color_factor;
    }

    vec3 color = baseColor * 0.30;
    vec3 R = reflect(-V, N);
    if (u_use_env_tex == 1) {
        vec3 env_spec = textureLod(u_env_tex, env_sample_uv(R), 3.0).rgb;
        vec3 env_diff = textureLod(u_env_tex, env_sample_uv(N), 5.0).rgb;
        float view_facing = smoothstep(-0.25, 0.65, dot(N, V));
        color = baseColor * mix(vec3(0.32), env_diff, 0.36) * u_env_intensity + env_spec * (0.30 * u_env_intensity * view_facing);
    }
    vec3 top_light_pos = u_camera_pos + vec3(0.0, 0.45, -0.18);
    vec3 top_light_dir = normalize(top_light_pos - v_position);
    float top_facing = max(dot(N, top_light_dir), 0.0);
    float top_fill = pow(top_facing, 1.25) * smoothstep(-0.20, 0.65, dot(N, V));
    color += baseColor * vec3(0.95, 0.97, 1.0) * (u_top_light_intensity * top_fill);

    if (u_screen_light_enabled == 1) {
        vec3 screen_tint = (
            textureLod(u_screen_light_tex, vec2(0.50, 0.50), 0.0).rgb +
            textureLod(u_screen_light_tex, vec2(0.25, 0.30), 0.0).rgb +
            textureLod(u_screen_light_tex, vec2(0.75, 0.30), 0.0).rgb +
            textureLod(u_screen_light_tex, vec2(0.25, 0.70), 0.0).rgb +
            textureLod(u_screen_light_tex, vec2(0.75, 0.70), 0.0).rgb
        ) * 0.20;
        vec3 screen_light_dir = normalize(u_screen_light_pos - v_position);
        float screen_facing = max(dot(N, screen_light_dir), 0.0);
        float screen_key = pow(screen_facing, 0.75);
        color += baseColor * screen_tint * (1.00 * u_screen_light_intensity * screen_key);

        float denom = dot(R, u_screen_light_normal);
        if (abs(denom) > 0.001) {
            float t = dot(u_screen_light_pos - v_position, u_screen_light_normal) / denom;
            if (t > 0.0) {
                vec3 hit = v_position + R * t;
                vec3 local = hit - u_screen_light_pos;
                vec2 screen_p = vec2(
                    dot(local, u_screen_light_right) / max(u_screen_light_half_size.x, 0.001),
                    dot(local, u_screen_light_up) / max(u_screen_light_half_size.y, 0.001)
                );
                if (abs(screen_p.x) <= 1.0 && abs(screen_p.y) <= 1.0) {
                    vec2 screen_uv = screen_p * 0.5 + 0.5;
                    vec3 screen_col = textureLod(u_screen_light_tex, vec2(1.0 - screen_uv.x, 1.0 - screen_uv.y), 0.0).rgb;
                    float fresnel = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 2.0);
                    color += mix(baseColor * screen_tint, screen_col, 0.72) * (0.38 + 0.95 * fresnel) * u_screen_light_intensity * screen_facing;
                }
            }
        }
    }

    fragColor = vec4(gltfLinearToOutput(color, u_env_gamma), 1.0);
}
"""

_ENV_VERT = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec2 in_uv1;
in vec4 in_tangent;  // xyz + bitangent_sign (glTF spec 3.7.4)
out vec2 v_uv;
out vec2 v_uv1;
out vec3 v_normal;
out vec3 v_position;
out vec3 v_tangent;
out float v_bitangent_sign;
uniform mat4 u_mvp;
uniform mat4 u_model;
void main() {
    v_uv = in_uv;
    v_uv1 = in_uv1;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_tangent = normalize(mat3(transpose(inverse(u_model))) * in_tangent.xyz);
    v_bitangent_sign = in_tangent.w;
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_position = world_pos.xyz;
    gl_Position = u_mvp * world_pos;
}
"""

_ENV_FRAG = """
#version 330
in vec2 v_uv;
in vec2 v_uv1;
in vec3 v_normal;
in vec3 v_position;
in vec3 v_tangent;
in float v_bitangent_sign;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform sampler2D u_normal_tex;    // normal map (texture unit 4)
uniform sampler2D u_occlusion_tex; // occlusion map (texture unit 5)
uniform sampler2D u_mr_tex;        // metallicRoughness (texture unit 6: B=metal, G=rough)
uniform sampler2D u_emissive_tex;  // emissive map (texture unit 7)
uniform vec3 u_light_color;
uniform float u_top_light_intensity;
uniform vec3 u_ambient_color;
uniform vec3 u_base_color_factor;
uniform float u_base_alpha;
uniform int u_use_texture;
uniform int u_use_normal_tex;
uniform float u_normal_scale;
uniform int u_use_occlusion_tex;
uniform float u_occlusion_strength;
uniform vec3 u_camera_pos;
uniform float u_roughness;
uniform float u_metallic;
uniform vec3 u_emissive_factor;
uniform int u_unlit;             // KHR_materials_unlit: skip lighting
uniform float u_alpha_cutoff;    // alphaMode=MASK discard threshold
uniform int u_alpha_mode;        // 0=OPAQUE, 1=MASK, 2=BLEND
uniform int u_double_sided;      // glTF doubleSided
uniform int u_use_mr_tex;        // 0: use uniform factors, 1: sample mr texture
uniform int u_use_emissive_tex;  // 0: use factor only, 1: sample emissive texture
uniform int u_normal_texcoord;
uniform int u_occlusion_texcoord;
uniform int u_mr_texcoord;
uniform int u_emissive_texcoord;
uniform int u_base_texcoord;
uniform int u_baked_lightmap;    // 1: occlusion texture stores RGB baked lightmap on UV1
uniform vec2 u_tex_offset;       // KHR_texture_transform offset
uniform vec2 u_tex_scale;        // KHR_texture_transform scale
uniform float u_tex_rotation;    // KHR_texture_transform rotation, radians
uniform vec3 u_light_dir;        // KHR_lights_punctual directional light
uniform vec3 u_light_intensity;  // light_color * intensity for directional light
uniform vec3 u_fill_light_pos0;  // viewer-side or KHR punctual fill light
uniform vec3 u_fill_light_color0;
uniform float u_fill_light_range0;
uniform vec3 u_fill_light_pos1;
uniform vec3 u_fill_light_color1;
uniform float u_fill_light_range1;
uniform int u_screen_light_enabled;
uniform vec3 u_screen_light_pos;
uniform vec3 u_screen_light_normal;
uniform vec3 u_screen_light_right;
uniform vec3 u_screen_light_up;
uniform vec2 u_screen_light_half_size;
uniform vec3 u_screen_light_color;
uniform sampler2D u_screen_light_tex;
uniform float u_screen_light_intensity;
uniform float u_env_exposure;
uniform float u_env_gamma;
uniform float u_emissive_strength;
uniform int u_shading_mode;       // 0=PBR, 1=preview diffuse
uniform int u_foliage_mode;       // 1=use preview-like two-sided foliage lighting

const float PI = 3.14159265359;

vec2 uvForTexCoord(int texcoord) {
    return (texcoord == 1) ? v_uv1 : v_uv;
}
vec3 gltfSrgbToLinear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + vec3(0.055)) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 gltfToneMap(vec3 linearColor) {
    linearColor = max(linearColor, vec3(0.0));
    return linearColor / (linearColor + vec3(1.0));
}

vec3 gltfLinearToOutput(vec3 linearColor, float gamma) {
    return pow(clamp(gltfToneMap(linearColor), 0.0, 1.0), vec3(1.0 / max(gamma, 0.001)));
}

// Fresnel-Schlick
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// GGX / Trowbridge-Reitz normal distribution
float DistributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith GGX geometry (Schlick)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

vec3 pbrLight(vec3 N, vec3 V, vec3 baseColor, float metallic, float roughness, vec3 L, vec3 lightColor, float attenuation) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0 || attenuation <= 0.0 || length(lightColor) <= 0.001) {
        return vec3(0.0);
    }

    vec3 H = normalize(L + V);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    float D = DistributionGGX(NdotH, roughness);
    float G = GeometrySmith(NdotV, NdotL, roughness);
    vec3 F = fresnelSchlick(VdotH, F0);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * baseColor / PI;
    return (diffuse + specular) * lightColor * NdotL * attenuation;
}

float softRangeAttenuation(float dist, float rangeMeters) {
    float r = max(rangeMeters, 0.001);
    float x = dist / r;
    return 1.0 / (1.0 + x * x * 4.0);
}

void main() {
    // KHR_texture_transform: scale, rotate, then offset on the selected base-color UV set.
    vec2 src_base_uv = uvForTexCoord(u_base_texcoord);
    vec2 scaled_uv = src_base_uv * u_tex_scale;
    float c = cos(u_tex_rotation);
    float s = sin(u_tex_rotation);
    vec2 t_uv = vec2(
        c * scaled_uv.x - s * scaled_uv.y,
        s * scaled_uv.x + c * scaled_uv.y
    ) + u_tex_offset;
    vec2 base_uv = t_uv;
    float texAlpha = (u_use_texture == 1) ? texture(u_tex, base_uv).a : 1.0;
    float materialAlpha = clamp(texAlpha * u_base_alpha, 0.0, 1.0);

    // alphaMode MASK: early discard (glTF spec 3.9.4)
    if (u_alpha_mode == 1) {
        if (materialAlpha < u_alpha_cutoff) discard;
    }

    if (!gl_FrontFacing && u_double_sided == 0) discard;
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) N = -N;

    // Normal map perturbation (glTF MikkTSpace tangent)
    if (u_use_normal_tex == 1) {
        vec3 nm = texture(u_normal_tex, uvForTexCoord(u_normal_texcoord)).rgb * 2.0 - 1.0;
        nm.xy *= u_normal_scale;
        nm = normalize(nm);
        // Use TANGENT attribute if available, else Gram-Schmidt fallback
        vec3 T = length(v_tangent) > 0.001 ? normalize(v_tangent) : normalize(cross(vec3(0.0, 1.0, 0.0), N));
        vec3 B = normalize(cross(N, T)) * v_bitangent_sign;
        N = normalize(T * nm.x + B * nm.y + N * nm.z);
    }

    vec3 baseColor;
    if (u_use_texture == 1) {
        baseColor = gltfSrgbToLinear(texture(u_tex, base_uv).rgb) * u_base_color_factor;
    } else {
        baseColor = u_base_color_factor;
    }

    if (u_shading_mode == 1) {
        vec3 L_preview = normalize(u_camera_pos + vec3(0.0, 0.2, 0.0) - v_position);
        float diff_preview = max(abs(dot(N, L_preview)), 0.12);
        vec3 color_preview = baseColor * (u_ambient_color + u_light_color * diff_preview) * u_env_exposure;
        float alpha = (u_alpha_mode == 2) ? materialAlpha : 1.0;
        fragColor = vec4(gltfLinearToOutput(color_preview, u_env_gamma), alpha);
        return;
    }

    if (u_foliage_mode == 1) {
        vec3 L_foliage = normalize(u_camera_pos + vec3(0.0, 0.2, 0.0) - v_position);
        float diff_foliage = max(abs(dot(N, L_foliage)), 0.12);
        vec3 color_foliage = baseColor * (u_ambient_color + u_light_color * diff_foliage) * u_env_exposure;
        float alpha = (u_alpha_mode == 2) ? materialAlpha : 1.0;
        fragColor = vec4(gltfLinearToOutput(color_foliage, u_env_gamma), alpha);
        return;
    }

    float metallic = clamp(u_metallic, 0.0, 1.0);
    float roughness = clamp(u_roughness, 0.04, 1.0);
    // metallicRoughnessTexture: B=metallic, G=roughness (glTF spec 3.9.4)
    if (u_use_mr_tex == 1) {
        vec3 mr = texture(u_mr_tex, uvForTexCoord(u_mr_texcoord)).rgb;
        roughness = clamp(roughness * mr.g, 0.04, 1.0);
        metallic = clamp(metallic * mr.b, 0.0, 1.0);
    }

    vec3 light_pos = u_camera_pos + vec3(0.0, 0.05, 0.0);
    vec3 L = normalize(light_pos - v_position);
    vec3 V = normalize(u_camera_pos - v_position);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // PBR: dielectric F0 = 0.04, metals use baseColor as F0
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    // Cook-Torrance specular BRDF
    float D = DistributionGGX(NdotH, roughness);
    float G = GeometrySmith(NdotV, NdotL, roughness);
    vec3 F = fresnelSchlick(VdotH, F0);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    // Diffuse: dielectrics scatter, metals have zero diffuse
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * baseColor / PI;

    // KHR_materials_unlit: skip all lighting, output baseColor directly
    if (u_unlit == 1) {
        float alpha = (u_alpha_mode == 2) ? materialAlpha : 1.0;
        fragColor = vec4(gltfLinearToOutput(baseColor * u_env_exposure, u_env_gamma), alpha);
        return;
    }

    // Head-lamp point light
    vec3 Lo = (diffuse + specular) * u_light_color * NdotL;

    // Controller-only top fill. Environment programs keep this at zero.
    vec3 top_light_pos = u_camera_pos + vec3(0.0, 0.45, -0.18);
    vec3 top_vec = top_light_pos - v_position;
    float top_dist = length(top_vec);
    if (top_dist > 0.001 && u_top_light_intensity > 0.0) {
        vec3 L_top = top_vec / top_dist;
        float top_facing = max(dot(N, L_top), 0.0);
        float top_fill = pow(top_facing, 1.25) * smoothstep(-0.20, 0.65, dot(N, V));
        Lo += diffuse * vec3(0.95, 0.97, 1.0) * (u_top_light_intensity * top_fill) * PI;
    }

    // Directional light (KHR_lights_punctual)
    if (length(u_light_intensity) > 0.001) {
        float NdotL_d = max(dot(N, -u_light_dir), 0.0);
        vec3 H_d = normalize(-u_light_dir + V);
        float D_d = DistributionGGX(max(dot(N, H_d), 0.0), roughness);
        float G_d = GeometrySmith(NdotV, NdotL_d, roughness);
        vec3 F_d = fresnelSchlick(max(dot(V, H_d), 0.0), F0);
        vec3 s_d = (D_d * G_d * F_d) / max(4.0 * NdotV * NdotL_d, 0.001);
        vec3 kD_d = (vec3(1.0) - F_d) * (1.0 - metallic);
        vec3 d_d = kD_d * baseColor / PI;
        Lo += (d_d + s_d) * u_light_intensity * NdotL_d;
    }

    vec3 toFill0 = u_fill_light_pos0 - v_position;
    float fillDist0 = length(toFill0);
    if (fillDist0 > 0.001) {
        Lo += pbrLight(N, V, baseColor, metallic, roughness,
                       toFill0 / fillDist0,
                       u_fill_light_color0,
                       softRangeAttenuation(fillDist0, u_fill_light_range0));
    }

    vec3 toFill1 = u_fill_light_pos1 - v_position;
    float fillDist1 = length(toFill1);
    if (fillDist1 > 0.001) {
        Lo += pbrLight(N, V, baseColor, metallic, roughness,
                       toFill1 / fillDist1,
                       u_fill_light_color1,
                       softRangeAttenuation(fillDist1, u_fill_light_range1));
    }

    // Cinema bias light from the virtual screen.
    if (u_screen_light_enabled == 1) {
        vec3 S_to_P = v_position - u_screen_light_pos;
        float d = length(S_to_P);
        vec3 L_s = S_to_P / max(d, 0.001);
        float front = smoothstep(0.0, 0.3, dot(u_screen_light_normal, L_s));
        if (front > 0.0) {
            float NdotL_s = max(dot(N, -L_s), 0.0);
            if (NdotL_s > 0.0) {
                float half_diag = length(u_screen_light_half_size);
                float r0 = max(half_diag * 2.0, 0.50);
                float attn = (r0 * r0) / (d * d + r0 * r0);
                float area = 4.0 * u_screen_light_half_size.x * u_screen_light_half_size.y;
                float r_near = max(half_diag * 0.5, 0.10);
                float area_term = area / (PI * max(d * d, r_near * r_near));
                float halo_free = smoothstep(
                    max(half_diag * 0.35, 0.75),
                    max(half_diag * 0.95, 1.75),
                    d
                );
                vec2 screen_p = vec2(
                    dot(S_to_P, u_screen_light_right) / max(u_screen_light_half_size.x, 0.001),
                    dot(S_to_P, u_screen_light_up) / max(u_screen_light_half_size.y, 0.001)
                );
                vec2 grid_uv = clamp(screen_p * 0.5 + 0.5, vec2(0.0), vec2(1.0));
                vec3 screen_col = textureLod(u_screen_light_tex, vec2(grid_uv.x, 1.0 - grid_uv.y), 0.0).rgb;
                screen_col = mix(screen_col, u_screen_light_color, 0.18);
                float vertical_soft = 1.0 - 0.12 * smoothstep(0.35, 1.30, abs(screen_p.y));
                vec3 E_s = screen_col
                           * u_screen_light_intensity
                           * front * NdotL_s * attn * area_term * halo_free * vertical_soft;
                Lo += diffuse * E_s * PI;
            }
        }
    }

    // Ambient / baked lightmap
    float ao = 1.0;
    vec3 bakedLight = vec3(1.0);
    if (u_use_occlusion_tex == 1 && u_baked_lightmap == 0) {
        ao = mix(1.0, texture(u_occlusion_tex, uvForTexCoord(u_occlusion_texcoord)).r, u_occlusion_strength);
    } else if (u_use_occlusion_tex == 1 && u_baked_lightmap == 1) {
        bakedLight = mix(vec3(1.0), texture(u_occlusion_tex, uvForTexCoord(u_occlusion_texcoord)).rgb, u_occlusion_strength);
    }
    vec3 ambient = u_ambient_color * baseColor * ao;

    vec3 emissive = u_emissive_factor;
    if (u_use_emissive_tex == 1) {
        emissive *= gltfSrgbToLinear(texture(u_emissive_tex, uvForTexCoord(u_emissive_texcoord)).rgb);
    }
    emissive *= u_emissive_strength;

    vec3 color = ((Lo + ambient) * bakedLight + emissive) * u_env_exposure;
    color = gltfLinearToOutput(color, u_env_gamma);

    float alpha = (u_alpha_mode == 2) ? materialAlpha : 1.0;
    fragColor = vec4(color, alpha);
}
"""

# Fullscreen swizzle blit: copies an RGBA texture into a target that the
# compositor reads as BGRA.
_BLIT_FRAG = """
#version 330
uniform sampler2D u_src;
uniform int u_swap_rb;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(u_src, uv);
    fragColor = (u_swap_rb != 0) ? c.bgra : c;
}
"""

# Glow fragment shader: renders a soft glow outside a centered rectangle
_GLOW_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform vec2 u_screen_half;   // screen half-size in UV space
uniform vec3 u_glow_color;
uniform sampler2D u_glow_tex;
uniform int u_glow_use_tex;
uniform float u_glow_width;   // glow decay distance in UV space
uniform float u_glow_extent;  // maximum outward glow distance in UV space
uniform float u_glow_intensity;

vec3 glow_grid_color(vec2 p) {
    vec2 screen_uv = clamp((p - (vec2(0.5) - u_screen_half)) / max(u_screen_half * 2.0, vec2(0.001)), vec2(0.0), vec2(1.0));
    vec3 grid_color = (u_glow_use_tex == 1)
        ? textureLod(u_glow_tex, vec2(screen_uv.x, 1.0 - screen_uv.y), 0.0).rgb
        : u_glow_color;
    return mix(u_glow_color, grid_color, 0.86);
}

void main() {
    vec2 d = abs(uv - 0.5) - u_screen_half;
    vec2 edge = max(d, vec2(0.0));
    float dist = length(edge);
    if (dist <= 0.001) {
        discard;
    }

    float width = max(u_glow_width, 0.001);
    float extent = max(u_glow_extent, width * 1.5);
    float near_glow = exp(-dist / width);
    float far_glow = exp(-dist / (width * 2.8)) * 0.42;
    float wrap = 1.0 - smoothstep(extent * 0.76, extent, dist);
    float corner_soften = 1.0 - 0.18 * smoothstep(0.0, extent, min(edge.x, edge.y));
    float glow = (near_glow + far_glow) * wrap * corner_soften * u_glow_intensity;
    if (glow <= 0.0001) {
        discard;
    }
    glow = min(glow, 1.0);
    vec2 nearest_screen_uv = clamp(uv, vec2(0.5) - u_screen_half, vec2(0.5) + u_screen_half);
    vec3 glow_color = glow_grid_color(nearest_screen_uv);
    frag_color = vec4(glow_color * glow, glow);
}
"""

_FROSTED_VEIL_VERT = """
#version 330
in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;
out vec3 v_local;
uniform mat4 u_model;
uniform mat4 u_vp;
void main() {
    v_uv = in_uv;
    v_local = in_position;
    gl_Position = u_vp * u_model * vec4(in_position, 1.0);
}
"""

_FROSTED_GLOW_FRAG = """
#version 330
in vec2 v_uv;
in vec3 v_local;
out vec4 frag_color;
uniform sampler2D u_screen_tex;
uniform float u_edge_inset;
uniform float u_lod;
uniform float u_threshold;
uniform float u_intensity;
uniform float u_frost_alpha;
uniform float u_noise_scale;
uniform float u_time;
uniform float u_beam_softness;
uniform float u_frost_blend;
uniform float u_beam_thickness;
uniform float u_diffuse_scatter;
uniform vec4 u_source_crop;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float wall_endpoint_fade(vec2 uv, vec3 local_pos) {
    float horizontal_wall = step(abs(local_pos.x), abs(local_pos.y));
    float endpoint_dist = mix(min(uv.y, 1.0 - uv.y), min(uv.x, 1.0 - uv.x), horizontal_wall);
    return smoothstep(0.015, 0.12, endpoint_dist);
}

void main() {
    vec2 sample_uv = clamp(u_source_crop.xy + v_uv * u_source_crop.zw, vec2(0.0), vec2(1.0));
    float depth = clamp(v_local.z, 0.0, 1.0);
    float beam = exp(-depth / max(u_beam_softness, 0.001));
    beam = pow(max(beam, 0.0), 1.0 / max(u_beam_thickness, 0.1));

    float edge_dist = min(min(v_uv.x, 1.0 - v_uv.x), min(v_uv.y, 1.0 - v_uv.y));
    float edge = 1.0 - smoothstep(max(u_edge_inset, 0.0001), max(u_edge_inset, 0.0001) * 4.0, edge_dist);

    vec3 src = textureLod(u_screen_tex, sample_uv, max(u_lod, 0.0)).rgb;
    float n = hash12(floor((sample_uv + vec2(u_time * 0.011, -u_time * 0.007)) * u_noise_scale));
    float luma = dot(src, vec3(0.2126, 0.7152, 0.0722));
    float bright = smoothstep(u_threshold, 1.0, luma);
    float scatter = max(bright, luma * clamp(u_diffuse_scatter, 0.0, 2.0) * 0.35);
    float alpha = edge * wall_endpoint_fade(v_uv, v_local) * beam * scatter * u_frost_alpha * u_intensity * (0.82 + 0.30 * n);
    if (alpha <= 0.002) {
        discard;
    }
    vec3 frost = mix(src, vec3(luma), 0.28);
    frost = frost * (0.55 + u_frost_blend * 0.35) + src * bright * 0.35;
    alpha = min(alpha, 1.0);
    frag_color = vec4(frost * alpha, alpha);
}
"""

_FROSTED_VEIL_FRAG = """
#version 330
in vec2 v_uv;
in vec3 v_local;
out vec4 frag_color;
uniform sampler2D u_screen_tex;
uniform float u_edge_inset;
uniform float u_intensity;
uniform float u_frost_alpha;
uniform float u_beam_softness;
uniform float u_beam_thickness;
uniform vec4 u_source_crop;

float wall_endpoint_fade(vec2 uv, vec3 local_pos) {
    float horizontal_wall = step(abs(local_pos.x), abs(local_pos.y));
    float endpoint_dist = mix(min(uv.y, 1.0 - uv.y), min(uv.x, 1.0 - uv.x), horizontal_wall);
    return smoothstep(0.015, 0.12, endpoint_dist);
}

void main() {
    vec2 sample_uv = clamp(u_source_crop.xy + v_uv * u_source_crop.zw, vec2(0.0), vec2(1.0));
    float depth = clamp(v_local.z, 0.0, 1.0);
    float beam = exp(-depth / max(u_beam_softness, 0.001));
    beam = pow(max(beam, 0.0), 1.0 / max(u_beam_thickness, 0.1));

    float edge_dist = min(min(v_uv.x, 1.0 - v_uv.x), min(v_uv.y, 1.0 - v_uv.y));
    float edge = 1.0 - smoothstep(max(u_edge_inset, 0.0001), max(u_edge_inset, 0.0001) * 4.0, edge_dist);
    float alpha = edge * wall_endpoint_fade(v_uv, v_local) * beam * u_frost_alpha * u_intensity;
    if (alpha <= 0.002) {
        discard;
    }
    alpha = min(alpha, 1.0);
    vec3 src = textureLod(u_screen_tex, sample_uv, 0.0).rgb;
    frag_color = vec4(src * alpha, alpha);
}
"""

_GLOW_SHELL_VERT = """
#version 330
in vec3 in_position;
in vec2 in_uv;
out vec2 uv;
uniform mat4 u_vp;
uniform vec3 u_center;
uniform vec3 u_shell_scale;
void main() {
    uv = in_uv;
    vec3 world = u_center + in_position * u_shell_scale;
    gl_Position = u_vp * vec4(world, 1.0);
}
"""

_GLOW_SHELL_FRAG = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform vec3 u_glow_color;
uniform sampler2D u_glow_tex;
uniform int u_glow_use_tex;
uniform float u_glow_intensity;

vec3 sample_border_color(vec2 p) {
    if (u_glow_use_tex != 1) {
        return u_glow_color;
    }
    float x = clamp(p.x, 0.0, 1.0);
    float y = clamp(1.0 - p.y, 0.0, 1.0);
    vec3 top_col = textureLod(u_glow_tex, vec2(x, 0.055), 0.0).rgb;
    vec3 bottom_col = textureLod(u_glow_tex, vec2(x, 0.945), 0.0).rgb;
    vec3 left_col = textureLod(u_glow_tex, vec2(0.055, y), 0.0).rgb;
    vec3 right_col = textureLod(u_glow_tex, vec2(0.945, y), 0.0).rgb;
    float top_weight = smoothstep(0.50, 0.95, p.y);
    float bottom_weight = smoothstep(0.50, 0.95, 1.0 - p.y);
    float left_weight = smoothstep(0.35, 0.95, 1.0 - p.x);
    float right_weight = smoothstep(0.35, 0.95, p.x);
    vec3 border = (
        top_col * top_weight +
        bottom_col * bottom_weight +
        left_col * left_weight +
        right_col * right_weight
    ) / max(top_weight + bottom_weight + left_weight + right_weight, 0.001);
    return mix(u_glow_color, border, 0.90);
}

vec3 sample_region_reflection(vec2 p) {
    if (u_glow_use_tex != 1) {
        return u_glow_color;
    }
    vec2 grid = vec2(4.0, 3.0);
    vec2 q = (floor(clamp(p, vec2(0.0), vec2(0.999)) * grid) + vec2(0.5)) / grid;
    q.y = 1.0 - q.y;
    vec3 region = textureLod(u_glow_tex, q, 0.0).rgb;
    return mix(u_glow_color, region, 0.92);
}

void main() {
    float horiz = clamp(1.0 - abs(uv.x - 0.5) * 2.0, 0.0, 1.0);
    float vertical_core = smoothstep(0.02, 0.20, uv.y) * (1.0 - smoothstep(0.82, 0.98, uv.y));
    float vertical_edges = max(1.0 - smoothstep(0.12, 0.42, uv.y), smoothstep(0.58, 0.88, uv.y)) * 0.30;
    float vertical = max(vertical_core, vertical_edges);
    float front_focus = pow(horiz, 1.55);
    float band = 0.58 + 0.42 * sin(uv.y * 3.14159265);
    float wrap = 0.65 + 0.35 * smoothstep(0.18, 0.70, horiz);
    float glow = front_focus * vertical * band * wrap * u_glow_intensity;
    if (glow <= 0.0001) {
        discard;
    }
    glow = min(glow, 1.0);
    vec3 border_color = sample_border_color(uv);
    vec3 region_color = sample_region_reflection(vec2(uv.x, 0.5 + (uv.y - 0.5) * 0.35));
    float region_mix = 0.38 + 0.46 * smoothstep(0.12, 0.72, horiz);
    vec3 shell_color = mix(border_color, region_color, region_mix);
    frag_color = vec4(shell_color * glow, glow * 0.92);
}
"""
