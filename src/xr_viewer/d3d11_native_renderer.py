import ctypes
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from utils.cpu_warnings import describe_tensor, warn_cpu_fallback, warn_cpu_operation, warn_cpu_transfer
from .controller_lighting import CONTROLLER_HEAD_LIGHT_COLOR, CONTROLLER_TOP_LIGHT_INTENSITY
from .gltf import (
    D3D11_VERTEX_OFFSETS_BYTES,
    D3D11_VERTEX_STRIDE_BYTES,
    format_gltf_scene_summary,
    render_pass_from_primitive,
    sort_transparent_primitives,
)
from .material_contract import GLTF_MATERIAL_TEXTURE_BINDINGS
from .laser_geometry import build_laser_beam_vertices
from .laser_params import CURSOR_RING_INNER_RATIO, LASER_BASE_HALF_WIDTH_M, LASER_MAX_LENGTH_M, LASER_TIP_HALF_WIDTH_M


DXGI_FORMAT_R32G32_FLOAT = 16
DXGI_FORMAT_R32G32B32A32_FLOAT = 2
DXGI_FORMAT_D32_FLOAT = 40
DXGI_FORMAT_R8G8B8A8_UNORM = 28
DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = 29
DXGI_FORMAT_B8G8R8A8_UNORM = 87
DXGI_FORMAT_B8G8R8A8_UNORM_SRGB = 91
DXGI_FORMAT_R32_FLOAT = 41
DXGI_FORMAT_R32G32B32_FLOAT = 6
DXGI_FORMAT_R32_UINT = 42

D3D11_USAGE_DEFAULT = 0
D3D11_BIND_VERTEX_BUFFER = 0x1
D3D11_BIND_INDEX_BUFFER = 0x2
D3D11_BIND_CONSTANT_BUFFER = 0x4
D3D11_BIND_SHADER_RESOURCE = 0x8
D3D11_BIND_RENDER_TARGET = 0x20
D3D11_BIND_DEPTH_STENCIL = 0x40
D3D11_INPUT_PER_VERTEX_DATA = 0
D3D11_PRIMITIVE_TOPOLOGY_POINTLIST = 1
D3D11_PRIMITIVE_TOPOLOGY_LINELIST = 2
D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP = 3
D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST = 4
D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP = 5
D3D11_SDK_VERSION = 7
D3D11_FILL_SOLID = 3
D3D11_CULL_NONE = 1
D3D11_RTV_DIMENSION_TEXTURE2D = 4
D3D11_RTV_DIMENSION_TEXTURE2DARRAY = 5
D3D11_FILTER_MIN_MAG_MIP_LINEAR = 0x15
D3D11_TEXTURE_ADDRESS_WRAP = 1
D3D11_TEXTURE_ADDRESS_CLAMP = 3
D3D11_COMPARISON_NEVER = 1
D3D11_COMPARISON_LESS_EQUAL = 4
D3D11_DEPTH_WRITE_MASK_ZERO = 0
D3D11_DEPTH_WRITE_MASK_ALL = 1
D3D11_STENCIL_OP_KEEP = 1
D3D11_CLEAR_DEPTH = 0x1
D3D11_BLEND_ZERO = 1
D3D11_BLEND_ONE = 2
D3D11_BLEND_SRC_ALPHA = 5
D3D11_BLEND_INV_SRC_ALPHA = 6
D3D11_BLEND_OP_ADD = 1
D3D11_COLOR_WRITE_ENABLE_ALL = 0x0F


class DXGISampleDesc(ctypes.Structure):
    _fields_ = [
        ("Count", ctypes.c_uint),
        ("Quality", ctypes.c_uint),
    ]


class D3D11Texture2DDesc(ctypes.Structure):
    _fields_ = [
        ("Width", ctypes.c_uint),
        ("Height", ctypes.c_uint),
        ("MipLevels", ctypes.c_uint),
        ("ArraySize", ctypes.c_uint),
        ("Format", ctypes.c_uint),
        ("SampleDesc", DXGISampleDesc),
        ("Usage", ctypes.c_uint),
        ("BindFlags", ctypes.c_uint),
        ("CPUAccessFlags", ctypes.c_uint),
        ("MiscFlags", ctypes.c_uint),
    ]


class D3D11BufferDesc(ctypes.Structure):
    _fields_ = [
        ("ByteWidth", ctypes.c_uint),
        ("Usage", ctypes.c_uint),
        ("BindFlags", ctypes.c_uint),
        ("CPUAccessFlags", ctypes.c_uint),
        ("MiscFlags", ctypes.c_uint),
        ("StructureByteStride", ctypes.c_uint),
    ]


class D3D11SubresourceData(ctypes.Structure):
    _fields_ = [
        ("pSysMem", ctypes.c_void_p),
        ("SysMemPitch", ctypes.c_uint),
        ("SysMemSlicePitch", ctypes.c_uint),
    ]


class D3D11InputElementDesc(ctypes.Structure):
    _fields_ = [
        ("SemanticName", ctypes.c_char_p),
        ("SemanticIndex", ctypes.c_uint),
        ("Format", ctypes.c_uint),
        ("InputSlot", ctypes.c_uint),
        ("AlignedByteOffset", ctypes.c_uint),
        ("InputSlotClass", ctypes.c_uint),
        ("InstanceDataStepRate", ctypes.c_uint),
    ]


class D3D11Viewport(ctypes.Structure):
    _fields_ = [
        ("TopLeftX", ctypes.c_float),
        ("TopLeftY", ctypes.c_float),
        ("Width", ctypes.c_float),
        ("Height", ctypes.c_float),
        ("MinDepth", ctypes.c_float),
        ("MaxDepth", ctypes.c_float),
    ]


class D3D11RenderTargetViewDesc(ctypes.Structure):
    _fields_ = [
        ("Format", ctypes.c_uint),
        ("ViewDimension", ctypes.c_uint),
        ("MipSlice", ctypes.c_uint),
        ("FirstArraySlice", ctypes.c_uint),
        ("ArraySize", ctypes.c_uint),
    ]


class D3D11DepthStencilOpDesc(ctypes.Structure):
    _fields_ = [
        ("StencilFailOp", ctypes.c_uint),
        ("StencilDepthFailOp", ctypes.c_uint),
        ("StencilPassOp", ctypes.c_uint),
        ("StencilFunc", ctypes.c_uint),
    ]


class D3D11DepthStencilDesc(ctypes.Structure):
    _fields_ = [
        ("DepthEnable", ctypes.c_int),
        ("DepthWriteMask", ctypes.c_uint),
        ("DepthFunc", ctypes.c_uint),
        ("StencilEnable", ctypes.c_int),
        ("StencilReadMask", ctypes.c_ubyte),
        ("StencilWriteMask", ctypes.c_ubyte),
        ("FrontFace", D3D11DepthStencilOpDesc),
        ("BackFace", D3D11DepthStencilOpDesc),
    ]


class D3D11SamplerDesc(ctypes.Structure):
    _fields_ = [
        ("Filter", ctypes.c_uint),
        ("AddressU", ctypes.c_uint),
        ("AddressV", ctypes.c_uint),
        ("AddressW", ctypes.c_uint),
        ("MipLODBias", ctypes.c_float),
        ("MaxAnisotropy", ctypes.c_uint),
        ("ComparisonFunc", ctypes.c_uint),
        ("BorderColor", ctypes.c_float * 4),
        ("MinLOD", ctypes.c_float),
        ("MaxLOD", ctypes.c_float),
    ]


class D3D11RasterizerDesc(ctypes.Structure):
    _fields_ = [
        ("FillMode", ctypes.c_uint),
        ("CullMode", ctypes.c_uint),
        ("FrontCounterClockwise", ctypes.c_int),
        ("DepthBias", ctypes.c_int),
        ("DepthBiasClamp", ctypes.c_float),
        ("SlopeScaledDepthBias", ctypes.c_float),
        ("DepthClipEnable", ctypes.c_int),
        ("ScissorEnable", ctypes.c_int),
        ("MultisampleEnable", ctypes.c_int),
        ("AntialiasedLineEnable", ctypes.c_int),
    ]


class D3D11RenderTargetBlendDesc(ctypes.Structure):
    _fields_ = [
        ("BlendEnable", ctypes.c_int),
        ("SrcBlend", ctypes.c_uint),
        ("DestBlend", ctypes.c_uint),
        ("BlendOp", ctypes.c_uint),
        ("SrcBlendAlpha", ctypes.c_uint),
        ("DestBlendAlpha", ctypes.c_uint),
        ("BlendOpAlpha", ctypes.c_uint),
        ("RenderTargetWriteMask", ctypes.c_ubyte),
    ]


class D3D11BlendDesc(ctypes.Structure):
    _fields_ = [
        ("AlphaToCoverageEnable", ctypes.c_int),
        ("IndependentBlendEnable", ctypes.c_int),
        ("RenderTarget", D3D11RenderTargetBlendDesc * 8),
    ]


def _ptr_value(ptr):
    if ptr is None:
        return 0
    if isinstance(ptr, int):
        return ptr
    if hasattr(ptr, "value"):
        return ptr.value
    return ctypes.cast(ptr, ctypes.c_void_p).value


def _com_fn(obj, index, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn_ptr = ctypes.cast(
        vtbl + index * ctypes.sizeof(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ).contents.value
    return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(fn_ptr)


def _release(ptr):
    if not ptr:
        return
    try:
        fn = _com_fn(ptr, 2, ctypes.c_ulong)
        fn(_ptr_value(ptr))
    except Exception:
        pass


def _blob_ptr(blob):
    fn = _com_fn(blob, 3, ctypes.c_void_p)
    return fn(_ptr_value(blob))


def _blob_size(blob):
    fn = _com_fn(blob, 4, ctypes.c_size_t)
    return fn(_ptr_value(blob))


def _compile_shader(source, entry, target):
    compiler = None
    for dll in ("d3dcompiler_47", "d3dcompiler_43"):
        try:
            compiler = ctypes.WinDLL(dll)
            break
        except OSError:
            continue
    if compiler is None:
        raise RuntimeError("d3dcompiler_47/d3dcompiler_43 not found")

    code = source.encode("utf-8")
    entry_b = entry.encode("ascii")
    target_b = target.encode("ascii")
    blob = ctypes.c_void_p()
    err_blob = ctypes.c_void_p()
    hr = compiler.D3DCompile(
        code,
        len(code),
        None,
        None,
        None,
        entry_b,
        target_b,
        0,
        0,
        ctypes.byref(blob),
        ctypes.byref(err_blob),
    )
    if hr != 0:
        msg = f"hr=0x{hr & 0xFFFFFFFF:08x}"
        if err_blob:
            try:
                msg = ctypes.string_at(_blob_ptr(err_blob), _blob_size(err_blob)).decode("utf-8", "replace")
            finally:
                _release(err_blob)
        raise RuntimeError(f"D3DCompile failed for {entry}/{target}: {msg}")
    if err_blob:
        _release(err_blob)
    return blob


def _check_hr(hr, label):
    if hr != 0:
        raise RuntimeError(f"{label} failed: hr=0x{hr & 0xFFFFFFFF:08x}")


def _hr_hex(hr):
    return f"0x{hr & 0xFFFFFFFF:08x}"


class CUDART_D3D11:
    CUDA_GRAPHICS_REGISTER_FLAGS_NONE = 0
    CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
    CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

    def __init__(self, torch_module, d3d11_device, device_id=0):
        torch_dir = os.path.dirname(torch_module.__file__)
        site_packages = os.path.dirname(torch_dir)
        candidates = [
            os.path.join(torch_dir, "lib"),
            os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
            os.path.join(site_packages, "nvidia", "cuda_runtime", "lib"),
        ]
        cudart_path = None
        for lib_dir in candidates:
            if not os.path.exists(lib_dir):
                continue
            for name in os.listdir(lib_dir):
                if sys.platform == "win32":
                    if name.startswith("cudart64") and name.endswith(".dll"):
                        cudart_path = os.path.join(lib_dir, name)
                        break
                elif name.startswith("libcudart") and ".so" in name:
                    cudart_path = os.path.join(lib_dir, name)
                    break
            if cudart_path:
                break
        if not cudart_path:
            raise RuntimeError("Could not find CUDA runtime library for D3D11 interop")

        self.lib = ctypes.WinDLL(cudart_path) if sys.platform == "win32" else ctypes.CDLL(cudart_path)
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaGetLastError.argtypes = []
        self.lib.cudaGetLastError.restype = ctypes.c_int
        self.lib.cudaGraphicsD3D11RegisterResource.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint
        ]
        self.lib.cudaGraphicsD3D11RegisterResource.restype = ctypes.c_int
        self.lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphicsUnregisterResource.restype = ctypes.c_int
        self.lib.cudaGraphicsMapResources.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        ]
        self.lib.cudaGraphicsMapResources.restype = ctypes.c_int
        self.lib.cudaGraphicsUnmapResources.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        ]
        self.lib.cudaGraphicsUnmapResources.restype = ctypes.c_int
        self.lib.cudaGraphicsSubResourceGetMappedArray.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint
        ]
        self.lib.cudaGraphicsSubResourceGetMappedArray.restype = ctypes.c_int
        self.lib.cudaMemcpy2DToArrayAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
        ]
        self.lib.cudaMemcpy2DToArrayAsync.restype = ctypes.c_int
        res = self.lib.cudaSetDevice(int(device_id))
        if res != 0:
            raise RuntimeError(f"cudaSetDevice failed: {res}")

    def clear_last_error(self):
        try:
            self.lib.cudaGetLastError()
        except Exception:
            pass

    def register_texture(self, texture_ptr):
        resource = ctypes.c_void_p()
        res = self.lib.cudaGraphicsD3D11RegisterResource(
            ctypes.byref(resource),
            ctypes.c_void_p(_ptr_value(texture_ptr)),
            self.CUDA_GRAPHICS_REGISTER_FLAGS_NONE,
        )
        if res != 0:
            self.clear_last_error()
            raise RuntimeError(f"cudaGraphicsD3D11RegisterResource failed: {res}")
        return resource

    def unregister_resource(self, resource):
        if resource:
            self.lib.cudaGraphicsUnregisterResource(resource)

    def copy_tensor_to_texture(self, resource, src_ptr, src_pitch, copy_width_bytes, height, stream=0):
        stream_ptr = ctypes.c_void_p(stream) if stream else None
        res = self.lib.cudaGraphicsMapResources(1, ctypes.byref(resource), stream_ptr)
        if res != 0:
            self.clear_last_error()
            raise RuntimeError(f"cudaGraphicsMapResources failed: {res}")
        try:
            array = ctypes.c_void_p()
            res = self.lib.cudaGraphicsSubResourceGetMappedArray(
                ctypes.byref(array), resource, 0, 0
            )
            if res != 0:
                self.clear_last_error()
                raise RuntimeError(f"cudaGraphicsSubResourceGetMappedArray failed: {res}")
            res = self.lib.cudaMemcpy2DToArrayAsync(
                array,
                0,
                0,
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(src_pitch),
                ctypes.c_size_t(copy_width_bytes),
                ctypes.c_size_t(height),
                self.CUDA_MEMCPY_DEVICE_TO_DEVICE,
                stream_ptr,
            )
            if res != 0:
                self.clear_last_error()
                raise RuntimeError(f"cudaMemcpy2DToArrayAsync failed: {res}")
        finally:
            unmap_res = self.lib.cudaGraphicsUnmapResources(1, ctypes.byref(resource), stream_ptr)
            if unmap_res != 0:
                self.clear_last_error()
                raise RuntimeError(f"cudaGraphicsUnmapResources failed: {unmap_res}")


HLSL_SOURCE = r"""
Texture2D texColor : register(t0);
Texture2D texDepth : register(t1);
SamplerState sampLinear : register(s0);

cbuffer Params : register(b0)
{
    float4 row0;
    float4 row1;
    float4 row2;
    float4 row3;
    float4 params;
};

#define parallaxOffset params.x
#define depthStrength params.y
#define convergence params.z
#define roll params.w

struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

VSOut vs_main(uint vertexId : SV_VertexID)
{
    static const float2 pos[4] = {
        float2(-1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0)
    };
    static const float2 uv[4] = {
        float2(0.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0)
    };
    VSOut output;
    float4 localPos = float4(pos[vertexId], 0.0, 1.0);
#if D2S_SPACE_MODE == 1
    output.pos = float4(pos[vertexId] * float2(0.35, 0.2), 0.5, 1.0);
#else
    output.pos = float4(
        dot(row0, localPos),
        dot(row1, localPos),
        dot(row2, localPos),
        dot(row3, localPos)
    );
#endif
    output.uv = uv[vertexId];
    return output;
}

float4 ps_main(VSOut input) : SV_TARGET
{
    float2 uv = float2(input.uv.x, 1.0 - input.uv.y);

#if D2S_SHADER_MODE == 0
    return float4(0.05, 0.10, 0.16, 1.0);
#elif D2S_SHADER_MODE == 1
    return float4(texColor.Sample(sampLinear, uv).rgb, 1.0);
#else
    if (abs(depthStrength) < 0.000001 || abs(parallaxOffset) < 0.000001) {
        return float4(texColor.Sample(sampLinear, uv).rgb, 1.0);
    }

    uint texW = 1;
    uint texH = 1;
    texColor.GetDimensions(texW, texH);
    float2 pixelSize = 1.0 / float2(max(float(texW), 1.0), max(float(texH), 1.0));
    float2 parDir = normalize(float2(cos(roll), sin(roll)));
    float2 signedParDir = parDir * (parallaxOffset >= 0.0 ? 1.0 : -1.0);

    float2 dsDir = signedParDir * pixelSize * 1.5;
    float d0 = texDepth.Sample(sampLinear, uv).r;
    float dm = texDepth.Sample(sampLinear, uv - dsDir).r;
    float dp = texDepth.Sample(sampLinear, uv + dsDir).r;
    float depth = saturate(d0 * 0.5 + dm * 0.25 + dp * 0.25);
    float depthResponse = depth - convergence;

    float edgeFalloff = smoothstep(0.0, 0.05, uv.x) * (1.0 - smoothstep(0.95, 1.0, uv.x));
    float shift = depthResponse * parallaxOffset * depthStrength * edgeFalloff;
    float2 shiftedUv = uv - parDir * shift;
    if (shiftedUv.x < 0.0 || shiftedUv.x > 1.0 || shiftedUv.y < 0.0 || shiftedUv.y > 1.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float4 color = texColor.Sample(sampLinear, shiftedUv);
    float2 step2 = signedParDir * pixelSize * 2.0;
    float jump = abs(texDepth.Sample(sampLinear, uv - step2).r - texDepth.Sample(sampLinear, uv + step2).r);
    float conf = smoothstep(0.04, 0.10, jump);
    if (conf > 0.001) {
        float4 accum = float4(0.0, 0.0, 0.0, 0.0);
        float totalWeight = 0.0;
        float centerInv = -depth;
        float2 sweep = signedParDir * pixelSize.x * (parallaxOffset > 0.0 ? -1.0 : 1.0);
        [loop]
        for (int i = 1; i <= 12; ++i) {
            float2 sampleUv = uv + sweep * float(i);
            if (sampleUv.x < 0.0 || sampleUv.x > 1.0 || sampleUv.y < 0.0 || sampleUv.y > 1.0) {
                continue;
            }
            float sampleInv = 1.0 - texDepth.Sample(sampLinear, sampleUv).r;
            if (sampleInv > centerInv + 0.012) {
                float weight = exp(-float(i) * 0.15) * (1.0 + (sampleInv - centerInv) * 10.0);
                accum += texColor.Sample(sampLinear, sampleUv) * weight;
                totalWeight += weight;
                if (totalWeight > 5.0) {
                    break;
                }
            }
        }
        if (totalWeight > 0.01) {
            color = lerp(color, accum / totalWeight, conf);
        }
    }

    return float4(color.rgb, 1.0);
#endif
}
"""


BACKGROUND_HLSL_SOURCE = r"""
Texture2D texBackground : register(t0);
SamplerState sampLinear : register(s0);

cbuffer BgParams : register(b0)
{
    float4 invProjRow0;
    float4 invProjRow1;
    float4 invProjRow2;
    float4 invProjRow3;
    float4 invViewRow0;
    float4 invViewRow1;
    float4 invViewRow2;
    float4 invViewRow3;
    float4 bgParams;
};

#define yawOffset bgParams.x
#define exposure bgParams.y
#define flipY bgParams.z

struct VSOut {
    float4 pos : SV_POSITION;
    float2 ndc : TEXCOORD0;
};

VSOut vs_main(uint vertexId : SV_VertexID)
{
    static const float2 pos[4] = {
        float2(-1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0)
    };
    VSOut output;
    output.pos = float4(pos[vertexId], 1.0, 1.0);
    output.ndc = pos[vertexId];
    return output;
}

float4 ps_main(VSOut input) : SV_TARGET
{
    static const float PI = 3.14159265358979323846;
    float4 clip = float4(input.ndc, 1.0, 1.0);
    float4 viewH = float4(
        dot(invProjRow0, clip),
        dot(invProjRow1, clip),
        dot(invProjRow2, clip),
        dot(invProjRow3, clip)
    );
    float3 viewDir = normalize(viewH.xyz / max(abs(viewH.w), 1e-6));
    float4 viewDir4 = float4(viewDir, 0.0);
    float3 dir = normalize(float3(
        dot(invViewRow0, viewDir4),
        dot(invViewRow1, viewDir4),
        dot(invViewRow2, viewDir4)
    ));
    float u = atan2(dir.x, -dir.z) / (2.0 * PI) + 0.5 + yawOffset;
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
    if (flipY != 0.0) {
        v = 1.0 - v;
    }
    float3 color = texBackground.Sample(sampLinear, float2(frac(u), clamp(v, 0.0, 1.0))).rgb;
    return float4(saturate(color * max(exposure, 0.0)), 1.0);
}
"""


CONTROLLER_HLSL_SOURCE = r"""
Texture2D texBase : register(t0);
Texture2D texEnv : register(t1);
Texture2D texScreenLight : register(t2);
Texture2D texNormal : register(t3);
Texture2D texOcclusion : register(t4);
Texture2D texMR : register(t5);
Texture2D texEmissive : register(t6);
SamplerState sampLinear : register(s0);

cbuffer ControllerParams : register(b0)
{
    float4 mvpRow0;
    float4 mvpRow1;
    float4 mvpRow2;
    float4 mvpRow3;
    float4 modelRow0;
    float4 modelRow1;
    float4 modelRow2;
    float4 modelRow3;
    float4 normalRow0;
    float4 normalRow1;
    float4 normalRow2;
    float4 baseColorAlpha;
    float4 materialParams;
    float4 cameraPosUseEnv;
    float4 lightParams;
    float4 screenLightPosEnabled;
    float4 screenLightNormalIntensity;
    float4 screenLightRightHalfX;
    float4 screenLightUpHalfY;
    float4 materialParams2;
    float4 emissiveFactorUse;
    float4 texTransform0;
    float4 texCoordParams;
    float4 texCoordParams2;
    float4 lightColor;
    float4 ambientColor;
    float4 directionalLight;
    float4 directionalColor;
    float4 fillLightPosRange0;
    float4 fillLightColor0;
    float4 fillLightPosRange1;
    float4 fillLightColor1;
};

#define roughness materialParams.x
#define metallic materialParams.y
#define useTexture materialParams.z
#define alphaMode materialParams.w
#define useEnv cameraPosUseEnv.w
#define envIntensity lightParams.x
#define alphaCutoff lightParams.y
#define unlit lightParams.z
#define doubleSided lightParams.w
#define screenLightEnabled screenLightPosEnabled.w
#define screenLightIntensity screenLightNormalIntensity.w
#define normalScale materialParams2.x
#define occlusionStrength materialParams2.y
#define useNormal materialParams2.z
#define useOcclusion materialParams2.w
#define texOffset texTransform0.xy
#define texScale texTransform0.zw
#define texRotation texCoordParams.x
#define baseTexcoord texCoordParams.y
#define normalTexcoord texCoordParams.z
#define occlusionTexcoord texCoordParams.w
#define mrTexcoord texCoordParams2.x
#define emissiveTexcoord texCoordParams2.y
#define useMR texCoordParams2.z
#define useEmissive texCoordParams2.w

struct VSIn {
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv0 : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
};

struct VSOut {
    float4 pos : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv0 : TEXCOORD2;
    float2 uv1 : TEXCOORD3;
};

float4 mulRows(float4 row0, float4 row1, float4 row2, float4 row3, float4 v)
{
    return float4(dot(row0, v), dot(row1, v), dot(row2, v), dot(row3, v));
}

VSOut vs_main(VSIn input)
{
    VSOut output;
    float4 localPos = float4(input.pos, 1.0);
    float4 world = mulRows(modelRow0, modelRow1, modelRow2, modelRow3, localPos);
    output.pos = mulRows(mvpRow0, mvpRow1, mvpRow2, mvpRow3, localPos);
    output.worldPos = world.xyz;
    output.normal = normalize(float3(
        dot(normalRow0.xyz, input.normal),
        dot(normalRow1.xyz, input.normal),
        dot(normalRow2.xyz, input.normal)
    ));
    output.uv0 = input.uv0;
    output.uv1 = input.uv1;
    return output;
}

float2 uvForTexCoord(VSOut input, float texcoord)
{
    return texcoord > 0.5 ? input.uv1 : input.uv0;
}

float2 transformedBaseUv(VSOut input)
{
    float2 uv = uvForTexCoord(input, baseTexcoord);
    float2 scaled = uv * texScale;
    float c = cos(texRotation);
    float s = sin(texRotation);
    return float2(c * scaled.x - s * scaled.y, s * scaled.x + c * scaled.y) + texOffset;
}

float2 envSampleUv(float3 dir)
{
    dir = normalize(dir);
    float u = atan2(dir.x, dir.z) / 6.28318530718 + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) / 3.14159265 + 0.5;
    return float2(frac(u), 1.0 - v);
}

float srgbChannelToLinear(float c)
{
    c = saturate(c);
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float3 gltfSrgbToLinear(float3 c)
{
    return float3(srgbChannelToLinear(c.r), srgbChannelToLinear(c.g), srgbChannelToLinear(c.b));
}

float3 gltfToneMap(float3 linearColor)
{
    linearColor = max(linearColor, float3(0.0, 0.0, 0.0));
    return linearColor / (linearColor + float3(1.0, 1.0, 1.0));
}

float3 gltfLinearToOutput(float3 linearColor)
{
    return pow(saturate(gltfToneMap(linearColor)), float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float DistributionGGX(float NdotH, float rough)
{
    float a = rough * rough;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * denom * denom, 0.001);
}

float GeometrySchlickGGX(float NdotV, float rough)
{
    float r = rough + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / max(NdotV * (1.0 - k) + k, 0.001);
}

float GeometrySmith(float NdotV, float NdotL, float rough)
{
    return GeometrySchlickGGX(NdotV, rough) * GeometrySchlickGGX(NdotL, rough);
}

float3 pbrLight(float3 n, float3 v, float3 baseColor, float metal, float rough, float3 l, float3 lightColor, float attenuation)
{
    float NdotL = max(dot(n, l), 0.0);
    if (NdotL <= 0.0 || attenuation <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 h = normalize(l + v);
    float NdotV = max(dot(n, v), 0.001);
    float NdotH = max(dot(n, h), 0.0);
    float VdotH = max(dot(v, h), 0.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metal);
    float D = DistributionGGX(NdotH, rough);
    float G = GeometrySmith(NdotV, NdotL, rough);
    float3 F = fresnelSchlick(VdotH, F0);
    float3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metal);
    float3 diffuse = kD * baseColor / 3.14159265;
    return (diffuse + specular) * lightColor * NdotL * attenuation;
}

float4 ps_main(VSOut input, bool isFrontFace : SV_IsFrontFace) : SV_TARGET
{
    float2 baseUv = transformedBaseUv(input);
    float4 texel = useTexture > 0.5 ? texBase.Sample(sampLinear, baseUv) : float4(1.0, 1.0, 1.0, 1.0);
    texel.rgb = gltfSrgbToLinear(texel.rgb);
    float alpha = saturate(texel.a * baseColorAlpha.a);
    if (alphaMode > 0.5 && alphaMode < 1.5 && alpha < alphaCutoff) {
        discard;
    }
    float3 baseColor = texel.rgb * baseColorAlpha.rgb;
    if (!isFrontFace && doubleSided < 0.5) {
        discard;
    }
    float3 n = normalize(input.normal);
    if (!isFrontFace) {
        n = -n;
    }
    float3 v = normalize(cameraPosUseEnv.xyz - input.worldPos);
    if (useNormal > 0.5) {
        float3 nm = texNormal.Sample(sampLinear, uvForTexCoord(input, normalTexcoord)).rgb * 2.0 - 1.0;
        nm.xy *= normalScale;
        float3 t = normalize(cross(abs(n.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0), n));
        float3 b = normalize(cross(n, t));
        n = normalize(t * nm.x + b * nm.y + n * nm.z);
    }
    float metal = saturate(metallic);
    float rough = clamp(roughness, 0.04, 1.0);
    if (useMR > 0.5) {
        float3 mr = texMR.Sample(sampLinear, uvForTexCoord(input, mrTexcoord)).rgb;
        rough = clamp(rough * mr.g, 0.04, 1.0);
        metal = saturate(metal * mr.b);
    }
    float3 color = baseColor * ambientColor.rgb;
    float3 topLightPos = cameraPosUseEnv.xyz + float3(0.0, 0.45, -0.18);
    float3 topLightDir = normalize(topLightPos - input.worldPos);
    color += pbrLight(n, v, baseColor, metal, rough, topLightDir, lightColor.rgb * lightColor.w, 1.0);
    color += pbrLight(n, v, baseColor, metal, rough, -normalize(directionalLight.xyz), directionalColor.rgb, 1.0);
    float3 fill0 = fillLightPosRange0.xyz - input.worldPos;
    color += pbrLight(n, v, baseColor, metal, rough, normalize(fill0), fillLightColor0.rgb, 1.0 / (1.0 + dot(fill0, fill0) / max(fillLightPosRange0.w * fillLightPosRange0.w, 0.001)));
    float3 fill1 = fillLightPosRange1.xyz - input.worldPos;
    color += pbrLight(n, v, baseColor, metal, rough, normalize(fill1), fillLightColor1.rgb, 1.0 / (1.0 + dot(fill1, fill1) / max(fillLightPosRange1.w * fillLightPosRange1.w, 0.001)));
    if (useEnv > 0.5) {
        float3 r = reflect(-v, n);
        float3 envSpec = texEnv.SampleLevel(sampLinear, envSampleUv(r), 3.0).rgb;
        float3 envDiff = texEnv.SampleLevel(sampLinear, envSampleUv(n), 5.0).rgb;
        float viewFacing = smoothstep(-0.25, 0.65, dot(n, v));
        color += baseColor * lerp(float3(0.32, 0.32, 0.32), envDiff, 0.36) * envIntensity
            + envSpec * ((0.30 + metal * 0.25) * envIntensity * viewFacing * (1.0 - rough * 0.35));
    }
    if (screenLightEnabled > 0.5) {
        float3 screenTint = (
            texScreenLight.SampleLevel(sampLinear, float2(0.50, 0.50), 0.0).rgb +
            texScreenLight.SampleLevel(sampLinear, float2(0.25, 0.30), 0.0).rgb +
            texScreenLight.SampleLevel(sampLinear, float2(0.75, 0.30), 0.0).rgb +
            texScreenLight.SampleLevel(sampLinear, float2(0.25, 0.70), 0.0).rgb +
            texScreenLight.SampleLevel(sampLinear, float2(0.75, 0.70), 0.0).rgb
        ) * 0.20;
        float3 screenLightDir = normalize(screenLightPosEnabled.xyz - input.worldPos);
        float screenFacing = max(dot(n, screenLightDir), 0.0);
        float screenKey = pow(screenFacing, 0.75);
        color += baseColor * screenTint * (1.00 * screenLightIntensity * screenKey);

        float3 r = reflect(-v, n);
        float denom = dot(r, screenLightNormalIntensity.xyz);
        if (abs(denom) > 0.001) {
            float t = dot(screenLightPosEnabled.xyz - input.worldPos, screenLightNormalIntensity.xyz) / denom;
            if (t > 0.0) {
                float3 hit = input.worldPos + r * t;
                float3 local = hit - screenLightPosEnabled.xyz;
                float2 screenP = float2(
                    dot(local, screenLightRightHalfX.xyz) / max(screenLightRightHalfX.w, 0.001),
                    dot(local, screenLightUpHalfY.xyz) / max(screenLightUpHalfY.w, 0.001)
                );
                if (abs(screenP.x) <= 1.0 && abs(screenP.y) <= 1.0) {
                    float2 screenUv = screenP * 0.5 + 0.5;
                    float3 screenCol = texScreenLight.SampleLevel(sampLinear, float2(1.0 - screenUv.x, 1.0 - screenUv.y), 0.0).rgb;
                    float fresnel = pow(clamp(1.0 - max(dot(n, v), 0.0), 0.0, 1.0), 2.0);
                    color += lerp(baseColor * screenTint, screenCol, 0.72) * (0.38 + 0.95 * fresnel) * screenLightIntensity * screenFacing;
                }
            }
        }
    }
    if (useOcclusion > 0.5) {
        float ao = texOcclusion.Sample(sampLinear, uvForTexCoord(input, occlusionTexcoord)).r;
        color *= lerp(1.0, ao, occlusionStrength);
    }
    float3 emissive = emissiveFactorUse.xyz;
    if (useEmissive > 0.5) {
        emissive *= gltfSrgbToLinear(texEmissive.Sample(sampLinear, uvForTexCoord(input, emissiveTexcoord)).rgb);
    }
    color += emissive;
    if (unlit > 0.5) {
        color = baseColor + emissive;
    }
    return float4(gltfLinearToOutput(color), alphaMode > 1.5 ? alpha : 1.0);
}
"""


LASER_HLSL_SOURCE = r"""
cbuffer LaserParams : register(b0)
{
    float4 mvpRow0;
    float4 mvpRow1;
    float4 mvpRow2;
    float4 mvpRow3;
    float4 laserParams;
    float4 laserColor;
};

struct VSIn {
    float3 pos : POSITION;
    float beamV : TEXCOORD0;
};

struct VSOut {
    float4 pos : SV_POSITION;
    float beamV : TEXCOORD0;
};

VSOut vs_main(VSIn input)
{
    VSOut output;
    float4 p = float4(input.pos, 1.0);
    output.pos = float4(dot(mvpRow0, p), dot(mvpRow1, p), dot(mvpRow2, p), dot(mvpRow3, p));
    output.beamV = input.beamV;
    return output;
}

float4 ps_main(VSOut input) : SV_TARGET
{
    if (laserParams.y > 0.5) {
        return laserColor;
    }
    float t = frac(input.beamV - laserParams.x * 0.4);
    float3 col;
    if (t < 0.167)      col = lerp(float3(0.0,0.4,1.0), float3(0.0,1.0,1.0), t/0.167);
    else if (t < 0.333) col = lerp(float3(0.0,1.0,1.0), float3(0.0,1.0,0.0), (t-0.167)/0.166);
    else if (t < 0.5)   col = lerp(float3(0.0,1.0,0.0), float3(1.0,1.0,0.0), (t-0.333)/0.167);
    else if (t < 0.667) col = lerp(float3(1.0,1.0,0.0), float3(1.0,0.5,0.0), (t-0.5)/0.167);
    else if (t < 0.833) col = lerp(float3(1.0,0.5,0.0), float3(1.0,0.0,0.0), (t-0.667)/0.166);
    else                col = lerp(float3(1.0,0.0,0.0), float3(0.0,0.4,1.0), (t-0.833)/0.167);
    return float4(col, 1.0);
}
"""


class D3D11NativeRenderer:
    def __init__(self, device, context, swapchain_format=DXGI_FORMAT_R8G8B8A8_UNORM):
        self.device = device
        self.context = context
        self.swapchain_format = int(swapchain_format or DXGI_FORMAT_R8G8B8A8_UNORM)
        self.color_format = DXGI_FORMAT_R8G8B8A8_UNORM
        self.color_tex = ctypes.c_void_p()
        self.depth_tex = ctypes.c_void_p()
        self.color_srv = ctypes.c_void_p()
        self.depth_srv = ctypes.c_void_p()
        self.color_cuda = ctypes.c_void_p()
        self.depth_cuda = ctypes.c_void_p()
        self.runtime_eye_tex = [ctypes.c_void_p(), ctypes.c_void_p()]
        self.runtime_eye_srv = [ctypes.c_void_p(), ctypes.c_void_p()]
        self.runtime_eye_cuda = [ctypes.c_void_p(), ctypes.c_void_p()]
        self.runtime_eye_size = None
        self.runtime_eye_cuda_logged = False
        self.cuda = None
        self.cuda_failed = False
        self.cuda_active_logged = False
        self.render_tex = ctypes.c_void_p()
        self.render_rtv = ctypes.c_void_p()
        self.render_target_size = None
        self.vertex_buffer = ctypes.c_void_p()
        self.constant_buffer = ctypes.c_void_p()
        self.input_layout = ctypes.c_void_p()
        self.vertex_shader = ctypes.c_void_p()
        self.pixel_shader = ctypes.c_void_p()
        self.background_vertex_shader = ctypes.c_void_p()
        self.background_pixel_shader = ctypes.c_void_p()
        self.background_srv = ctypes.c_void_p()
        self.background_tex = ctypes.c_void_p()
        self.background_path = None
        self.background_constant_buffer = ctypes.c_void_p()
        self.controller_vertex_shader = ctypes.c_void_p()
        self.controller_pixel_shader = ctypes.c_void_p()
        self.controller_input_layout = ctypes.c_void_p()
        self.controller_constant_buffer = ctypes.c_void_p()
        self.controller_prims = {}
        self.controller_textures = {}
        self.laser_vertex_shader = ctypes.c_void_p()
        self.laser_pixel_shader = ctypes.c_void_p()
        self.laser_input_layout = ctypes.c_void_p()
        self.laser_constant_buffer = ctypes.c_void_p()
        self.depth_stencil_state = ctypes.c_void_p()
        self.depth_read_state = ctypes.c_void_p()
        self.depth_disabled_state = ctypes.c_void_p()
        self.blend_state = ctypes.c_void_p()
        self.projection_depth_tex = ctypes.c_void_p()
        self.projection_depth_dsv = ctypes.c_void_p()
        self.projection_depth_size = None
        self.sampler = ctypes.c_void_p()
        self.rasterizer = ctypes.c_void_p()
        self.swapchain_rtvs = {}
        self._logged_swapchain_desc = set()
        self._logged_srgb_rtv_fallback = set()
        self.shader_mode = os.environ.get("D2S_D3D11_SHADER_MODE", "stereo").strip().lower()
        self.space_mode = os.environ.get("D2S_D3D11_SPACE_MODE", "world").strip().lower()
        self.debug = str(
            os.environ.get("D2S_D3D11_DEBUG", os.environ.get("D2S_OPENXR_DEBUG", "0")) or "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._logged_world_mvp = False
        self.size = None
        self.has_frame = False
        self._init_pipeline()

    def _device_call(self, index, restype, *argtypes):
        return _com_fn(self.device, index, restype, *argtypes)

    def _context_call(self, index, restype, *argtypes):
        return _com_fn(self.context, index, restype, *argtypes)

    def _device_removed_reason(self):
        try:
            get_reason = self._device_call(39, ctypes.c_long)
            return get_reason(_ptr_value(self.device))
        except Exception:
            return None

    def _shader_source(self):
        mode_id = {
            "clear": -1,
            "solid": 0,
            "color": 1,
            "color_only": 1,
            "stereo": 2,
        }.get(self.shader_mode, 2)
        self.shader_mode = (
            "clear" if mode_id < 0 else
            "solid" if mode_id == 0 else
            "color" if mode_id == 1 else
            "stereo"
        )
        if self.debug or self.shader_mode != "stereo":
            print(f"[OpenXRViewer] D3D11 shader mode: {self.shader_mode}")
        space_id = 1 if self.space_mode in ("clip", "screen", "debug") else 0
        self.space_mode = "clip" if space_id else "world"
        if self.debug or self.space_mode != "world":
            print(f"[OpenXRViewer] D3D11 space mode: {self.space_mode}")
        return (
            f"#define D2S_SHADER_MODE {mode_id}\n"
            f"#define D2S_SPACE_MODE {space_id}\n"
            + HLSL_SOURCE
        )

    def _init_pipeline(self):
        shader_source = self._shader_source()
        vs_blob = _compile_shader(shader_source, "vs_main", "vs_5_0")
        ps_blob = _compile_shader(shader_source, "ps_main", "ps_5_0")
        try:
            create_vs = self._device_call(
                12, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            create_ps = self._device_call(
                15, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            _check_hr(create_vs(_ptr_value(self.device), _blob_ptr(vs_blob), _blob_size(vs_blob), None, ctypes.byref(self.vertex_shader)), "CreateVertexShader")
            _check_hr(create_ps(_ptr_value(self.device), _blob_ptr(ps_blob), _blob_size(ps_blob), None, ctypes.byref(self.pixel_shader)), "CreatePixelShader")

            self.input_layout = ctypes.c_void_p()
        finally:
            _release(vs_blob)
            _release(ps_blob)

        bg_vs_blob = _compile_shader(BACKGROUND_HLSL_SOURCE, "vs_main", "vs_5_0")
        bg_ps_blob = _compile_shader(BACKGROUND_HLSL_SOURCE, "ps_main", "ps_5_0")
        try:
            create_vs = self._device_call(
                12, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            create_ps = self._device_call(
                15, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            _check_hr(create_vs(_ptr_value(self.device), _blob_ptr(bg_vs_blob), _blob_size(bg_vs_blob), None, ctypes.byref(self.background_vertex_shader)), "CreateVertexShader(background)")
            _check_hr(create_ps(_ptr_value(self.device), _blob_ptr(bg_ps_blob), _blob_size(bg_ps_blob), None, ctypes.byref(self.background_pixel_shader)), "CreatePixelShader(background)")
        finally:
            _release(bg_vs_blob)
            _release(bg_ps_blob)

        self.vertex_buffer = ctypes.c_void_p()
        self.constant_buffer = self._create_buffer(np.zeros(20, dtype=np.float32), D3D11_BIND_CONSTANT_BUFFER)
        self.background_constant_buffer = self._create_buffer(np.zeros(36, dtype=np.float32), D3D11_BIND_CONSTANT_BUFFER)
        self._init_controller_pipeline()
        self._init_laser_pipeline()
        self._init_depth_states()

        samp_desc = D3D11SamplerDesc(
            D3D11_FILTER_MIN_MAG_MIP_LINEAR,
            D3D11_TEXTURE_ADDRESS_WRAP,
            D3D11_TEXTURE_ADDRESS_WRAP,
            D3D11_TEXTURE_ADDRESS_WRAP,
            0.0,
            1,
            D3D11_COMPARISON_NEVER,
            (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0),
            0.0,
            3.4028234663852886e38,
        )
        create_sampler = self._device_call(23, ctypes.c_long, ctypes.POINTER(D3D11SamplerDesc), ctypes.POINTER(ctypes.c_void_p))
        _check_hr(create_sampler(_ptr_value(self.device), ctypes.byref(samp_desc), ctypes.byref(self.sampler)), "CreateSamplerState")

        rast_desc = D3D11RasterizerDesc(
            D3D11_FILL_SOLID,
            D3D11_CULL_NONE,
            1,
            0,
            0.0,
            0.0,
            1,
            0,
            0,
            0,
        )
        create_rasterizer = self._device_call(
            22, ctypes.c_long, ctypes.POINTER(D3D11RasterizerDesc), ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(
            create_rasterizer(_ptr_value(self.device), ctypes.byref(rast_desc), ctypes.byref(self.rasterizer)),
            "CreateRasterizerState",
        )
        rt_blends = (D3D11RenderTargetBlendDesc * 8)()
        rt_blends[0] = D3D11RenderTargetBlendDesc(
            1,
            D3D11_BLEND_SRC_ALPHA,
            D3D11_BLEND_INV_SRC_ALPHA,
            D3D11_BLEND_OP_ADD,
            D3D11_BLEND_ONE,
            D3D11_BLEND_INV_SRC_ALPHA,
            D3D11_BLEND_OP_ADD,
            D3D11_COLOR_WRITE_ENABLE_ALL,
        )
        blend_desc = D3D11BlendDesc(0, 0, rt_blends)
        create_blend = self._device_call(20, ctypes.c_long, ctypes.POINTER(D3D11BlendDesc), ctypes.POINTER(ctypes.c_void_p))
        _check_hr(create_blend(_ptr_value(self.device), ctypes.byref(blend_desc), ctypes.byref(self.blend_state)), "CreateBlendState(controller_alpha)")

    def _init_controller_pipeline(self):
        vs_blob = _compile_shader(CONTROLLER_HLSL_SOURCE, "vs_main", "vs_5_0")
        ps_blob = _compile_shader(CONTROLLER_HLSL_SOURCE, "ps_main", "ps_5_0")
        try:
            create_vs = self._device_call(
                12, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            create_ps = self._device_call(
                15, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            _check_hr(create_vs(_ptr_value(self.device), _blob_ptr(vs_blob), _blob_size(vs_blob), None, ctypes.byref(self.controller_vertex_shader)), "CreateVertexShader(controller)")
            _check_hr(create_ps(_ptr_value(self.device), _blob_ptr(ps_blob), _blob_size(ps_blob), None, ctypes.byref(self.controller_pixel_shader)), "CreatePixelShader(controller)")

            names = [b"POSITION", b"NORMAL", b"TEXCOORD"]
            position_offset, normal_offset, uv0_offset, uv1_offset = D3D11_VERTEX_OFFSETS_BYTES
            layout = (D3D11InputElementDesc * 4)(
                D3D11InputElementDesc(names[0], 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, position_offset, D3D11_INPUT_PER_VERTEX_DATA, 0),
                D3D11InputElementDesc(names[1], 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, normal_offset, D3D11_INPUT_PER_VERTEX_DATA, 0),
                D3D11InputElementDesc(names[2], 0, DXGI_FORMAT_R32G32_FLOAT, 0, uv0_offset, D3D11_INPUT_PER_VERTEX_DATA, 0),
                D3D11InputElementDesc(names[2], 1, DXGI_FORMAT_R32G32_FLOAT, 0, uv1_offset, D3D11_INPUT_PER_VERTEX_DATA, 0),
            )
            create_layout = self._device_call(
                11, ctypes.c_long, ctypes.POINTER(D3D11InputElementDesc), ctypes.c_uint,
                ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)
            )
            _check_hr(
                create_layout(_ptr_value(self.device), layout, 4, _blob_ptr(vs_blob), _blob_size(vs_blob), ctypes.byref(self.controller_input_layout)),
                "CreateInputLayout(controller)",
            )
        finally:
            _release(vs_blob)
            _release(ps_blob)
        self.controller_constant_buffer = self._create_buffer(np.zeros(128, dtype=np.float32), D3D11_BIND_CONSTANT_BUFFER)

    def _init_laser_pipeline(self):
        vs_blob = _compile_shader(LASER_HLSL_SOURCE, "vs_main", "vs_5_0")
        ps_blob = _compile_shader(LASER_HLSL_SOURCE, "ps_main", "ps_5_0")
        try:
            create_vs = self._device_call(
                12, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            create_ps = self._device_call(
                15, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            _check_hr(create_vs(_ptr_value(self.device), _blob_ptr(vs_blob), _blob_size(vs_blob), None, ctypes.byref(self.laser_vertex_shader)), "CreateVertexShader(laser)")
            _check_hr(create_ps(_ptr_value(self.device), _blob_ptr(ps_blob), _blob_size(ps_blob), None, ctypes.byref(self.laser_pixel_shader)), "CreatePixelShader(laser)")
            names = [b"POSITION", b"TEXCOORD"]
            layout = (D3D11InputElementDesc * 2)(
                D3D11InputElementDesc(names[0], 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0),
                D3D11InputElementDesc(names[1], 0, DXGI_FORMAT_R32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0),
            )
            create_layout = self._device_call(
                11, ctypes.c_long, ctypes.POINTER(D3D11InputElementDesc), ctypes.c_uint,
                ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)
            )
            _check_hr(
                create_layout(_ptr_value(self.device), layout, 2, _blob_ptr(vs_blob), _blob_size(vs_blob), ctypes.byref(self.laser_input_layout)),
                "CreateInputLayout(laser)",
            )
        finally:
            _release(vs_blob)
            _release(ps_blob)
        self.laser_constant_buffer = self._create_buffer(np.zeros(24, dtype=np.float32), D3D11_BIND_CONSTANT_BUFFER)
        self._init_laser_hit_circle_buffers()

    def _init_laser_hit_circle_buffers(self):
        segs = 40
        ring = []
        disk = []
        for i in range(segs):
            a0 = (i / segs) * math.tau
            a1 = ((i + 1) / segs) * math.tau
            outer0 = (math.cos(a0), math.sin(a0))
            outer1 = (math.cos(a1), math.sin(a1))
            inner0 = (outer0[0] * CURSOR_RING_INNER_RATIO, outer0[1] * CURSOR_RING_INNER_RATIO)
            inner1 = (outer1[0] * CURSOR_RING_INNER_RATIO, outer1[1] * CURSOR_RING_INNER_RATIO)
            for x, y in (outer0, inner0, outer1, outer1, inner0, inner1):
                ring.extend([x, y, 0.0, 0.0])
            for x, y in ((0.0, 0.0), outer0, outer1):
                disk.extend([x, y, 0.0, 0.0])
        ring_data = np.asarray(ring, dtype=np.float32)
        disk_data = np.asarray(disk, dtype=np.float32)
        self.laser_hit_ring_vertex_buffer = self._create_buffer(ring_data, D3D11_BIND_VERTEX_BUFFER)
        self.laser_hit_disk_vertex_buffer = self._create_buffer(disk_data, D3D11_BIND_VERTEX_BUFFER)
        self.laser_hit_ring_vertex_count = int(ring_data.size // 4)
        self.laser_hit_disk_vertex_count = int(disk_data.size // 4)

    def _init_depth_states(self):
        keep = D3D11DepthStencilOpDesc(
            D3D11_STENCIL_OP_KEEP,
            D3D11_STENCIL_OP_KEEP,
            D3D11_STENCIL_OP_KEEP,
            D3D11_COMPARISON_NEVER,
        )
        create_state = self._device_call(
            21, ctypes.c_long, ctypes.POINTER(D3D11DepthStencilDesc), ctypes.POINTER(ctypes.c_void_p)
        )
        enabled = D3D11DepthStencilDesc(
            1, D3D11_DEPTH_WRITE_MASK_ALL, D3D11_COMPARISON_LESS_EQUAL,
            0, 0xFF, 0xFF, keep, keep,
        )
        read_only = D3D11DepthStencilDesc(
            1, D3D11_DEPTH_WRITE_MASK_ZERO, D3D11_COMPARISON_LESS_EQUAL,
            0, 0xFF, 0xFF, keep, keep,
        )
        disabled = D3D11DepthStencilDesc(
            0, D3D11_DEPTH_WRITE_MASK_ZERO, D3D11_COMPARISON_LESS_EQUAL,
            0, 0xFF, 0xFF, keep, keep,
        )
        _check_hr(create_state(_ptr_value(self.device), ctypes.byref(enabled), ctypes.byref(self.depth_stencil_state)), "CreateDepthStencilState(enabled)")
        _check_hr(create_state(_ptr_value(self.device), ctypes.byref(read_only), ctypes.byref(self.depth_read_state)), "CreateDepthStencilState(read_only)")
        _check_hr(create_state(_ptr_value(self.device), ctypes.byref(disabled), ctypes.byref(self.depth_disabled_state)), "CreateDepthStencilState(disabled)")

    def _set_depth_enabled(self, enabled):
        state = self.depth_stencil_state if enabled else self.depth_disabled_state
        self._context_call(36, None, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(state), 0)

    def _set_depth_read_only(self):
        self._context_call(36, None, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.depth_read_state), 0)

    def _set_blend_disabled(self):
        factor = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
        self._context_call(35, None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint)(
            _ptr_value(self.context), None, factor, 0xFFFFFFFF
        )

    def _set_blend_alpha(self):
        factor = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
        self._context_call(35, None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint)(
            _ptr_value(self.context), _ptr_value(self.blend_state), factor, 0xFFFFFFFF
        )

    def _ensure_projection_depth(self, width, height):
        if self.projection_depth_size == (width, height):
            return self.projection_depth_dsv
        _release(self.projection_depth_dsv)
        _release(self.projection_depth_tex)
        self.projection_depth_dsv = ctypes.c_void_p()
        self.projection_depth_tex = ctypes.c_void_p()
        desc = D3D11Texture2DDesc(
            width, height, 1, 1, DXGI_FORMAT_D32_FLOAT, DXGISampleDesc(1, 0),
            D3D11_USAGE_DEFAULT, D3D11_BIND_DEPTH_STENCIL, 0, 0,
        )
        create_tex = self._device_call(
            5, ctypes.c_long, ctypes.POINTER(D3D11Texture2DDesc), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_tex(_ptr_value(self.device), ctypes.byref(desc), None, ctypes.byref(self.projection_depth_tex)), "CreateTexture2D(projection_depth)")
        create_dsv = self._device_call(
            10, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_dsv(_ptr_value(self.device), _ptr_value(self.projection_depth_tex), None, ctypes.byref(self.projection_depth_dsv)), "CreateDepthStencilView(projection)")
        self.projection_depth_size = (width, height)
        return self.projection_depth_dsv

    def _create_buffer(self, array, bind_flags):
        data = np.ascontiguousarray(array)
        desc = D3D11BufferDesc(data.nbytes, D3D11_USAGE_DEFAULT, bind_flags, 0, 0, 0)
        init = D3D11SubresourceData(ctypes.c_void_p(data.ctypes.data), 0, 0)
        out = ctypes.c_void_p()
        create_buffer = self._device_call(
            3, ctypes.c_long, ctypes.POINTER(D3D11BufferDesc), ctypes.POINTER(D3D11SubresourceData), ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_buffer(_ptr_value(self.device), ctypes.byref(desc), ctypes.byref(init), ctypes.byref(out)), "CreateBuffer")
        return out

    def _create_texture_srv(self, width, height, fmt):
        tex = ctypes.c_void_p()
        srv = ctypes.c_void_p()
        desc = D3D11Texture2DDesc(
            width, height, 1, 1, fmt, DXGISampleDesc(1, 0),
            D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0,
        )
        create_tex = self._device_call(
            5, ctypes.c_long, ctypes.POINTER(D3D11Texture2DDesc), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_tex(_ptr_value(self.device), ctypes.byref(desc), None, ctypes.byref(tex)), "CreateTexture2D")
        create_srv = self._device_call(
            7, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_srv(_ptr_value(self.device), tex, None, ctypes.byref(srv)), "CreateShaderResourceView")
        return tex, srv

    def _create_texture_srv_from_rgba(self, rgba):
        arr = np.ascontiguousarray(rgba, dtype=np.uint8)
        height, width = arr.shape[:2]
        tex = ctypes.c_void_p()
        srv = ctypes.c_void_p()
        desc = D3D11Texture2DDesc(
            width, height, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, DXGISampleDesc(1, 0),
            D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0,
        )
        init = D3D11SubresourceData(ctypes.c_void_p(arr.ctypes.data), width * 4, width * height * 4)
        create_tex = self._device_call(
            5, ctypes.c_long, ctypes.POINTER(D3D11Texture2DDesc), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_tex(_ptr_value(self.device), ctypes.byref(desc), ctypes.byref(init), ctypes.byref(tex)), "CreateTexture2D(background)")
        create_srv = self._device_call(
            7, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_srv(_ptr_value(self.device), tex, None, ctypes.byref(srv)), "CreateShaderResourceView(background)")
        return tex, srv

    def update_panorama_background(self, path):
        if not path or self.background_path == path:
            return
        _release(self.background_srv)
        _release(self.background_tex)
        self.background_srv = ctypes.c_void_p()
        self.background_tex = ctypes.c_void_p()
        if os.path.splitext(path)[1].lower() == ".hdr":
            from .environment_renderer import _hdr_to_ldr_u8, _read_radiance_hdr

            rgb, _size = _read_radiance_hdr(path)
            rgb = _hdr_to_ldr_u8(rgb)
        else:
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = rgb[..., :3]
        rgba[..., 3] = 255
        self.background_tex, self.background_srv = self._create_texture_srv_from_rgba(rgba)
        self.background_path = path
        print(f"[OpenXRViewer] D3D11 panorama background active: {path} ({rgba.shape[1]}x{rgba.shape[0]})")

    def _ensure_render_target(self, width, height):
        if self.render_target_size == (width, height, self.swapchain_format):
            return
        _release(self.render_rtv)
        _release(self.render_tex)
        self.render_rtv = ctypes.c_void_p()
        self.render_tex = ctypes.c_void_p()
        desc = D3D11Texture2DDesc(
            width, height, 1, 1, self.swapchain_format, DXGISampleDesc(1, 0),
            D3D11_USAGE_DEFAULT, D3D11_BIND_RENDER_TARGET, 0, 0,
        )
        create_tex = self._device_call(
            5, ctypes.c_long, ctypes.POINTER(D3D11Texture2DDesc), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_tex(_ptr_value(self.device), ctypes.byref(desc), None, ctypes.byref(self.render_tex)), "CreateTexture2D(render)")
        create_rtv = self._device_call(
            9, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_rtv(_ptr_value(self.device), _ptr_value(self.render_tex), None, ctypes.byref(self.render_rtv)), "CreateRenderTargetView(render)")
        self.render_target_size = (width, height, self.swapchain_format)

    def _get_texture_desc(self, texture_ptr):
        desc = D3D11Texture2DDesc()
        get_desc = _com_fn(texture_ptr, 10, None, ctypes.POINTER(D3D11Texture2DDesc))
        get_desc(texture_ptr, ctypes.byref(desc))
        return desc

    def _format_texture_desc(self, desc):
        return (
            f"size={desc.Width}x{desc.Height} fmt={desc.Format} "
            f"mips={desc.MipLevels} array={desc.ArraySize} "
            f"sample={desc.SampleDesc.Count}/{desc.SampleDesc.Quality} "
            f"usage={desc.Usage} bind=0x{desc.BindFlags:x} "
            f"cpu_access_flags=0x{desc.CPUAccessFlags:x} misc=0x{desc.MiscFlags:x}"
        )

    def _get_or_create_swapchain_rtv(self, swapchain_texture):
        texture_ptr = _ptr_value(swapchain_texture)
        if not texture_ptr:
            raise RuntimeError("OpenXR D3D11 swapchain texture is null")

        cached = self.swapchain_rtvs.get(texture_ptr)
        if cached:
            return cached

        desc = self._get_texture_desc(texture_ptr)
        if self.debug and texture_ptr not in self._logged_swapchain_desc:
            print(f"[OpenXRViewer] D3D11 swapchain texture desc: {self._format_texture_desc(desc)}")
            self._logged_swapchain_desc.add(texture_ptr)

        if not (desc.BindFlags & D3D11_BIND_RENDER_TARGET):
            raise RuntimeError(
                "OpenXR D3D11 swapchain texture is not render-target bindable: "
                f"{self._format_texture_desc(desc)}"
            )

        rtv_format = self.swapchain_format or desc.Format
        if rtv_format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            rtv_format = DXGI_FORMAT_R8G8B8A8_UNORM
        elif rtv_format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            rtv_format = DXGI_FORMAT_B8G8R8A8_UNORM
        if desc.ArraySize > 1:
            rtv_desc = D3D11RenderTargetViewDesc(
                rtv_format,
                D3D11_RTV_DIMENSION_TEXTURE2DARRAY,
                0,
                0,
                1,
            )
        else:
            rtv_desc = D3D11RenderTargetViewDesc(
                rtv_format,
                D3D11_RTV_DIMENSION_TEXTURE2D,
                0,
                0,
                0,
            )

        rtv = ctypes.c_void_p()
        create_rtv = self._device_call(
            9, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        hr = create_rtv(_ptr_value(self.device), texture_ptr, ctypes.byref(rtv_desc), ctypes.byref(rtv))
        if hr != 0 and rtv_format != (self.swapchain_format or desc.Format):
            srgb_format = self.swapchain_format or desc.Format
            rtv_desc.Format = srgb_format
            hr = create_rtv(_ptr_value(self.device), texture_ptr, ctypes.byref(rtv_desc), ctypes.byref(rtv))
            if hr == 0 and texture_ptr not in self._logged_srgb_rtv_fallback:
                print(
                    "[OpenXRViewer] D3D11 linear swapchain RTV unavailable; "
                    f"using runtime format {srgb_format}"
                )
                self._logged_srgb_rtv_fallback.add(texture_ptr)
        if hr != 0:
            removed_reason = self._device_removed_reason()
            reason_text = "" if removed_reason is None else f" removed_reason={_hr_hex(removed_reason)}"
            raise RuntimeError(
                "CreateRenderTargetView(OpenXR swapchain direct) failed: "
                f"hr={_hr_hex(hr)}{reason_text} rtv_fmt={rtv_format} {self._format_texture_desc(desc)}"
            )

        self.swapchain_rtvs[texture_ptr] = rtv
        return rtv

    def _ensure_frame_textures(self, width, height):
        if self.size == (width, height):
            return
        self._release_frame_textures()
        self.color_tex, self.color_srv = self._create_texture_srv(width, height, self.color_format)
        self.depth_tex, self.depth_srv = self._create_texture_srv(width, height, DXGI_FORMAT_R32_FLOAT)
        self.size = (width, height)
        self.has_frame = False

    def _release_frame_textures(self):
        if self.cuda is not None:
            for resource in (self.color_cuda, self.depth_cuda):
                try:
                    self.cuda.unregister_resource(resource)
                except Exception:
                    pass
        self.color_cuda = ctypes.c_void_p()
        self.depth_cuda = ctypes.c_void_p()
        for ptr in (self.color_srv, self.depth_srv, self.color_tex, self.depth_tex):
            _release(ptr)
        self.color_srv = ctypes.c_void_p()
        self.depth_srv = ctypes.c_void_p()
        self.color_tex = ctypes.c_void_p()
        self.depth_tex = ctypes.c_void_p()

    def _ensure_cuda_resources(self, torch_module, device_id):
        if self.cuda_failed:
            return False
        if self.cuda is None:
            self.cuda = CUDART_D3D11(torch_module, self.device, device_id)
        if not self.color_cuda:
            self.color_cuda = self.cuda.register_texture(self.color_tex)
        if not self.depth_cuda:
            self.depth_cuda = self.cuda.register_texture(self.depth_tex)
        return True

    def _ensure_runtime_eye_textures(self, width, height):
        if self.runtime_eye_size == (width, height):
            return
        self._release_runtime_eye_textures()
        for idx in range(2):
            self.runtime_eye_tex[idx], self.runtime_eye_srv[idx] = self._create_texture_srv(
                width, height, self.color_format
            )
        self.runtime_eye_size = (width, height)

    def _release_runtime_eye_textures(self):
        if self.cuda is not None:
            for resource in self.runtime_eye_cuda:
                try:
                    self.cuda.unregister_resource(resource)
                except Exception:
                    pass
        self.runtime_eye_cuda = [ctypes.c_void_p(), ctypes.c_void_p()]
        for ptr in (*self.runtime_eye_srv, *self.runtime_eye_tex):
            _release(ptr)
        self.runtime_eye_tex = [ctypes.c_void_p(), ctypes.c_void_p()]
        self.runtime_eye_srv = [ctypes.c_void_p(), ctypes.c_void_p()]
        self.runtime_eye_size = None

    def _ensure_runtime_eye_cuda_resources(self, torch_module, device_id):
        if self.cuda_failed:
            return False
        if self.cuda is None:
            self.cuda = CUDART_D3D11(torch_module, self.device, device_id)
        for idx in range(2):
            if not self.runtime_eye_cuda[idx]:
                self.runtime_eye_cuda[idx] = self.cuda.register_texture(self.runtime_eye_tex[idx])
        return True

    def _runtime_eye_tensor_rgba(self, torch_module, frame):
        tensor = frame.detach()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise RuntimeError(f"Unsupported OpenXR runtime eye batch for D3D11: {tuple(tensor.shape)}")
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
            tensor = tensor[:3].permute(1, 2, 0)
        elif tensor.ndim == 3 and tensor.shape[-1] >= 3:
            tensor = tensor[..., :3]
        else:
            raise RuntimeError(f"Unsupported OpenXR runtime eye shape for D3D11: {tuple(tensor.shape)}")
        if tensor.is_floating_point():
            tensor = tensor * 255.0
        tensor = tensor.contiguous().clamp(0, 255).to(torch_module.uint8)
        h, w = tensor.shape[:2]
        rgba = torch_module.empty((h, w, 4), device=tensor.device, dtype=torch_module.uint8)
        rgba[..., :3] = tensor
        rgba[..., 3] = 255
        return rgba

    def _update_runtime_eyes_cuda(self, torch_module, left, right):
        if not (
            hasattr(left, "is_cuda") and left.is_cuda and
            hasattr(right, "is_cuda") and right.is_cuda
        ):
            return False
        left_rgba = self._runtime_eye_tensor_rgba(torch_module, left)
        right_rgba = self._runtime_eye_tensor_rgba(torch_module, right)
        if left_rgba.shape[:2] != right_rgba.shape[:2]:
            raise RuntimeError(f"Runtime eye size mismatch for D3D11: left={tuple(left_rgba.shape)} right={tuple(right_rgba.shape)}")
        h, w = left_rgba.shape[:2]
        device = left_rgba.device
        device_id = 0 if device.index is None else int(device.index)
        if right_rgba.device != device:
            right_rgba = right_rgba.to(device, non_blocking=True)
        self._ensure_runtime_eye_textures(w, h)
        self._ensure_runtime_eye_cuda_resources(torch_module, device_id)
        stream = torch_module.cuda.current_stream(device_id)
        stream_ptr = stream.cuda_stream
        self.cuda.copy_tensor_to_texture(self.runtime_eye_cuda[0], left_rgba.data_ptr(), w * 4, w * 4, h, stream_ptr)
        self.cuda.copy_tensor_to_texture(self.runtime_eye_cuda[1], right_rgba.data_ptr(), w * 4, w * 4, h, stream_ptr)
        stream.synchronize()
        if not self.runtime_eye_cuda_logged:
            print("[OpenXRViewer] D3D11 runtime eye CUDA upload active (device-to-D3D11 texture)")
            self.runtime_eye_cuda_logged = True
        self.has_frame = True
        return w, h

    def update_runtime_eyes(self, left, right):
        try:
            import torch
        except Exception:
            torch = None
        if torch is None:
            return False
        if self.cuda_failed:
            return False
        try:
            result = self._update_runtime_eyes_cuda(torch, left, right)
            if result:
                return result
        except Exception as e:
            self.cuda_failed = True
            if self.cuda is not None:
                try:
                    self.cuda.clear_last_error()
                except Exception:
                    pass
            try:
                self._release_runtime_eye_textures()
            except Exception:
                pass
            print(f"[OpenXRViewer] D3D11 runtime eye CUDA upload unavailable: {e}")
        return False

    def _update_frame_cuda(self, torch_module, rgb, depth):
        if not (
            hasattr(rgb, "is_cuda") and rgb.is_cuda and
            hasattr(depth, "is_cuda") and depth.is_cuda
        ):
            return False

        device = depth.device
        device_id = 0 if device.index is None else int(device.index)
        depth_gpu = depth.detach()
        h, w = depth_gpu.shape[:2]
        self._ensure_frame_textures(w, h)
        self._ensure_cuda_resources(torch_module, device_id)

        rgb_gpu = rgb.detach()
        if rgb_gpu.device != device:
            rgb_gpu = rgb_gpu.to(device, non_blocking=True)
        if rgb_gpu.ndim == 3 and rgb_gpu.shape[0] in (3, 4):
            rgb_hwc = rgb_gpu[:3].permute(1, 2, 0)
        elif rgb_gpu.ndim == 3 and rgb_gpu.shape[-1] >= 3:
            rgb_hwc = rgb_gpu[..., :3]
        else:
            raise RuntimeError(f"Unsupported RGB tensor shape for D3D11 CUDA upload: {tuple(rgb_gpu.shape)}")

        if rgb_hwc.shape[0] != h or rgb_hwc.shape[1] != w:
            raise RuntimeError(
                f"RGB/depth size mismatch for D3D11 CUDA upload: rgb={tuple(rgb_hwc.shape)} depth={(h, w)}"
            )

        rgba = torch_module.empty((h, w, 4), device=device, dtype=torch_module.uint8)
        rgba[..., :3] = rgb_hwc.contiguous().clamp(0, 255).to(torch_module.uint8)
        rgba[..., 3] = 255

        depth_f = depth_gpu.contiguous().float()
        depth_f = torch_module.nan_to_num(depth_f, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        stream = torch_module.cuda.current_stream(device_id)
        stream_ptr = stream.cuda_stream
        self.cuda.copy_tensor_to_texture(self.color_cuda, rgba.data_ptr(), w * 4, w * 4, h, stream_ptr)
        self.cuda.copy_tensor_to_texture(self.depth_cuda, depth_f.data_ptr(), w * 4, w * 4, h, stream_ptr)
        stream.synchronize()
        if not self.cuda_active_logged:
            print("[OpenXRViewer] D3D11 CUDA upload active (device-to-device)")
            self.cuda_active_logged = True
        self.has_frame = True
        return w, h

    def update_frame(self, rgb, depth):
        try:
            import torch
        except Exception:
            torch = None

        if torch is not None and not self.cuda_failed:
            try:
                result = self._update_frame_cuda(torch, rgb, depth)
                if result:
                    return result
            except Exception as e:
                self.cuda_failed = True
                if self.cuda is not None:
                    try:
                        self.cuda.clear_last_error()
                    except Exception:
                        pass
                try:
                    self._release_frame_textures()
                except Exception:
                    pass
                warn_cpu_fallback(
                    "OpenXR D3D11 RGB+depth CUDA upload",
                    "upload_failed",
                    detail=str(e),
                    key="openxr_d3d11_frame_cuda_failed",
                )

        warn_cpu_fallback(
            "OpenXR D3D11 RGB+depth texture upload",
            "using_cpu_update_subresource",
            key="openxr_d3d11_frame_cpu_upload",
        )
        if torch is not None and hasattr(rgb, "detach"):
            warn_cpu_transfer(
                "OpenXR D3D11 RGB texture upload",
                ".cpu().numpy()",
                detail=describe_tensor(rgb),
                key="openxr_d3d11_rgb_cpu_transfer",
            )
            rgb_np = rgb.detach().permute(1, 2, 0).contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
        else:
            warn_cpu_transfer(
                "OpenXR D3D11 RGB texture upload",
                "numpy input path",
                detail=f"type={type(rgb).__name__}",
                key="openxr_d3d11_rgb_numpy_input",
            )
            rgb_np = np.asarray(rgb, dtype=np.uint8)
        if torch is not None and hasattr(depth, "detach"):
            warn_cpu_transfer(
                "OpenXR D3D11 depth texture upload",
                ".cpu().numpy()",
                detail=describe_tensor(depth),
                key="openxr_d3d11_depth_cpu_transfer",
            )
            depth_np = depth.detach().contiguous().float().cpu().numpy()
        else:
            warn_cpu_transfer(
                "OpenXR D3D11 depth texture upload",
                "numpy input path",
                detail=f"type={type(depth).__name__}",
                key="openxr_d3d11_depth_numpy_input",
            )
            depth_np = np.asarray(depth, dtype=np.float32)

        h, w = depth_np.shape[:2]
        self._ensure_frame_textures(w, h)
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.ascontiguousarray(rgb_np[:, :, :3])
        rgba[:, :, 3] = 255
        depth_f = depth_np.astype(np.float32, copy=False)
        depth_f = np.nan_to_num(depth_f, nan=0.0, posinf=1.0, neginf=0.0)
        depth_f = np.ascontiguousarray(np.clip(depth_f, 0.0, 1.0))
        self._update_subresource(self.color_tex, rgba.ctypes.data, w * 4)
        self._update_subresource(self.depth_tex, depth_f.ctypes.data, w * 4)
        self.has_frame = True
        return w, h

    def _update_subresource(self, dst, src_ptr, row_pitch):
        fn = self._context_call(
            48, None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        )
        fn(_ptr_value(self.context), _ptr_value(dst), 0, None, src_ptr, row_pitch, 0)

    def _log_world_mvp_once(self, mvp):
        if not self.debug or self._logged_world_mvp or self.space_mode != "world":
            return
        self._logged_world_mvp = True
        try:
            mat = np.asarray(mvp, dtype=np.float32)
            corners = np.array([
                [-1.0, -1.0, 0.0, 1.0],
                [-1.0,  1.0, 0.0, 1.0],
                [ 1.0, -1.0, 0.0, 1.0],
                [ 1.0,  1.0, 0.0, 1.0],
            ], dtype=np.float32)
            clip = (mat @ corners.T).T
            ndc = clip[:, :3] / np.maximum(np.abs(clip[:, 3:4]), 1e-6)
            parts = []
            for i in range(4):
                parts.append(
                    f"{i}:clip=({clip[i,0]:.3f},{clip[i,1]:.3f},{clip[i,2]:.3f},{clip[i,3]:.3f}) "
                    f"ndc=({ndc[i,0]:.3f},{ndc[i,1]:.3f},{ndc[i,2]:.3f})"
                )
            print("[OpenXRViewer] D3D11 world screen corners " + " | ".join(parts))
        except Exception as e:
            print(f"[OpenXRViewer] D3D11 world MVP debug failed: {e}")

    def render_eye(self, swapchain_texture, width, height, eye_index, eye_offset, depth_strength, convergence, mvp, roll=0.0, *, view_mat=None, proj_mat=None, overlay_viewer=None):
        return self._render_eye_with_srv(
            swapchain_texture, width, height, eye_index,
            self.color_srv, eye_offset, depth_strength, convergence, mvp, roll=roll,
            depth_srv=self.depth_srv, background_view_mat=view_mat, background_proj_mat=proj_mat,
            overlay_viewer=overlay_viewer,
        )

    def render_runtime_eye(self, swapchain_texture, width, height, eye_index, mvp, *, view_mat=None, proj_mat=None, overlay_viewer=None):
        if self.runtime_eye_size is None or not self.runtime_eye_srv[eye_index]:
            raise RuntimeError("D3D11 runtime eye textures are not ready")
        return self._render_eye_with_srv(
            swapchain_texture, width, height, eye_index,
            self.runtime_eye_srv[eye_index], 0.0, 0.0, 0.0, mvp, roll=0.0,
            depth_srv=None, background_view_mat=view_mat, background_proj_mat=proj_mat,
            overlay_viewer=overlay_viewer,
        )

    def _draw_background(self, view_mat, proj_mat):
        if not self.background_srv or view_mat is None or proj_mat is None:
            return
        try:
            view_rot = np.array(view_mat, dtype=np.float32, copy=True)
            view_rot[:3, 3] = 0.0
            inv_proj = np.linalg.inv(np.asarray(proj_mat, dtype=np.float32))
            inv_view = np.linalg.inv(view_rot)
        except Exception:
            return
        constants = np.zeros(36, dtype=np.float32)
        constants[:16] = inv_proj.reshape(16)
        constants[16:32] = inv_view.reshape(16)
        constants[32:36] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self._update_subresource(self.background_constant_buffer, constants.ctypes.data, constants.nbytes)
        self._context_call(11, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.background_vertex_shader), None, 0)
        self._context_call(9, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.background_pixel_shader), None, 0)
        cb_arr = (ctypes.c_void_p * 1)(_ptr_value(self.background_constant_buffer))
        self._context_call(7, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
        self._context_call(16, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
        srv_arr = (ctypes.c_void_p * 1)(_ptr_value(self.background_srv))
        self._context_call(8, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, srv_arr)
        sampler_arr = (ctypes.c_void_p * 1)(_ptr_value(self.sampler))
        self._context_call(10, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, sampler_arr)
        self._context_call(13, None, ctypes.c_uint, ctypes.c_uint)(_ptr_value(self.context), 4, 0)

    def _controller_prim_resources(self, prim):
        key = id(prim)
        cached = self.controller_prims.get(key)
        if cached is not None:
            return cached
        vertices = np.asarray(prim.get("vertices"), dtype=np.float32)
        indices = np.asarray(prim.get("indices"), dtype=np.uint32)
        if vertices.ndim != 2 or vertices.shape[1] < 8 or indices.size == 0:
            return None
        if vertices.shape[1] < 10:
            vertices = np.hstack([vertices[:, :8], vertices[:, 6:8]]).astype(np.float32, copy=False)
        topology = self._controller_topology(prim.get("primitive_mode", 4))
        indices, topology = self._controller_indices_for_topology(indices.reshape(-1), prim.get("primitive_mode", 4), topology)
        if indices.size == 0:
            return None
        vb = self._create_buffer(np.ascontiguousarray(vertices[:, :10], dtype=np.float32), D3D11_BIND_VERTEX_BUFFER)
        ib = self._create_buffer(np.ascontiguousarray(indices.reshape(-1), dtype=np.uint32), D3D11_BIND_INDEX_BUFFER)
        cached = (vb, ib, int(indices.size), topology)
        self.controller_prims[key] = cached
        return cached

    @staticmethod
    def _controller_topology(mode):
        return {
            0: D3D11_PRIMITIVE_TOPOLOGY_POINTLIST,
            1: D3D11_PRIMITIVE_TOPOLOGY_LINELIST,
            2: D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP,
            3: D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP,
            4: D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            5: D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            6: D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        }.get(int(mode or 4), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)

    @staticmethod
    def _controller_indices_for_topology(indices, mode, topology):
        mode = int(mode or 4)
        if mode == 2 and indices.size > 1:
            indices = np.concatenate([indices, indices[:1]]).astype(np.uint32, copy=False)
        elif mode == 6 and indices.size >= 3:
            fan = []
            for i in range(1, indices.size - 1):
                fan.extend((indices[0], indices[i], indices[i + 1]))
            indices = np.asarray(fan, dtype=np.uint32)
            topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST
        return np.ascontiguousarray(indices, dtype=np.uint32), topology

    def _controller_texture_srv(self, tex_key, tex_images):
        if not tex_key:
            return None
        cached = self.controller_textures.get(tex_key)
        if cached is not None:
            return cached[1]
        image = tex_images.get(tex_key) if isinstance(tex_images, dict) else None
        if image is None:
            return None
        tex, srv = self._create_texture_srv_from_rgba(image)
        self.controller_textures[tex_key] = (tex, srv)
        return srv

    @staticmethod
    def _alpha_mode_id(alpha_mode):
        return {"OPAQUE": 0.0, "MASK": 1.0, "BLEND": 2.0}.get(str(alpha_mode or "OPAQUE").upper(), 0.0)

    @staticmethod
    def _controller_model_base(viewer, grip_mat):
        t_mat = np.eye(4, dtype=np.float32)
        off = getattr(viewer, "_calibration_temp_offset", None) if getattr(viewer, "_calibration_mode", False) else getattr(viewer, "_ctrl_model_offset", None)
        rot_deg = getattr(viewer, "_calibration_temp_rot", None) if getattr(viewer, "_calibration_mode", False) else getattr(viewer, "_ctrl_model_rot_deg", 0.0)
        off = off if off is not None else (0.0, 0.0, 0.0)
        t_mat[0, 3], t_mat[1, 3], t_mat[2, 3] = float(off[0]), float(off[1]), float(off[2])
        ang = np.deg2rad(float(rot_deg or 0.0))
        ca, sa = np.cos(ang), np.sin(ang)
        r_mat = np.eye(4, dtype=np.float32)
        r_mat[1, 1], r_mat[1, 2], r_mat[2, 1], r_mat[2, 2] = ca, -sa, sa, ca
        return (np.asarray(grip_mat, dtype=np.float32) @ (r_mat @ t_mat)).astype(np.float32)

    def _draw_controller_models(self, viewer, view_mat, proj_mat, screen_light_srv=None):
        if viewer is None or view_mat is None or proj_mat is None:
            return
        now = float(getattr(viewer, "_frame_now", 0.0) or 0.0)
        hide_after = float(getattr(viewer, "_LASER_HIDE_AFTER", 10.0) or 10.0)
        controllers = []
        view_np = np.asarray(view_mat, dtype=np.float32)
        eye_pos = (-(view_np[:3, :3].T @ view_np[:3, 3])).astype(np.float32)
        for grip_mat, prims, last_move_attr, press_attr in (
            (getattr(viewer, "_grip_mat_l", None), getattr(viewer, "_ctrl_prims_l", None), "_laser_last_move_l", "_ctrl_press_l"),
            (getattr(viewer, "_grip_mat_r", None), getattr(viewer, "_ctrl_prims_r", None), "_laser_last_move_r", "_ctrl_press_r"),
        ):
            if grip_mat is None or not prims:
                continue
            if now - float(getattr(viewer, last_move_attr, now) or now) > hide_after:
                continue
            dist = float(np.linalg.norm(np.asarray(grip_mat, dtype=np.float32)[:3, 3] - eye_pos))
            controllers.append((dist, grip_mat, prims, getattr(viewer, press_attr, {}) or {}))
        if not controllers:
            return

        self._set_blend_disabled()
        self._set_depth_enabled(True)
        self._context_call(17, None, ctypes.c_void_p)(_ptr_value(self.context), _ptr_value(self.controller_input_layout))
        self._context_call(11, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.controller_vertex_shader), None, 0)
        self._context_call(9, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.controller_pixel_shader), None, 0)
        self._context_call(24, None, ctypes.c_uint)(_ptr_value(self.context), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
        cb_arr = (ctypes.c_void_p * 1)(_ptr_value(self.controller_constant_buffer))
        self._context_call(7, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
        self._context_call(16, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
        sampler_arr = (ctypes.c_void_p * 1)(_ptr_value(self.sampler))
        self._context_call(10, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, sampler_arr)

        vp_mat = (np.asarray(proj_mat, dtype=np.float32) @ np.asarray(view_mat, dtype=np.float32)).astype(np.float32)
        tex_images = getattr(viewer, "_ctrl_tex_images", {}) or {}
        config_material_diag = ""
        for _dist, _grip_mat, prims, _press_map in controllers:
            for prim in prims:
                config_material_diag = str((prim.get("material") or {}).get("material_diag", "") or "").strip().lower()
                if config_material_diag:
                    break
            if config_material_diag:
                break
        material_diag = os.environ.get("D2S_OPENXR_CONTROLLER_MATERIAL_DIAG", "").strip().lower() or config_material_diag
        diag_opaque_unlit = material_diag in ("1", "true", "unlit", "opaque_unlit")
        controller_hdr = bool(getattr(viewer, "_controller_hdr_lighting", True))
        env_srv = self.background_srv if controller_hdr and self.background_srv else None
        if diag_opaque_unlit:
            env_srv = None
            screen_light_srv = None
        elif not controller_hdr:
            screen_light_srv = None
        head_light = np.asarray(getattr(viewer, "_env_head_light_color", CONTROLLER_HEAD_LIGHT_COLOR), dtype=np.float32)
        ambient = np.asarray(getattr(viewer, "_env_ambient_color", (0.14, 0.13, 0.15)), dtype=np.float32)
        dir_vec = np.asarray(getattr(viewer, "_env_fallback_dir", (0.25, -0.82, -0.52)), dtype=np.float32)
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
        dir_color = np.zeros(3, dtype=np.float32)
        fill_specs = (
            (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0),
            (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0),
        )
        controllers.sort(key=lambda x: x[0], reverse=True)
        for _dist, grip_mat, prims, press_map in controllers:
            base_model = self._controller_model_base(viewer, grip_mat)
            sorted_prims = sorted(prims, key=lambda p: int(p.get("tri_count", 0) or 0), reverse=True)
            solid_prims = []
            sky_prims = []
            transparent_prims = []
            for prim in sorted_prims:
                render_pass = "opaque" if diag_opaque_unlit else render_pass_from_primitive(prim)
                if render_pass == "transparent":
                    transparent_prims.append(prim)
                elif render_pass == "sky":
                    sky_prims.append(prim)
                else:
                    solid_prims.append(prim)
            transparent_prims = sort_transparent_primitives(transparent_prims, eye_pos, base_model)
            for prim in sky_prims + solid_prims + transparent_prims:
                visible_key = prim.get("visible_key", "")
                if visible_key and float(press_map.get(visible_key, 0.0) or 0.0) <= 0.001:
                    continue
                model = base_model
                anim_key = prim.get("anim_key", "") or prim.get("node_name", "")
                press_amount = max(0.0, min(1.0, float(press_map.get(anim_key, press_map.get(prim.get("node_name", ""), 0.0)) or 0.0)))
                anim_delta = getattr(viewer, "_controller_anim_delta", lambda *_args: None)(prim.get("press_anim"), press_amount)
                if anim_delta is not None:
                    model = (model @ anim_delta).astype(np.float32)
                axis_anim = prim.get("axis_anim") or {}
                for axis in ("x", "y"):
                    amount = press_map.get(f"{anim_key}_{axis}", press_map.get(f"{prim.get('node_name', '')}_{axis}", 0.0))
                    delta = getattr(viewer, "_controller_anim_delta", lambda *_args: None)(axis_anim.get(axis), amount)
                    if delta is not None:
                        model = (model @ delta).astype(np.float32)

                res = self._controller_prim_resources(prim)
                if res is None:
                    continue
                vb, ib, index_count, topology = res
                mvp = (vp_mat @ model).astype(np.float32)
                mat = prim.get("material") or {}
                material_srvs = {
                    binding.role: self._controller_texture_srv(mat.get(binding.material_key), tex_images)
                    for binding in GLTF_MATERIAL_TEXTURE_BINDINGS
                }
                srv0 = material_srvs["base"]
                normal_srv = material_srvs["normal"]
                occlusion_srv = material_srvs["occlusion"]
                mr_srv = material_srvs["mr"]
                emissive_srv = material_srvs["emissive"]
                base_color = np.asarray(mat.get("base_color", (1.0, 1.0, 1.0)), dtype=np.float32)
                if base_color.size < 3:
                    base_color = np.array((1.0, 1.0, 1.0), dtype=np.float32)
                constants = np.zeros(128, dtype=np.float32)
                constants[0:16] = mvp.reshape(16)
                constants[16:32] = model.reshape(16)
                try:
                    normal_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)
                except Exception:
                    normal_mat = np.eye(3, dtype=np.float32)
                constants[32:36] = (float(normal_mat[0, 0]), float(normal_mat[0, 1]), float(normal_mat[0, 2]), 0.0)
                constants[36:40] = (float(normal_mat[1, 0]), float(normal_mat[1, 1]), float(normal_mat[1, 2]), 0.0)
                constants[40:44] = (float(normal_mat[2, 0]), float(normal_mat[2, 1]), float(normal_mat[2, 2]), 0.0)
                constants[44:48] = (
                    float(base_color[0]),
                    float(base_color[1]),
                    float(base_color[2]),
                    1.0 if diag_opaque_unlit else float(mat.get("base_alpha", 1.0) or 1.0),
                )
                use_texture = 1.0 if srv0 is not None else 0.0
                constants[48:52] = (
                    float(mat.get("roughness", 1.0) or 1.0),
                    float(mat.get("metallic", 0.0) or 0.0),
                    use_texture,
                    0.0 if diag_opaque_unlit else float(mat.get("alpha_mode_id", 0)),
                )
                constants[52:56] = (float(eye_pos[0]), float(eye_pos[1]), float(eye_pos[2]), 1.0 if env_srv else 0.0)
                constants[56:60] = (
                    float(getattr(viewer, "_env_exposure", 1.0) or 1.0),
                    float(mat.get("alpha_cutoff", 0.5) or 0.5),
                    1.0 if diag_opaque_unlit or mat.get("unlit") else 0.0,
                    1.0 if mat.get("double_sided", False) else 0.0,
                )
                if screen_light_srv is not None and getattr(viewer, "screen_height", None) is not None:
                    try:
                        sh, screen_pos, r_ax, u_ax, screen_n = viewer._screen_basis()
                        screen_light_intensity = max(0.0, float(getattr(viewer, "_screen_light_intensity", 3.5) or 0.0)) * (0.32 / 3.5)
                        constants[60:64] = (float(screen_pos[0]), float(screen_pos[1]), float(screen_pos[2]), 1.0)
                        constants[64:68] = (float(screen_n[0]), float(screen_n[1]), float(screen_n[2]), screen_light_intensity)
                        constants[68:72] = (float(r_ax[0]), float(r_ax[1]), float(r_ax[2]), float(viewer.screen_width) * 0.5)
                        constants[72:76] = (float(u_ax[0]), float(u_ax[1]), float(u_ax[2]), float(sh) * 0.5)
                    except Exception:
                        pass
                constants[76:80] = (
                    float(mat.get("normal_scale", 1.0) or 1.0),
                    float(mat.get("occlusion_strength", 1.0) or 1.0),
                    0.0 if diag_opaque_unlit else 1.0 if normal_srv is not None else 0.0,
                    0.0 if diag_opaque_unlit else 1.0 if occlusion_srv is not None else 0.0,
                )
                emissive = np.asarray(mat.get("emissive_factor", (0.0, 0.0, 0.0)), dtype=np.float32)
                if emissive.size < 3:
                    emissive = np.zeros(3, dtype=np.float32)
                constants[80:84] = (
                    float(emissive[0]),
                    float(emissive[1]),
                    float(emissive[2]),
                    0.0,
                )
                tex_offset = np.asarray(mat.get("tex_offset", (0.0, 0.0)), dtype=np.float32)
                tex_scale = np.asarray(mat.get("tex_scale", (1.0, 1.0)), dtype=np.float32)
                constants[84:88] = (
                    float(tex_offset[0]) if tex_offset.size > 0 else 0.0,
                    float(tex_offset[1]) if tex_offset.size > 1 else 0.0,
                    float(tex_scale[0]) if tex_scale.size > 0 else 1.0,
                    float(tex_scale[1]) if tex_scale.size > 1 else 1.0,
                )
                constants[88:92] = (
                    float(mat.get("tex_rotation", 0.0) or 0.0),
                    float(mat.get("base_texcoord", 0) or 0),
                    float(mat.get("normal_texcoord", 0) or 0),
                    float(mat.get("occlusion_texcoord", 0) or 0),
                )
                constants[92:96] = (
                    float(mat.get("mr_texcoord", 0) or 0),
                    float(mat.get("emissive_texcoord", 0) or 0),
                    0.0 if diag_opaque_unlit else 1.0 if mr_srv is not None else 0.0,
                    0.0 if diag_opaque_unlit else 1.0 if emissive_srv is not None else 0.0,
                )
                constants[96:100] = (
                    float(head_light[0]),
                    float(head_light[1]),
                    float(head_light[2]),
                    CONTROLLER_TOP_LIGHT_INTENSITY,
                )
                constants[100:104] = (float(ambient[0]), float(ambient[1]), float(ambient[2]), 0.0)
                constants[104:108] = (float(dir_vec[0]), float(dir_vec[1]), float(dir_vec[2]), 0.0)
                constants[108:112] = (float(dir_color[0]), float(dir_color[1]), float(dir_color[2]), 0.0)
                fill_pos0, fill_color0, fill_range0 = fill_specs[0]
                fill_pos1, fill_color1, fill_range1 = fill_specs[1]
                constants[112:116] = (float(fill_pos0[0]), float(fill_pos0[1]), float(fill_pos0[2]), max(float(fill_range0), 0.001))
                constants[116:120] = (float(fill_color0[0]), float(fill_color0[1]), float(fill_color0[2]), 0.0)
                constants[120:124] = (float(fill_pos1[0]), float(fill_pos1[1]), float(fill_pos1[2]), max(float(fill_range1), 0.001))
                constants[124:128] = (float(fill_color1[0]), float(fill_color1[1]), float(fill_color1[2]), 0.0)
                self._update_subresource(self.controller_constant_buffer, constants.ctypes.data, constants.nbytes)

                render_pass = "opaque" if diag_opaque_unlit else render_pass_from_primitive(prim)
                if render_pass == "transparent":
                    self._set_blend_alpha()
                    self._set_depth_read_only()
                elif render_pass == "sky":
                    self._set_blend_disabled()
                    self._set_depth_read_only()
                else:
                    self._set_blend_disabled()
                    self._set_depth_enabled(True)
                self._context_call(43, None, ctypes.c_void_p)(_ptr_value(self.context), _ptr_value(self.rasterizer))

                srv_values = [
                    _ptr_value(srv0 or env_srv or 0),
                    _ptr_value(env_srv or srv0 or 0),
                    _ptr_value(screen_light_srv or 0),
                    0,
                    0,
                    0,
                    0,
                ]
                for binding in GLTF_MATERIAL_TEXTURE_BINDINGS:
                    srv_values[binding.d3d11_srv_slot] = _ptr_value(material_srvs[binding.role] or 0)
                srv_arr = (ctypes.c_void_p * 7)(*srv_values)
                self._context_call(8, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 7, srv_arr)
                stride = ctypes.c_uint(D3D11_VERTEX_STRIDE_BYTES)
                offset = ctypes.c_uint(0)
                vb_arr = (ctypes.c_void_p * 1)(_ptr_value(vb))
                self._context_call(18, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint))(
                    _ptr_value(self.context), 0, 1, vb_arr, ctypes.byref(stride), ctypes.byref(offset)
                )
                self._context_call(19, None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(ib), DXGI_FORMAT_R32_UINT, 0)
                self._context_call(24, None, ctypes.c_uint)(_ptr_value(self.context), topology)
                self._context_call(12, None, ctypes.c_uint, ctypes.c_uint, ctypes.c_int)(_ptr_value(self.context), index_count, 0, 0)

    def _draw_environment_model(self, viewer, view_mat, proj_mat):
        """Draw environment GLB primitives with environment-owned lighting state.

        The D3D11 controller shader is shared as a rendering implementation, but
        the proxy carries the room primitives and room profile values explicitly.
        """
        if viewer is None or view_mat is None or proj_mat is None:
            return
        prims = getattr(viewer, "_env_model_prims", None) or []
        if not prims:
            ensure_env = getattr(viewer, "_ensure_env_model_initialized", None)
            if callable(ensure_env):
                ensure_env("D3D11")
            prims = getattr(viewer, "_env_model_prims", None) or []
        if not prims or not getattr(viewer, "_env_model_visible", True):
            if not getattr(self, "_d3d11_environment_empty_logged", False):
                self._d3d11_environment_empty_logged = True
                print(
                    "[OpenXRViewer] D3D11 environment skipped: "
                    f"prims={len(prims)} visible={bool(getattr(viewer, '_env_model_visible', False))} "
                    f"init_done={bool(getattr(viewer, '_env_model_init_done', False))} "
                    f"path={getattr(viewer, '_env_model_path', None)!s} "
                    f"panorama={getattr(viewer, '_panorama_background_path', None)!s}",
                    flush=True,
                )
            return
        summary = getattr(viewer, "_env_model_summary", None)
        if summary and not getattr(self, "_d3d11_environment_summary_logged", False):
            self._d3d11_environment_summary_logged = True
            print("[OpenXRViewer] D3D11 " + format_gltf_scene_summary(summary, label="Active environment"), flush=True)
        try:
            model_fn = getattr(viewer, "_env_model_mat4", None) or getattr(viewer, "_build_env_model_mat4")
            model = np.asarray(model_fn(), dtype=np.float32)
        except Exception as exc:
            if not getattr(self, "_d3d11_environment_error_logged", False):
                self._d3d11_environment_error_logged = True
                print(f"[OpenXRViewer] D3D11 environment transform failed: {type(exc).__name__}: {exc}", flush=True)
            return
        images = {}
        for prim in prims:
            images.update(prim.get("d3d_tex_images", {}) or {})
            # Room geometry is viewed from the inside; keep controller back-face
            # rejection from hiding outward-wound interior surfaces.
            material = prim.get("material")
            if isinstance(material, dict):
                material["double_sided"] = True
        proxy = SimpleNamespace(
            _frame_now=getattr(viewer, "_frame_now", 0.0),
            _LASER_HIDE_AFTER=0.0,
            _grip_mat_l=model,
            _grip_mat_r=None,
            _ctrl_prims_l=prims,
            _ctrl_prims_r=None,
            _laser_last_move_l=getattr(viewer, "_frame_now", 0.0),
            _laser_last_move_r=0.0,
            _ctrl_press_l={},
            _ctrl_press_r={},
            _ctrl_tex_images=images,
            _ctrl_model_offset=(0.0, 0.0, 0.0),
            _ctrl_model_rot_deg=0.0,
            _calibration_mode=False,
            _controller_hdr_lighting=bool(getattr(viewer, "_controller_hdr_lighting", False)),
            _env_head_light_color=getattr(viewer, "_env_head_light_color", CONTROLLER_HEAD_LIGHT_COLOR),
            _env_ambient_color=getattr(viewer, "_env_ambient_color", (0.14, 0.13, 0.15)),
            _env_fallback_dir=getattr(viewer, "_env_fallback_dir", (0.25, -0.82, -0.52)),
            _env_exposure=getattr(viewer, "_env_exposure", 1.0),
            _controller_anim_delta=lambda *_args: None,
        )
        if hasattr(viewer, "_screen_basis"):
            proxy._screen_basis = viewer._screen_basis
        self._draw_controller_models(proxy, view_mat, proj_mat, None)

    def _draw_lasers(self, viewer, view_mat, proj_mat):
        if viewer is None or view_mat is None or proj_mat is None or not hasattr(viewer, "_laser_beam_setup"):
            return
        try:
            if getattr(viewer, "_beams_frame", -1) != getattr(viewer, "_frame_count", 0):
                viewer._cached_beams = viewer._laser_beam_setup()
                viewer._beams_frame = getattr(viewer, "_frame_count", 0)
            beams = getattr(viewer, "_cached_beams", None) or []
        except Exception:
            return
        if not beams:
            return
        draws = [
            (ctrl_pos, fwd_w, right2, up, LASER_MAX_LENGTH_M)
            for _now, _ctrl_name, _aim_mat, ctrl_pos, fwd_w, right2, _fwd, up in beams
        ]
        data = build_laser_beam_vertices(draws, LASER_BASE_HALF_WIDTH_M, LASER_TIP_HALF_WIDTH_M)
        if data.size == 0:
            return
        vb = self._create_buffer(data, D3D11_BIND_VERTEX_BUFFER)
        try:
            self._set_blend_disabled()
            mvp = (np.asarray(proj_mat, dtype=np.float32) @ np.asarray(view_mat, dtype=np.float32)).astype(np.float32)
            constants = np.zeros(24, dtype=np.float32)
            constants[:16] = mvp.reshape(16)
            constants[16] = float(getattr(viewer, "_frame_now", 0.0) or 0.0)
            self._update_subresource(self.laser_constant_buffer, constants.ctypes.data, constants.nbytes)
            self._set_depth_enabled(True)
            self._context_call(17, None, ctypes.c_void_p)(_ptr_value(self.context), _ptr_value(self.laser_input_layout))
            self._context_call(11, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.laser_vertex_shader), None, 0)
            self._context_call(9, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.laser_pixel_shader), None, 0)
            cb_arr = (ctypes.c_void_p * 1)(_ptr_value(self.laser_constant_buffer))
            self._context_call(7, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
            self._context_call(16, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
            self._context_call(24, None, ctypes.c_uint)(_ptr_value(self.context), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
            stride = ctypes.c_uint(16)
            offset = ctypes.c_uint(0)
            vb_arr = (ctypes.c_void_p * 1)(_ptr_value(vb))
            self._context_call(18, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint))(
                _ptr_value(self.context), 0, 1, vb_arr, ctypes.byref(stride), ctypes.byref(offset)
            )
            self._context_call(13, None, ctypes.c_uint, ctypes.c_uint)(_ptr_value(self.context), data.size // 4, 0)
        finally:
            _release(vb)

    def _draw_laser_hit_circles(self, viewer, view_mat, proj_mat):
        if viewer is None or view_mat is None or proj_mat is None or not hasattr(viewer, "_laser_hit_circle_draws"):
            return
        old_view = getattr(viewer, "_current_view_mat", None)
        try:
            if getattr(viewer, "_beams_frame", -1) != getattr(viewer, "_frame_count", 0):
                viewer._cached_beams = viewer._laser_beam_setup()
                viewer._beams_frame = getattr(viewer, "_frame_count", 0)
            beams = getattr(viewer, "_cached_beams", None) or []
            viewer._current_view_mat = np.asarray(view_mat, dtype=np.float32)
            draws = viewer._laser_hit_circle_draws(beams)
        except Exception as exc:
            if not getattr(self, "_d3d11_hit_circle_error_logged", False):
                self._d3d11_hit_circle_error_logged = True
                print(f"[OpenXRViewer] D3D11 laser hit circle failed: {type(exc).__name__}: {exc}", flush=True)
            return
        finally:
            viewer._current_view_mat = old_view
        if not draws:
            return
        view_np = np.asarray(view_mat, dtype=np.float32)
        eye_pos = (-(view_np[:3, :3].T @ view_np[:3, 3])).astype(np.float32)
        vp = (np.asarray(proj_mat, dtype=np.float32) @ view_np).astype(np.float32)
        self._set_blend_alpha()
        self._set_depth_enabled(False)
        self._context_call(17, None, ctypes.c_void_p)(_ptr_value(self.context), _ptr_value(self.laser_input_layout))
        self._context_call(11, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.laser_vertex_shader), None, 0)
        self._context_call(9, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.laser_pixel_shader), None, 0)
        cb_arr = (ctypes.c_void_p * 1)(_ptr_value(self.laser_constant_buffer))
        self._context_call(7, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
        self._context_call(16, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
        self._context_call(24, None, ctypes.c_uint)(_ptr_value(self.context), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
        for shape, model, color in draws:
            m = np.asarray(model, dtype=np.float32).copy()
            to_eye = eye_pos - m[:3, 3]
            dist = float(np.linalg.norm(to_eye))
            if dist > 1e-5:
                m[:3, 3] += (to_eye / dist) * 0.01
            constants = np.zeros(24, dtype=np.float32)
            constants[:16] = (vp @ m).reshape(16)
            constants[17] = 1.0
            constants[20:24] = np.asarray(color, dtype=np.float32)
            self._update_subresource(self.laser_constant_buffer, constants.ctypes.data, constants.nbytes)
            stride = ctypes.c_uint(16)
            offset = ctypes.c_uint(0)
            vb = self.laser_hit_ring_vertex_buffer if shape == "ring" else self.laser_hit_disk_vertex_buffer
            vertex_count = self.laser_hit_ring_vertex_count if shape == "ring" else self.laser_hit_disk_vertex_count
            vb_arr = (ctypes.c_void_p * 1)(_ptr_value(vb))
            self._context_call(18, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint))(
                _ptr_value(self.context), 0, 1, vb_arr, ctypes.byref(stride), ctypes.byref(offset)
            )
            self._context_call(13, None, ctypes.c_uint, ctypes.c_uint)(_ptr_value(self.context), vertex_count, 0)

    def _render_eye_with_srv(self, swapchain_texture, width, height, eye_index, color_srv, eye_offset, depth_strength, convergence, mvp, *, roll=0.0, depth_srv=None, background_view_mat=None, background_proj_mat=None, overlay_viewer=None):
        rtv = self._get_or_create_swapchain_rtv(swapchain_texture)
        dsv = self._ensure_projection_depth(int(width), int(height))
        try:
            clear = self._context_call(50, None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float))
            color = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 1.0)
            clear(_ptr_value(self.context), _ptr_value(rtv), color)
            if self.shader_mode == "clear":
                removed_reason = self._device_removed_reason()
                if removed_reason not in (0, None):
                    raise RuntimeError(f"D3D11 device removed after ClearRenderTargetView: removed_reason={_hr_hex(removed_reason)}")
                return

            viewport = D3D11Viewport(0.0, 0.0, float(width), float(height), 0.0, 1.0)
            self._context_call(44, None, ctypes.c_uint, ctypes.POINTER(D3D11Viewport))(_ptr_value(self.context), 1, ctypes.byref(viewport))

            rtv_arr = (ctypes.c_void_p * 1)(_ptr_value(rtv))
            self._context_call(33, None, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)(_ptr_value(self.context), 1, rtv_arr, _ptr_value(dsv))
            self._context_call(53, None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_float, ctypes.c_uint)(
                _ptr_value(self.context), _ptr_value(dsv), D3D11_CLEAR_DEPTH, 1.0, 0
            )

            self._set_blend_disabled()
            self._set_depth_enabled(False)
            self._context_call(17, None, ctypes.c_void_p)(_ptr_value(self.context), 0)
            self._context_call(43, None, ctypes.c_void_p)(_ptr_value(self.context), _ptr_value(self.rasterizer))
            self._context_call(24, None, ctypes.c_uint)(_ptr_value(self.context), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP)
            self._draw_background(background_view_mat, background_proj_mat)
            self._draw_environment_model(overlay_viewer, background_view_mat, background_proj_mat)
            self._context_call(11, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.vertex_shader), None, 0)
            self._context_call(9, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.pixel_shader), None, 0)

            self._log_world_mvp_once(mvp)
            constants = np.zeros(20, dtype=np.float32)
            constants[:16] = np.asarray(mvp, dtype=np.float32).reshape(16)
            constants[16:20] = np.array([eye_offset, depth_strength, convergence, roll], dtype=np.float32)
            self._update_subresource(self.constant_buffer, constants.ctypes.data, constants.nbytes)
            cb_arr = (ctypes.c_void_p * 1)(_ptr_value(self.constant_buffer))
            self._context_call(7, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
            self._context_call(16, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)

            srv_arr = (ctypes.c_void_p * 2)(_ptr_value(color_srv), _ptr_value(depth_srv or color_srv))
            self._context_call(8, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 2, srv_arr)
            sampler_arr = (ctypes.c_void_p * 1)(_ptr_value(self.sampler))
            self._context_call(10, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, sampler_arr)
            self._context_call(13, None, ctypes.c_uint, ctypes.c_uint)(_ptr_value(self.context), 4, 0)
            self._draw_lasers(overlay_viewer, background_view_mat, background_proj_mat)
            self._draw_controller_models(overlay_viewer, background_view_mat, background_proj_mat, color_srv)
            self._draw_laser_hit_circles(overlay_viewer, background_view_mat, background_proj_mat)
            removed_reason = self._device_removed_reason()
            if removed_reason not in (0, None):
                raise RuntimeError(f"D3D11 device removed after Draw: removed_reason={_hr_hex(removed_reason)}")
        finally:
            null_srvs = (ctypes.c_void_p * 7)(0, 0, 0, 0, 0, 0, 0)
            null_rtvs = (ctypes.c_void_p * 1)(0)
            try:
                self._context_call(8, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 7, null_srvs)
            except Exception:
                pass
            try:
                self._context_call(33, None, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)(_ptr_value(self.context), 1, null_rtvs, None)
            except Exception:
                pass
            try:
                self._set_depth_enabled(False)
            except Exception:
                pass

    def cleanup(self):
        for rtv in self.swapchain_rtvs.values():
            _release(rtv)
        self.swapchain_rtvs.clear()
        self._release_frame_textures()
        self._release_runtime_eye_textures()
        _release(self.projection_depth_dsv)
        _release(self.projection_depth_tex)
        self.projection_depth_dsv = ctypes.c_void_p()
        self.projection_depth_tex = ctypes.c_void_p()
        self.projection_depth_size = None
        for vb, ib, _index_count, _topology in self.controller_prims.values():
            _release(vb)
            _release(ib)
        self.controller_prims.clear()
        for tex, srv in self.controller_textures.values():
            _release(srv)
            _release(tex)
        self.controller_textures.clear()
        for attr in (
            "sampler", "pixel_shader", "vertex_shader", "input_layout",
            "background_pixel_shader", "background_vertex_shader", "background_srv", "background_tex", "background_constant_buffer",
            "controller_pixel_shader", "controller_vertex_shader", "controller_input_layout", "controller_constant_buffer",
            "laser_pixel_shader", "laser_vertex_shader", "laser_input_layout", "laser_constant_buffer",
            "laser_hit_ring_vertex_buffer", "laser_hit_disk_vertex_buffer",
            "depth_stencil_state", "depth_read_state", "depth_disabled_state", "blend_state",
            "rasterizer", "constant_buffer", "vertex_buffer", "render_rtv", "render_tex",
        ):
            ptr = getattr(self, attr, None)
            _release(ptr)
            setattr(self, attr, ctypes.c_void_p())
        self.render_target_size = None
