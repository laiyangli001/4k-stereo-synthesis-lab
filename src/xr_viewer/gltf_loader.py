# Desktop2Stereo OpenXR viewer: GLB/glTF loading helpers.

import base64
import io as _io
import json
import math
import os
import urllib.parse

import moderngl
import numpy as np
from PIL import Image
from pygltflib import GLTF2

from .gltf.contract import (
    GltfMaterial,
    GltfScene,
    TextureBinding,
    TextureTransform,
    attach_primitive_contract,
    build_render_plan,
)
from .material_contract import GLTF_MATERIAL_TEXTURE_BINDINGS

# GLB/glTF parsing is delegated to pygltflib so file/container/schema handling
# follows a maintained glTF 2.0 implementation. The renderer-facing contract
# below remains our own: numpy vertex/index arrays plus material texture ids.
def _gltf_to_dict(gltf):
    return json.loads(gltf.to_json())


def _read_glb_chunks(data):
    """Compatibility helper for diagnostics/tests that still pass GLB bytes."""
    gltf = GLTF2().load_from_bytes(data)
    return _gltf_to_dict(gltf), gltf.binary_blob()


def _decode_data_uri(uri):
    try:
        header, payload = uri.split(',', 1)
    except ValueError:
        raise ValueError('Invalid glTF data URI')
    if ';base64' in header:
        return base64.b64decode(payload)
    return urllib.parse.unquote_to_bytes(payload)


def _load_gltf_document(path):
    gltf_obj = GLTF2().load(path)
    gltf = _gltf_to_dict(gltf_obj)
    base_dir = os.path.dirname(os.path.abspath(path))
    binary_blob = gltf_obj.binary_blob()
    buffers = []
    for index, buf in enumerate(gltf.get('buffers') or []):
        uri = buf.get('uri') if isinstance(buf, dict) else None
        data = None
        if uri:
            if uri.startswith('data:'):
                data = _decode_data_uri(uri)
            else:
                parsed = urllib.parse.urlparse(uri)
                if parsed.scheme not in ('', 'file'):
                    raise ValueError(f'Unsupported glTF buffer URI scheme: {parsed.scheme}')
                rel_path = urllib.parse.unquote(parsed.path if parsed.scheme == 'file' else uri)
                rel_path = rel_path.replace('/', os.sep)
                buffer_path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
                with open(buffer_path, 'rb') as bf:
                    data = bf.read()
        elif index == 0 and binary_blob is not None:
            data = binary_blob
        else:
            data = b''
        buffers.append(data)
    return gltf, buffers


def _buffer_data(buffers, buffer_index=0):
    if isinstance(buffers, (bytes, bytearray, memoryview)):
        return buffers if int(buffer_index or 0) == 0 else None
    if not isinstance(buffers, (list, tuple)):
        return None
    try:
        index = int(buffer_index or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if index < 0 or index >= len(buffers):
        return None
    return buffers[index]



_DTYPE_MAP = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
            5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
            'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


def gltf_primitive_mode_to_moderngl(mode):
    return {
        0: moderngl.POINTS,
        1: moderngl.LINES,
        2: moderngl.LINE_LOOP,
        3: moderngl.LINE_STRIP,
        4: moderngl.TRIANGLES,
        5: moderngl.TRIANGLE_STRIP,
        6: moderngl.TRIANGLE_FAN,
    }.get(int(mode), moderngl.TRIANGLES)


_DEFAULT_GLTF_SAMPLER = (9729, 9987, 10497, 10497)  # mag, min, wrapS, wrapT
_VALID_GLTF_MAG_FILTERS = {9728, 9729}
_VALID_GLTF_MIN_FILTERS = {9728, 9729, 9984, 9985, 9986, 9987}
_VALID_GLTF_WRAPS = {33071, 33648, 10497}


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value, default=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return v if math.isfinite(v) else default


def _clamp_float(value, lo=0.0, hi=1.0, default=0.0):
    v = _safe_float(value, default)
    return max(lo, min(hi, v))


def _clamp_vec(values, size, default=1.0, lo=0.0, hi=1.0):
    out = [default] * size
    if isinstance(values, (list, tuple, np.ndarray)):
        for i in range(min(size, len(values))):
            out[i] = _clamp_float(values[i], lo, hi, default)
    return np.array(out, dtype=np.float32)


def _safe_nonnegative_float(value, default=1.0):
    return max(0.0, _safe_float(value, default))


def _safe_texcoord(value, default=0):
    idx = _safe_int(value, default)
    return idx if idx >= 0 else default


def _texture_index(tex_info):
    if not isinstance(tex_info, dict):
        return None
    idx = tex_info.get('index')
    return idx if isinstance(idx, int) and idx >= 0 else None


def _texture_image_id(tex_img_map, all_textures, tex_index):
    if not isinstance(tex_index, int):
        return -1
    image_id = tex_img_map.get(tex_index, -1)
    if isinstance(image_id, int) and 0 <= image_id < len(all_textures) and all_textures[image_id] is not None:
        return image_id
    return -1


def _texture_sampler(tex_sampler_map, tex_index):
    if not isinstance(tex_index, int):
        return _DEFAULT_GLTF_SAMPLER
    return tex_sampler_map.get(tex_index, _DEFAULT_GLTF_SAMPLER)


def _texture_transform(tex_info):
    if not isinstance(tex_info, dict):
        return None
    extensions = tex_info.get('extensions', {})
    if not isinstance(extensions, dict):
        return None
    transform = extensions.get('KHR_texture_transform')
    return transform if isinstance(transform, dict) else None


def _append_spec_gloss_mr_texture(all_textures, tex_img_map, spec_gloss_index, glossiness_factor, cache):
    """Convert specularGlossiness alpha into a glTF metallicRoughness texture."""
    src_id = _texture_image_id(tex_img_map, all_textures, spec_gloss_index)
    if src_id < 0:
        return -1
    glossiness = _clamp_float(glossiness_factor, 0.0, 1.0, 1.0)
    cache_key = (spec_gloss_index, glossiness)
    if cache_key in cache:
        return cache[cache_key]
    src = all_textures[src_id]
    alpha = src[:, :, 3].astype(np.float32) / 255.0
    roughness = np.clip(1.0 - alpha * glossiness, 0.0, 1.0)
    mr = np.empty_like(src)
    mr[:, :, 0] = 255
    mr[:, :, 1] = np.rint(roughness * 255.0).astype(np.uint8)
    mr[:, :, 2] = 0
    mr[:, :, 3] = 255
    mr_id = len(all_textures)
    all_textures.append(mr)
    cache[cache_key] = mr_id
    return mr_id


def _coerce_vec_array(values, rows, cols, fill=0.0):
    out = np.full((rows, cols), fill, dtype=np.float32)
    try:
        arr = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return out
    if arr.ndim != 2 or arr.shape[0] != rows:
        return out
    ncols = min(cols, arr.shape[1])
    if ncols > 0:
        out[:, :ncols] = arr[:, :ncols]
    return out


def normalize_gltf_sampler(sampler):
    if not isinstance(sampler, dict):
        return _DEFAULT_GLTF_SAMPLER
    mag_filter = _safe_int(sampler.get('magFilter'), _DEFAULT_GLTF_SAMPLER[0])
    min_filter = _safe_int(sampler.get('minFilter'), _DEFAULT_GLTF_SAMPLER[1])
    wrap_s = _safe_int(sampler.get('wrapS'), _DEFAULT_GLTF_SAMPLER[2])
    wrap_t = _safe_int(sampler.get('wrapT'), _DEFAULT_GLTF_SAMPLER[3])
    if mag_filter not in _VALID_GLTF_MAG_FILTERS:
        mag_filter = _DEFAULT_GLTF_SAMPLER[0]
    if min_filter not in _VALID_GLTF_MIN_FILTERS:
        min_filter = _DEFAULT_GLTF_SAMPLER[1]
    if wrap_s not in _VALID_GLTF_WRAPS:
        wrap_s = _DEFAULT_GLTF_SAMPLER[2]
    if wrap_t not in _VALID_GLTF_WRAPS:
        wrap_t = _DEFAULT_GLTF_SAMPLER[3]
    return (
        mag_filter,
        min_filter,
        wrap_s,
        wrap_t,
    )


def gltf_texture_cache_key(prefix, image_id, sampler):
    mag_filter, min_filter, wrap_s, wrap_t = normalize_gltf_sampler(sampler)
    return f"{prefix}:{int(image_id)}:{mag_filter}:{min_filter}:{wrap_s}:{wrap_t}"


def apply_gltf_sampler_to_texture(texture, sampler):
    mag_filter, min_filter, wrap_s, wrap_t = normalize_gltf_sampler(sampler)
    mag_map = {
        9728: moderngl.NEAREST,
        9729: moderngl.LINEAR,
    }
    min_map = {
        9728: moderngl.NEAREST,
        9729: moderngl.LINEAR,
        9984: moderngl.NEAREST_MIPMAP_NEAREST,
        9985: moderngl.LINEAR_MIPMAP_NEAREST,
        9986: moderngl.NEAREST_MIPMAP_LINEAR,
        9987: moderngl.LINEAR_MIPMAP_LINEAR,
    }
    texture.filter = (
        min_map.get(min_filter, moderngl.LINEAR_MIPMAP_LINEAR),
        mag_map.get(mag_filter, moderngl.LINEAR),
    )
    # ModernGL exposes repeat/clamp booleans; mirrored repeat is approximated as repeat.
    texture.repeat_x = wrap_s != 33071
    texture.repeat_y = wrap_t != 33071


def _is_foliage_material_name(material_name):
    material_name_l = str(material_name or '').lower()
    return (
        'plant' in material_name_l
        or 'leaf' in material_name_l
        or 'leaves' in material_name_l
        or 'foliage' in material_name_l
        or 'grass' in material_name_l
        or 'bush' in material_name_l
        or 'tree' in material_name_l
    )


def _default_gltf_material_fields():
    return {
        'tex_id': -1,
        'base_color': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'base_sampler': _DEFAULT_GLTF_SAMPLER,
        'base_texcoord': 0,
        'base_alpha': 1.0,
        'roughness_factor': 1.0,
        'metallic_factor': 1.0,
        'emissive_factor': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'normal_tex_id': -1,
        'normal_sampler': _DEFAULT_GLTF_SAMPLER,
        'normal_texcoord': 0,
        'normal_scale': 1.0,
        'occlusion_tex_id': -1,
        'occlusion_sampler': _DEFAULT_GLTF_SAMPLER,
        'occlusion_texcoord': 0,
        'occlusion_strength': 1.0,
        'unlit': False,
        'alpha_mode': 'OPAQUE',
        'alpha_cutoff': 0.5,
        'mr_tex_id': -1,
        'mr_sampler': _DEFAULT_GLTF_SAMPLER,
        'mr_texcoord': 0,
        'emissive_tex_id': -1,
        'emissive_sampler': _DEFAULT_GLTF_SAMPLER,
        'emissive_texcoord': 0,
        'double_sided': False,
        'tex_offset': np.array([0.0, 0.0], dtype=np.float32),
        'tex_scale': np.array([1.0, 1.0], dtype=np.float32),
        'tex_rotation': 0.0,
        'foliage_mode': False,
    }


def _material_contract_from_fields(fields):
    texture_slots = {}
    for binding in GLTF_MATERIAL_TEXTURE_BINDINGS:
        image_id = int(fields.get(binding.source_tex_field, -1))
        if image_id < 0:
            continue
        transform = TextureTransform()
        if binding.role == 'base':
            transform = TextureTransform(
                offset=tuple(float(v) for v in np.asarray(fields.get('tex_offset', (0.0, 0.0)), dtype=np.float32)[:2]),
                scale=tuple(float(v) for v in np.asarray(fields.get('tex_scale', (1.0, 1.0)), dtype=np.float32)[:2]),
                rotation=float(fields.get('tex_rotation', 0.0) or 0.0),
            )
        texture_slots[binding.role] = TextureBinding(
            image_id=image_id,
            sampler=tuple(int(v) for v in fields.get(binding.source_sampler_field, _DEFAULT_GLTF_SAMPLER)),
            texcoord=_safe_texcoord(fields.get(f'{binding.role}_texcoord'), 0),
            transform=transform,
            color_space=binding.color_space,
        )
    return GltfMaterial(
        base_color=tuple(float(v) for v in np.asarray(fields.get('base_color', (1.0, 1.0, 1.0)), dtype=np.float32)[:3]),
        base_alpha=float(fields.get('base_alpha', 1.0)),
        alpha_mode=fields.get('alpha_mode', 'OPAQUE'),
        alpha_cutoff=float(fields.get('alpha_cutoff', 0.5)),
        double_sided=bool(fields.get('double_sided', False)),
        unlit=bool(fields.get('unlit', False)),
        texture_slots=texture_slots,
        roughness=float(fields.get('roughness_factor', 1.0)),
        metallic=float(fields.get('metallic_factor', 1.0)),
        normal_scale=float(fields.get('normal_scale', 1.0)),
        occlusion_strength=float(fields.get('occlusion_strength', 1.0)),
        emissive_factor=tuple(float(v) for v in np.asarray(fields.get('emissive_factor', (0.0, 0.0, 0.0)), dtype=np.float32)[:3]),
    )


def _set_texture_transform(fields, tex_info, texcoord_key):
    tx_ext = _texture_transform(tex_info)
    if not tx_ext:
        return
    if 'texCoord' in tx_ext:
        fields[texcoord_key] = _safe_texcoord(tx_ext.get('texCoord'), fields[texcoord_key])
    if texcoord_key != 'base_texcoord':
        return
    if isinstance(tx_ext.get('offset'), (list, tuple)) and len(tx_ext['offset']) >= 2:
        fields['tex_offset'] = np.array([
            _safe_float(tx_ext['offset'][0], 0.0),
            _safe_float(tx_ext['offset'][1], 0.0),
        ], dtype=np.float32)
    if isinstance(tx_ext.get('scale'), (list, tuple)) and len(tx_ext['scale']) >= 2:
        fields['tex_scale'] = np.array([
            _safe_float(tx_ext['scale'][0], 1.0),
            _safe_float(tx_ext['scale'][1], 1.0),
        ], dtype=np.float32)
    if 'rotation' in tx_ext:
        fields['tex_rotation'] = _safe_float(tx_ext.get('rotation'), 0.0)


def _get_accessor(gltf, bin_data, acc_idx):
    """Extract numpy array from a glTF accessor.
    Handles both contiguous and interleaved (byteStride) vertex attributes.
    """
    accessors = gltf.get('accessors', [])
    if not isinstance(accessors, list) or not isinstance(acc_idx, int) or acc_idx < 0 or acc_idx >= len(accessors):
        raise ValueError(f"Invalid accessor index: {acc_idx}")
    acc = accessors[acc_idx]
    if not isinstance(acc, dict):
        raise ValueError(f"Invalid accessor object: {acc_idx}")
    if acc.get('type') not in _TYPE_NC:
        raise ValueError(f"Unsupported accessor type: {acc.get('type')}")
    if acc.get('componentType') not in _DTYPE_MAP:
        raise ValueError(f"Unsupported accessor componentType: {acc.get('componentType')}")
    count = _safe_int(acc.get('count'), -1)
    if count < 0:
        raise ValueError(f"Invalid accessor count: {acc.get('count')}")
    nc = _TYPE_NC[acc['type']]
    dt = np.dtype(_DTYPE_MAP[acc['componentType']]).newbyteorder('<')
    elem_size = nc * dt.itemsize

    if 'bufferView' in acc:
        buffer_views = gltf.get('bufferViews', [])
        bv_idx = acc.get('bufferView')
        if not isinstance(buffer_views, list) or not isinstance(bv_idx, int) or bv_idx < 0 or bv_idx >= len(buffer_views):
            raise ValueError(f"Invalid bufferView index: {bv_idx}")
        bv = buffer_views[bv_idx]
        if not isinstance(bv, dict):
            raise ValueError(f"Invalid bufferView object: {bv_idx}")
        byte_offset = _safe_int(bv.get('byteOffset'), 0) + _safe_int(acc.get('byteOffset'), 0)
        byte_stride = _safe_int(bv.get('byteStride'), 0)
        if byte_offset < 0 or byte_stride < 0:
            raise ValueError("Negative accessor byte offset or stride")
        if byte_stride and byte_stride < elem_size:
            raise ValueError(f"Accessor byteStride smaller than element size: {byte_stride} < {elem_size}")
        required_bytes = elem_size * count if (byte_stride == 0 or byte_stride == elem_size or count == 0) else byte_stride * (count - 1) + elem_size
        buffer_data = _buffer_data(bin_data, bv.get('buffer', 0))
        if buffer_data is None or byte_offset + required_bytes > len(buffer_data):
            raise ValueError("Accessor buffer range exceeds buffer data")
        if byte_stride == 0 or byte_stride == elem_size:
            # Contiguous (no stride or stride equals element size)
            arr = np.frombuffer(buffer_data, dtype=dt, count=count * nc,
                               offset=byte_offset).copy()
        else:
            # Interleaved vertex attributes -read each row with stride
            arr = np.ndarray(shape=(count, nc), dtype=dt,
                             buffer=buffer_data,
                             offset=byte_offset,
                             strides=(byte_stride, dt.itemsize)).copy()
    else:
        arr = np.zeros(count * nc, dtype=dt)
    if nc > 1:
        arr = arr.reshape(count, nc)

    sparse = acc.get('sparse')
    if sparse:
        if not isinstance(sparse, dict):
            raise ValueError("Invalid sparse accessor object")
        sparse_count = _safe_int(sparse.get('count'), 0)
        if sparse_count < 0:
            raise ValueError(f"Invalid sparse accessor count: {sparse.get('count')}")
        indices_info = sparse.get('indices', {})
        values_info = sparse.get('values', {})
        if indices_info.get('bufferView') is None or values_info.get('bufferView') is None:
            raise ValueError("Sparse accessor missing bufferView")
        buffer_views = gltf.get('bufferViews', [])
        index_bv_idx = indices_info.get('bufferView')
        value_bv_idx = values_info.get('bufferView')
        if not isinstance(index_bv_idx, int) or index_bv_idx < 0 or index_bv_idx >= len(buffer_views):
            raise ValueError(f"Invalid sparse index bufferView: {index_bv_idx}")
        if not isinstance(value_bv_idx, int) or value_bv_idx < 0 or value_bv_idx >= len(buffer_views):
            raise ValueError(f"Invalid sparse value bufferView: {value_bv_idx}")
        index_bv = buffer_views[index_bv_idx]
        if indices_info.get('componentType') not in (5121, 5123, 5125):
            raise ValueError(f"Unsupported sparse index componentType: {indices_info.get('componentType')}")
        index_dt = np.dtype(_DTYPE_MAP[indices_info['componentType']]).newbyteorder('<')
        index_offset = _safe_int(index_bv.get('byteOffset'), 0) + _safe_int(indices_info.get('byteOffset'), 0)
        index_required = sparse_count * index_dt.itemsize
        index_buffer = _buffer_data(bin_data, index_bv.get('buffer', 0))
        if index_buffer is None or index_offset < 0 or index_offset + index_required > len(index_buffer):
            raise ValueError("Sparse index buffer range exceeds buffer data")
        sparse_indices = np.frombuffer(
            index_buffer, dtype=index_dt, count=sparse_count, offset=index_offset
        ).astype(np.uint32)
        if sparse_indices.size and int(sparse_indices.max()) >= count:
            raise ValueError("Sparse accessor index out of range")

        value_bv = buffer_views[value_bv_idx]
        value_offset = _safe_int(value_bv.get('byteOffset'), 0) + _safe_int(values_info.get('byteOffset'), 0)
        value_required = sparse_count * nc * dt.itemsize
        value_buffer = _buffer_data(bin_data, value_bv.get('buffer', 0))
        if value_buffer is None or value_offset < 0 or value_offset + value_required > len(value_buffer):
            raise ValueError("Sparse value buffer range exceeds buffer data")
        sparse_values = np.frombuffer(
            value_buffer, dtype=dt, count=sparse_count * nc, offset=value_offset
        ).copy()
        if nc > 1:
            sparse_values = sparse_values.reshape(sparse_count, nc)
        arr[sparse_indices] = sparse_values

    component_type = acc['componentType']
    if acc.get('normalized', False) and component_type in (5120, 5121, 5122, 5123, 5125):
        arr = arr.astype(np.float32)
        if component_type == 5120:
            arr = np.maximum(arr / 127.0, -1.0)
        elif component_type == 5121:
            arr = arr / 255.0
        elif component_type == 5122:
            arr = np.maximum(arr / 32767.0, -1.0)
        elif component_type == 5123:
            arr = arr / 65535.0
        elif component_type == 5125:
            arr = arr / 4294967295.0
    elif component_type in (5121, 5123, 5125):
        arr = arr.astype(np.uint32)
    elif component_type == 5126:
        arr = arr.astype(np.float32)
    return arr


def _quat_to_mat4(q):
    """Convert quaternion [x, y, z, w] to 4x4 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),   0],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),   0],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy), 0],
        [0,           0,           0,           1],
    ], dtype=np.float64)


def _node_local_matrix(node):
    matrix = node.get('matrix')
    if isinstance(matrix, list) and len(matrix) == 16:
        # glTF stores matrices in column-major order.
        return np.array(matrix, dtype=np.float64).reshape((4, 4)).T

    t = node.get('translation', [0, 0, 0])
    r = node.get('rotation', [0, 0, 0, 1])  # [x, y, z, w]
    s = node.get('scale', [1, 1, 1])

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = t
    R = _quat_to_mat4(r)
    S_mat = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float64)
    return T @ R @ S_mat


def _build_node_matrices(gltf):
    """Compute world matrix for each node (top-down). Returns list of 4x4 float64 matrices.
    Parent world matrix = parent_matrix @ local_matrix.
    Local matrix = translation * rotation * scale.
    Root nodes assume identity parent matrix.
    """
    nodes = gltf.get('nodes', [])
    n = len(nodes)
    if n == 0:
        return []

    # Build local matrices
    local_mats = [_node_local_matrix(node) for node in nodes]

    # Build child -> parent mapping
    parent = [-1] * n
    for pi, node in enumerate(nodes):
        for ci in node.get('children', []):
            if isinstance(ci, int) and 0 <= ci < n:
                parent[ci] = pi

    # Topological order (BFS from roots) to compute world matrices
    world_mats = [None] * n
    queue = [i for i in range(n) if parent[i] == -1]
    for i in queue:
        world_mats[i] = local_mats[i].copy()

    head = 0
    while head < len(queue):
        pi = queue[head]
        head += 1
        for ci in nodes[pi].get('children', []):
            if not isinstance(ci, int) or ci < 0 or ci >= n:
                continue
            if world_mats[ci] is None:
                world_mats[ci] = world_mats[pi] @ local_mats[ci]
                queue.append(ci)

    # Isolated nodes (no parent) just use local matrix
    for i in range(n):
        if world_mats[i] is None:
            world_mats[i] = local_mats[i].copy()

    return world_mats


def _iter_scene_mesh_nodes(gltf, world_mats):
    """Yield mesh node records reachable from the active scene."""
    nodes = gltf.get('nodes', [])
    scenes = gltf.get('scenes', [])
    scene_idx = gltf.get('scene', 0)
    if isinstance(scene_idx, int) and 0 <= scene_idx < len(scenes):
        roots = scenes[scene_idx].get('nodes', [])
    else:
        roots = [i for i, node in enumerate(nodes) if node.get('mesh') is not None]

    stack = list(reversed([i for i in roots if isinstance(i, int) and 0 <= i < len(nodes)]))
    visited = set()
    while stack:
        ni = stack.pop()
        if ni in visited:
            continue
        visited.add(ni)
        node = nodes[ni]
        mi = node.get('mesh')
        if isinstance(mi, int):
            meshes = gltf.get('meshes', [])
            mesh_name = str(meshes[mi].get('name') or '') if 0 <= mi < len(meshes) else ''
            yield mi, world_mats[ni], ni, str(node.get('name') or ''), mesh_name
        children = node.get('children', [])
        stack.extend(reversed([ci for ci in children if isinstance(ci, int) and 0 <= ci < len(nodes)]))


def _apply_transform(vertices_xyz, matrix_4x4):
    """Apply 4x4 transformation matrix to vertex positions."""
    n = vertices_xyz.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    v4 = np.hstack([vertices_xyz.astype(np.float64), ones])
    t = (matrix_4x4 @ v4.T).T
    return t[:, :3].astype(np.float32)


def _apply_normal_transform(vectors_xyz, matrix_4x4):
    """Apply inverse-transpose normal transform and normalize the result."""
    rot3 = matrix_4x4[:3, :3].astype(np.float64)
    normal_mat = np.linalg.inv(rot3.T)
    out = (normal_mat @ vectors_xyz.astype(np.float64).T).T
    lens = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.maximum(lens, 1e-8)
    return out.astype(np.float32)


def _orthogonalize_tangent(tangent_xyz, normal_xyz):
    """Project tangent away from normal and normalize it."""
    tangent_xyz = tangent_xyz.astype(np.float32)
    normal_xyz = normal_xyz.astype(np.float32)
    tangent_xyz = tangent_xyz - normal_xyz * np.sum(tangent_xyz * normal_xyz, axis=1, keepdims=True)
    tangent_xyz /= np.maximum(np.linalg.norm(tangent_xyz, axis=1, keepdims=True), 1e-8)
    return tangent_xyz.astype(np.float32)


def load_glb_model(path):
    """Load a glTF/GLB model, apply node transformations.
    Returns:
        primitives: list of dict with keys:
            vertices (N, 10 float32: position3, normal3, uv0, uv1)
            tangent (N, 4 float32: tangent xyz, bitangent sign)
            indices (M, uint32)
            material_contract (GltfMaterial)
            gltf_primitive (GltfPrimitive)
            render_pass (opaque, mask, transparent, or sky)
        textures: list of numpy RGBA uint8 arrays
    """
    _mat_log = open(os.devnull, 'w', encoding='utf-8')
    _mat_log.write(f"=== Material debug for: {path} ===\n")
    gltf, bin_data = _load_gltf_document(path)
    _raise_unsupported_required_extensions(gltf, path)
    base_dir = os.path.dirname(os.path.abspath(path))

    # World matrices for all nodes
    world_mats = _build_node_matrices(gltf)
    nodes = gltf.get('nodes', [])
    local_mats = [_node_local_matrix(node) for node in nodes]
    parent = [-1] * len(nodes)
    name_to_node = {}
    for ni, node in enumerate(nodes):
        name = node.get('name')
        if name:
            name_to_node[str(name)] = ni
        for child in node.get('children', []):
            if isinstance(child, int) and 0 <= child < len(nodes):
                parent[child] = ni

    def _semantic_from_value_name(value_name, node_name=''):
        value_l = value_name.lower()
        node_l = node_name.lower()
        if 'thumbstick_xaxis' in value_l:
            return 'joystick_x'
        if 'thumbstick_yaxis' in value_l:
            return 'joystick_y'
        if 'thumbstick_pressed' in value_l:
            return 'joystick'
        if 'thumbstick' in value_l and 'touched' in value_l:
            return 'joystick_touched'
        if 'touchpad_xaxis' in value_l:
            return 'touchpad_x'
        if 'touchpad_yaxis' in value_l:
            return 'touchpad_y'
        if 'touchpad_pressed' in value_l:
            return 'touchpad'
        if 'touchpad' in value_l and 'touched' in value_l:
            return 'touchpad_touched'
        if 'trigger' in value_l:
            return 'trigger'
        if 'squeeze' in value_l or 'grasp' in value_l or 'grip' in node_l:
            return 'grip'
        if 'a_button' in value_l or node_l.endswith('abutton') or node_l in ('a_button', 'a_button_mesh'):
            return 'a_button'
        if 'b_button' in value_l or node_l.endswith('bbutton') or node_l in ('b_button', 'b_button_mesh'):
            return 'b_button'
        if 'x_button' in value_l or node_l.endswith('xbutton') or node_l in ('x_button', 'x_button_mesh'):
            return 'x_button'
        if 'y_button' in value_l or node_l.endswith('ybutton') or node_l in ('y_button', 'y_button_mesh'):
            return 'y_button'
        if 'menu' in value_l or 'menu' in node_l:
            return 'menu_button'
        if 'home' in value_l or 'pico' in value_l or 'home' in node_l:
            return 'home_button'
        return ''

    def _anim_from_value_node(node_index, value_node_index):
        value_name = str(nodes[value_node_index].get('name') or '')
        if not value_name.endswith('_value'):
            return None
        prefix = value_name[:-len('_value')]
        min_index = name_to_node.get(prefix + '_min')
        max_index = name_to_node.get(prefix + '_max')
        if min_index is None or max_index is None:
            return None
        mesh_world = world_mats[node_index].astype(np.float32)
        value_world = world_mats[value_node_index].astype(np.float32)
        try:
            child_local = (np.linalg.inv(value_world.astype(np.float64)) @ mesh_world.astype(np.float64)).astype(np.float32)
            inv_mesh_world = np.linalg.inv(mesh_world.astype(np.float64)).astype(np.float32)
        except Exception:
            return None
        parent_index = parent[value_node_index] if value_node_index < len(parent) else -1
        value_parent_world = world_mats[parent_index] if parent_index >= 0 else np.eye(4, dtype=np.float64)
        return {
            'value_name': value_name,
            'semantic': _semantic_from_value_name(value_name, str(nodes[node_index].get('name') or '')),
            'value_world': world_mats[value_node_index].astype(np.float32),
            'min_world': world_mats[min_index].astype(np.float32),
            'max_world': world_mats[max_index].astype(np.float32),
            'value_parent_world': value_parent_world.astype(np.float32),
            'value_local': local_mats[value_node_index].astype(np.float32),
            'min_local': local_mats[min_index].astype(np.float32),
            'max_local': local_mats[max_index].astype(np.float32),
            'child_local': child_local,
            'inv_mesh_world': inv_mesh_world,
        }

    def _press_anim_for_mesh_node(node_index):
        node_name = str(nodes[node_index].get('name') or '') if 0 <= node_index < len(nodes) else ''
        node_l = node_name.lower()
        candidates = []
        parent_index = parent[node_index] if 0 <= node_index < len(parent) else -1
        while parent_index >= 0:
            value_name = str(nodes[parent_index].get('name') or '')
            if value_name.endswith('_value'):
                prefix = value_name[:-len('_value')]
                min_index = name_to_node.get(prefix + '_min')
                max_index = name_to_node.get(prefix + '_max')
                if min_index is not None and max_index is not None:
                    candidates.append((parent_index, value_name, min_index, max_index))
            parent_index = parent[parent_index] if parent_index < len(parent) else -1
        if not candidates:
            return None
        is_stick = (
            'joystick' in node_l
            or 'thumbstick' in node_l
            or 'touchpad' in node_l
            or any(('thumbstick' in candidate[1].lower() or 'touchpad' in candidate[1].lower()) for candidate in candidates)
        )
        if is_stick:
            for candidate in candidates:
                if candidate[1].endswith('thumbstick_pressed_value') or candidate[1].endswith('touchpad_pressed_value'):
                    return _anim_from_value_node(node_index, candidate[0])
            return None
        return _anim_from_value_node(node_index, candidates[0][0])

    def _visible_key_for_mesh_node(node_index):
        parent_index = parent[node_index] if 0 <= node_index < len(parent) else -1
        while parent_index >= 0:
            value_name = str(nodes[parent_index].get('name') or '')
            if value_name.endswith('_value') and 'touched' in value_name.lower():
                return _semantic_from_value_name(value_name, str(nodes[node_index].get('name') or ''))
            parent_index = parent[parent_index] if parent_index < len(parent) else -1
        return ''

    def _axis_anim_for_mesh_node(node_index):
        node_name = str(nodes[node_index].get('name') or '') if 0 <= node_index < len(nodes) else ''
        node_l = node_name.lower()
        if 'joystick' not in node_l and 'thumbstick' not in node_l and 'touchpad' not in node_l:
            parent_index = parent[node_index] if 0 <= node_index < len(parent) else -1
            has_stick_axis = False
            while parent_index >= 0:
                value_name = str(nodes[parent_index].get('name') or '').lower()
                if 'thumbstick_' in value_name or 'touchpad_' in value_name:
                    has_stick_axis = True
                    break
                parent_index = parent[parent_index] if parent_index < len(parent) else -1
            if not has_stick_axis:
                return None
        parent_index = parent[node_index] if 0 <= node_index < len(parent) else -1
        result = {}
        while parent_index >= 0:
            value_name = str(nodes[parent_index].get('name') or '')
            if value_name.endswith('thumbstick_xaxis_pressed_value') or value_name.endswith('touchpad_xaxis_pressed_value'):
                result['x'] = _anim_from_value_node(node_index, parent_index)
            elif value_name.endswith('thumbstick_yaxis_pressed_value') or value_name.endswith('touchpad_yaxis_pressed_value'):
                result['y'] = _anim_from_value_node(node_index, parent_index)
            parent_index = parent[parent_index] if parent_index < len(parent) else -1
        return result or None

    # Map mesh index to all node instances that reference it.  glTF allows
    # table legs, curtains, string lights, etc. to reuse one mesh from many
    # nodes; the old loader kept only the first node and dropped/misplaced the
    # rest.
    mesh_world_mat = {}
    mesh_world_mats = {}
    mesh_node_meta = {}
    for mi, world_mat_for_node, node_index, node_name, mesh_name in _iter_scene_mesh_nodes(gltf, world_mats):
        press_anim = _press_anim_for_mesh_node(node_index)
        axis_anim = _axis_anim_for_mesh_node(node_index)
        visible_key = _visible_key_for_mesh_node(node_index)
        mesh_world_mats.setdefault(mi, []).append((world_mat_for_node, node_index, node_name, mesh_name, press_anim, axis_anim, visible_key))
        if mi not in mesh_world_mat:
            mesh_world_mat[mi] = world_mat_for_node
            mesh_node_meta[mi] = (node_index, node_name, mesh_name, press_anim, axis_anim, visible_key)

    # Extract textures
    all_textures = []
    if 'images' in gltf:
        for img in gltf['images']:
            tex_data = None
            if isinstance(img, dict) and 'bufferView' in img:
                buffer_views = gltf.get('bufferViews', [])
                bv_idx = img.get('bufferView')
                if isinstance(bv_idx, int) and 0 <= bv_idx < len(buffer_views):
                    bv = buffer_views[bv_idx]
                    off = _safe_int(bv.get('byteOffset'), 0)
                    byte_len = _safe_int(bv.get('byteLength'), 0)
                    image_buffer = _buffer_data(bin_data, bv.get('buffer', 0))
                    if image_buffer is not None and off >= 0 and byte_len > 0 and off + byte_len <= len(image_buffer):
                        tex_data = image_buffer[off:off + byte_len]
            elif isinstance(img, dict) and 'uri' in img and img['uri'].startswith('data:'):
                tex_data = _decode_data_uri(img['uri'])
            elif isinstance(img, dict) and 'uri' in img:
                uri = img['uri']
                parsed = urllib.parse.urlparse(uri)
                if parsed.scheme in ('', 'file'):
                    rel_path = urllib.parse.unquote(parsed.path if parsed.scheme == 'file' else uri)
                    rel_path = rel_path.replace('/', os.sep)
                    tex_path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
                    if os.path.exists(tex_path):
                        with open(tex_path, 'rb') as tf:
                            tex_data = tf.read()
            if tex_data:
                pil_img = Image.open(_io.BytesIO(tex_data))
                pil_img = pil_img.convert('RGBA')
                all_textures.append(np.array(pil_img, dtype=np.uint8))
            else:
                all_textures.append(None)

    # Map texture index to image index
    tex_img_map = {}
    tex_sampler_map = {}
    if 'textures' in gltf:
        for ti, tex in enumerate(gltf['textures']):
            tex = tex if isinstance(tex, dict) else {}
            si = tex.get('source', 0)
            tex_img_map[ti] = si if isinstance(si, int) and 0 <= si < len(all_textures) else -1
            sampler_idx = tex.get('sampler')
            sampler = None
            if isinstance(sampler_idx, int) and 0 <= sampler_idx < len(gltf.get('samplers', [])):
                sampler = gltf['samplers'][sampler_idx]
            tex_sampler_map[ti] = normalize_gltf_sampler(sampler)
    spec_gloss_mr_cache = {}

    primitives = []
    for mi, mesh in enumerate(gltf.get('meshes', [])):
        if mi not in mesh_world_mats:
            continue
        world_mat = mesh_world_mat.get(mi, np.eye(4, dtype=np.float64))
        node_index, node_name, mesh_name, press_anim, axis_anim, visible_key = mesh_node_meta.get(mi, (-1, '', str(mesh.get('name') or ''), None, None, ''))
        for prim in mesh.get('primitives', []):
            attrs = prim.get('attributes', {})
            if 'POSITION' not in attrs:
                continue
            try:
                pos = _get_accessor(gltf, bin_data, attrs['POSITION'])
            except Exception as exc:
                _mat_log.write(f"[PRIM] skip mesh={mi}: invalid POSITION ({exc})\n")
                continue
            if pos.ndim != 2 or pos.shape[1] < 3 or pos.shape[0] == 0:
                _mat_log.write(f"[PRIM] skip mesh={mi}: POSITION must be non-empty VEC3\n")
                continue
            pos = pos[:, :3].astype(np.float32, copy=False)

            # Extract normals if present, else zeros
            if 'NORMAL' in attrs:
                try:
                    norm = _get_accessor(gltf, bin_data, attrs['NORMAL'])
                except Exception:
                    norm = np.zeros((pos.shape[0], 3), dtype=np.float32)
            else:
                norm = np.zeros((pos.shape[0], 3), dtype=np.float32)
            norm = _coerce_vec_array(norm, pos.shape[0], 3, 0.0)

            # Extract tangent (vec4: xyz + bitangent_sign), or zeros if absent
            if 'TANGENT' in attrs:
                try:
                    tangent = _get_accessor(gltf, bin_data, attrs['TANGENT'])
                except Exception:
                    tangent = np.zeros((pos.shape[0], 4), dtype=np.float32)
                    tangent[:, 3] = 1.0
                tangent = _coerce_vec_array(tangent, pos.shape[0], 4, 0.0)
                if tangent.shape[0] > 0:
                    tangent[:, 3] = np.where(np.abs(tangent[:, 3]) > 1e-8, tangent[:, 3], 1.0)
            else:
                tangent = np.zeros((pos.shape[0], 4), dtype=np.float32)
                tangent[:, 3] = 1.0  # bitangent sign defaults to 1

            # Apply node world matrix: position with full 4x4, normals with inverse-transpose
            if not np.allclose(world_mat, np.eye(4)):
                pos = _apply_transform(pos, world_mat)
                rot3 = world_mat[:3, :3].astype(np.float64)
                normal_mat = np.linalg.inv(rot3.T)  # inverse-transpose handles non-uniform scaling
                norm = (normal_mat @ norm.T).T.astype(np.float32)
                norm /= np.maximum(np.linalg.norm(norm, axis=1, keepdims=True), 1e-8)
                # Transform tangent xyz with rotation, keep w (bitangent sign)
                if tangent is not None:
                    t_xyz = (rot3[:3, :3].astype(np.float64) @ tangent[:, :3].T).T.astype(np.float32)
                    t_xyz = _orthogonalize_tangent(t_xyz, norm)
                    tangent = np.hstack([t_xyz, tangent[:, 3:4]]).astype(np.float32)

            # Extract UV coordinates. Keep UV1 for glTF textureInfo.texCoord=1 lightmaps.
            if 'TEXCOORD_0' in attrs:
                try:
                    uv = _get_accessor(gltf, bin_data, attrs['TEXCOORD_0'])
                except Exception:
                    uv = np.zeros((pos.shape[0], 2), dtype=np.float32)
            else:
                uv = np.zeros((pos.shape[0], 2), dtype=np.float32)
            uv = _coerce_vec_array(uv, pos.shape[0], 2, 0.0)

            has_uv1 = 'TEXCOORD_1' in attrs
            if has_uv1:
                try:
                    uv1 = _get_accessor(gltf, bin_data, attrs['TEXCOORD_1'])
                except Exception:
                    uv1 = uv.copy()
            else:
                uv1 = uv.copy()
            uv1 = _coerce_vec_array(uv1, pos.shape[0], 2, 0.0)

            uv_min = uv.min(axis=0) if uv.size else np.array([0.0, 0.0], dtype=np.float32)
            uv_max = uv.max(axis=0) if uv.size else np.array([0.0, 0.0], dtype=np.float32)

            # Combine: position (3), normal (3), uv0 (2), uv1 (2) -> 10 floats
            vertices = np.hstack([pos, norm, uv, uv1]).astype(np.float32)

            # Indices
            if 'indices' in prim:
                try:
                    indices = _get_accessor(gltf, bin_data, prim['indices']).reshape(-1).astype(np.uint32, copy=False)
                except Exception:
                    indices = np.arange(pos.shape[0], dtype=np.uint32)
            else:
                indices = np.arange(pos.shape[0], dtype=np.uint32)
            if indices.size == 0 or int(indices.max()) >= pos.shape[0]:
                indices = np.arange(pos.shape[0], dtype=np.uint32)

            material_fields = parse_gltf_material(
                gltf,
                prim.get('material'),
                tex_img_map=tex_img_map,
                tex_sampler_map=tex_sampler_map,
                all_textures=all_textures,
                uv_min=uv_min,
                uv_max=uv_max,
                spec_gloss_mr_cache=spec_gloss_mr_cache,
                log_writer=_mat_log,
            )
            primitive_record = {'vertices': vertices, 'indices': indices,
                                'primitive_mode': _safe_int(prim.get('mode'), 4),
                                'has_uv1': has_uv1,
                                'tangent': tangent,
                                'node_index': node_index,
                                'node_name': node_name,
                                'mesh_name': mesh_name,
                                'press_anim': press_anim,
                                'axis_anim': axis_anim,
                                'anim_key': press_anim.get('semantic', '') if press_anim else '',
                                'visible_key': visible_key,
                                '_mesh_index': mi,
                                '_world_matrix': world_mat}
            primitive_record.update(material_fields)
            primitives.append(primitive_record)

    extra_instances = []
    for primitive in primitives:
        mi = primitive.get('_mesh_index')
        instances = mesh_world_mats.get(mi, [])
        if len(instances) <= 1:
            continue

        first_world = primitive.get('_world_matrix', np.eye(4, dtype=np.float64)).astype(np.float64)
        try:
            inv_first_world = np.linalg.inv(first_world)
        except Exception:
            continue

        local_positions = _apply_transform(primitive['vertices'][:, :3], inv_first_world)
        first_rot = first_world[:3, :3].astype(np.float64)
        local_normals = (first_rot.T @ primitive['vertices'][:, 3:6].astype(np.float64).T).T
        local_normals /= np.maximum(np.linalg.norm(local_normals, axis=1, keepdims=True), 1e-8)

        tangent = primitive.get('tangent')
        if tangent is not None:
            local_tangent = tangent.copy()
            local_tangent[:, :3] = (first_rot.T @ tangent[:, :3].astype(np.float64).T).T.astype(np.float32)
            local_tangent[:, :3] = _orthogonalize_tangent(local_tangent[:, :3], local_normals)
        else:
            local_tangent = None

        for inst_world, node_index, node_name, mesh_name, press_anim, axis_anim, visible_key in instances[1:]:
            inst_world = inst_world.astype(np.float64)
            clone = dict(primitive)
            clone_vertices = primitive['vertices'].copy()
            clone_vertices[:, :3] = _apply_transform(local_positions, inst_world)
            clone_vertices[:, 3:6] = _apply_normal_transform(local_normals, inst_world)
            clone['vertices'] = clone_vertices
            clone['indices'] = primitive['indices'].copy()
            if local_tangent is not None:
                inst_tangent = local_tangent.copy()
                inst_tangent[:, :3] = (inst_world[:3, :3].astype(np.float64) @ local_tangent[:, :3].astype(np.float64).T).T.astype(np.float32)
                inst_tangent[:, :3] = _orthogonalize_tangent(inst_tangent[:, :3], clone_vertices[:, 3:6])
                clone['tangent'] = inst_tangent
            clone['_world_matrix'] = inst_world
            clone['node_index'] = node_index
            clone['node_name'] = node_name
            clone['mesh_name'] = mesh_name
            clone['press_anim'] = press_anim
            clone['axis_anim'] = axis_anim
            clone['anim_key'] = press_anim.get('semantic', '') if press_anim else ''
            clone['visible_key'] = visible_key
            extra_instances.append(clone)

    if extra_instances:
        primitives.extend(extra_instances)
        _mat_log.write(f"[INSTANCE] Added {len(extra_instances)} mesh node instances\n")

    # Freeze the renderer-facing mesh/material/pass contract after all node
    # instances have received their final world-space geometry.
    for primitive in primitives:
        attach_primitive_contract(primitive)

    # Extract KHR_lights_punctual
    lights = []
    try:
        gltf_lights = gltf.get('extensions', {}).get('KHR_lights_punctual', {})
        if isinstance(gltf_lights, dict):
            gltf_lights = gltf_lights.get('lights', [])
        else:
            gltf_lights = []
        for ni, node in enumerate(gltf.get('nodes', [])):
            lext = node.get('extensions', {}).get('KHR_lights_punctual')
            if lext and 'light' in lext:
                li = lext['light']
                if li < len(gltf_lights):
                    ldef = gltf_lights[li]
                    world_mat = world_mats[ni] if ni < len(world_mats) else np.eye(4, dtype=np.float64)
                    direction = -world_mat[:3, 2].astype(np.float32)
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    position = world_mat[:3, 3].astype(np.float32)
                    spot = ldef.get('spot', {}) if isinstance(ldef.get('spot', {}), dict) else {}
                    lights.append({
                        'type': ldef.get('type', 'directional'),
                        'color': np.array(ldef.get('color', [1, 1, 1])[:3], dtype=np.float32),
                        'intensity': float(ldef.get('intensity', 1.0)),
                        'direction': direction,
                        'position': position,
                        'range': float(ldef.get('range', 0.0) or 0.0),
                        'innerConeAngle': float(spot.get('innerConeAngle', 0.0) or 0.0),
                        'outerConeAngle': float(spot.get('outerConeAngle', 0.7853981633974483) or 0.7853981633974483),
                    })
                    _mat_log.write(f"[LIGHT] {ldef.get('name', '?')}: type={ldef.get('type')} color={ldef.get('color')} intensity={ldef.get('intensity')}\n")
    except Exception as e:
        _mat_log.write(f"[LIGHT] extraction failed: {e}\n")

    _mat_log.write("=== End ===\n")
    _mat_log.close()
    return primitives, all_textures, lights


def _attach_color_management_diagnostics(diagnostics):
    from .gltf.color_management import color_management_diagnostics

    merged = dict(diagnostics or {})
    merged['colorManagement'] = color_management_diagnostics()
    return merged


def load_gltf_scene(path):
    gltf, _buffers = _load_gltf_document(path)
    diagnostics = _attach_color_management_diagnostics(_raise_unsupported_required_extensions(gltf, path))
    primitives, textures, lights = load_glb_model(path)
    return GltfScene(
        primitives=tuple(primitives),
        textures=tuple(textures),
        lights=tuple(lights),
        render_plan=build_render_plan(primitives),
        diagnostics=diagnostics,
    )


def summarize_gltf_scene(primitives, textures, lights, diagnostics=None):
    alpha_modes = {}
    render_passes = {}
    vertex_widths = set()
    scene_min = None
    scene_max = None
    for primitive in primitives:
        material = primitive.get('material_contract') if isinstance(primitive, dict) else None
        alpha_mode = str(material.alpha_mode if isinstance(material, GltfMaterial) else 'OPAQUE').upper()
        render_pass = str(primitive.get('render_pass', '') or '')
        alpha_modes[alpha_mode] = alpha_modes.get(alpha_mode, 0) + 1
        if render_pass:
            render_passes[render_pass] = render_passes.get(render_pass, 0) + 1
        vertices = primitive.get('vertices')
        if isinstance(vertices, np.ndarray) and vertices.ndim == 2:
            vertex_widths.add(int(vertices.shape[1]))
            if vertices.shape[0] and vertices.shape[1] >= 3:
                mn = vertices[:, :3].min(axis=0).astype(np.float32)
                mx = vertices[:, :3].max(axis=0).astype(np.float32)
                scene_min = mn if scene_min is None else np.minimum(scene_min, mn)
                scene_max = mx if scene_max is None else np.maximum(scene_max, mx)

    summary = {
        'primitive_count': len(primitives),
        'texture_count': len(textures),
        'light_count': len(lights),
        'alpha_modes': dict(sorted(alpha_modes.items())),
        'render_passes': dict(sorted(render_passes.items())),
        'vertex_widths': sorted(vertex_widths),
        'scene_bounds': None,
        'diagnostics': diagnostics or {},
    }
    if scene_min is not None and scene_max is not None:
        summary['scene_bounds'] = (scene_min, scene_max)
    return summary


def format_gltf_scene_summary(summary, *, label='glTF model'):
    diagnostics = summary.get('diagnostics') or {}
    alpha_modes = summary.get('alpha_modes') or {}
    render_passes = summary.get('render_passes') or {}
    unsupported_required = diagnostics.get('unsupportedRequired') or []
    unsupported_optional = diagnostics.get('unsupportedOptional') or []
    vertex_widths = summary.get('vertex_widths') or []
    return (
        f"{label}: primitives={summary.get('primitive_count', 0)} "
        f"textures={summary.get('texture_count', 0)} "
        f"lights={summary.get('light_count', 0)} "
        f"vertex_widths={vertex_widths} "
        f"alpha_modes={alpha_modes} "
        f"render_passes={render_passes} "
        f"unsupported_required={unsupported_required} "
        f"unsupported_optional={unsupported_optional}"
    )


def diagnose_gltf_model(path):
    scene = load_gltf_scene(path)
    summary = summarize_gltf_scene(scene.primitives, scene.textures, scene.lights, scene.diagnostics)
    summary['render_plan'] = scene.render_plan
    return summary


# Compatibility exports: keep legacy xr_viewer.gltf_loader symbols while the
# implementations live in the glTF compliance package. These imports stay at the
# end to avoid package initialization cycles with xr_viewer.gltf.__init__.
from .gltf.materials import parse_gltf_material  # noqa: E402
from .gltf.validation import audit_gltf_extensions  # noqa: E402
from .gltf.validation import (  # noqa: E402
    raise_unsupported_required_extensions as _raise_unsupported_required_extensions,
)
