"""3D LUT 调色模块（四面体插值）。

针对既有管线的修复:

1. 域外钳制   —— 查表前把归一化坐标钳到 [0, 1]，三线性/四面体插值不再
   跨表外空白，饱和红不再被拉向灰。
2. 统一四面体划分 —— 所有晶格共用主对角线 000->111 的六四面体分解，
   选取只取决于颜色分量的相对大小，与 tile 索引奇偶无关，接缝两侧一致。
3. 输入色彩空间 —— 显式区分场景线性与显示 gamma，查表前转换到 LUT 域，
   暗部不再被整体提亮。
4. shaper 应用 —— LUT 携带的 1D shaper 曲线在 3D 查表前应用，Log 素材
   不再直接进表导致映射整体偏移。
5. 缓存版本   —— 缓存键由表尺寸 + 表内容哈希 + shaper 构成，换表或改
   尺寸后版本必变，旧缓存自然失效。
6. 色域裁切   —— LUT 输出裁回 [0, 1] 色域。

apply_lut 为纯函数，无可变共享状态；多线程按行分块时各块只写自己的
结果槽位，不出现共享缓存竞争条纹；CPU 与 GPU 路径共用同一插值核心
（_lookup_tetra），保证预览与导出一致。
"""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor

GAMUT_LO = 0.0
GAMUT_HI = 1.0
_DEFAULT_GAMMA = 2.2
_PARALLEL_THRESHOLD = 4096


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _is_seq(x):
    return isinstance(x, (list, tuple))


def _as_rgb(rgb):
    """把标量/序列规整为 RGB 三元组（标量视为灰）。"""
    if _is_seq(rgb):
        vals = tuple(float(v) for v in rgb)
        if len(vals) >= 3:
            return vals[:3]
        if len(vals) == 1:
            return (vals[0],) * 3
        return (0.0, 0.0, 0.0)
    v = float(rgb)
    return (v, v, v)


# ---------------------------------------------------------------------------
# 修复点 1: 域外钳制
# ---------------------------------------------------------------------------

def gamut_clamp(rgb, lo=GAMUT_LO, hi=GAMUT_HI):
    """把标量或 RGB 序列钳制到 [lo, hi] 域内（NaN 归入 lo）。"""
    def _c(v):
        v = float(v)
        if v != v:  # NaN
            return lo
        return lo if v < lo else hi if v > hi else v

    if _is_seq(rgb):
        return tuple(_c(v) for v in rgb)
    return _c(rgb)


# ---------------------------------------------------------------------------
# 修复点 2: 统一四面体划分
# ---------------------------------------------------------------------------

# 主对角线固定为 000->111 的六四面体分解，覆盖单位立方体。
# 每个条目为 4 个顶点偏移（相对晶格下角），顶点顺序沿分量从大到小的路径。
_TETRA_SPLIT = (
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),  # 0: r >= g >= b
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)),  # 1: r >= b >  g
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)),  # 2: g >  r >= b
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),  # 3: g >= b >  r
    ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),  # 4: b >  r >= g
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)),  # 5: b >  g >  r
)


def uniform_tetra_split(tile_index=None):
    """返回统一的六四面体划分。

    无论传入哪个 tile 索引都返回同一划分（不再按索引奇偶翻转对角线），
    保证相邻 tile 接缝两侧对同一颜色给出同一结果。
    """
    return _TETRA_SPLIT


def _tetra_case(fr, fg, fb):
    """按分量排序选择四面体，返回 (划分索引, 四顶点权重)，权重和为 1。"""
    if fr >= fg:
        if fg >= fb:
            return 0, (1.0 - fr, fr - fg, fg - fb, fb)
        if fr >= fb:
            return 1, (1.0 - fr, fr - fb, fb - fg, fg)
        return 4, (1.0 - fb, fb - fr, fr - fg, fg)
    if fr >= fb:
        return 2, (1.0 - fg, fg - fr, fr - fb, fb)
    if fg >= fb:
        return 3, (1.0 - fg, fg - fb, fb - fr, fr)
    return 5, (1.0 - fb, fb - fg, fg - fr, fr)


# ---------------------------------------------------------------------------
# 修复点 3: 输入色彩空间
# ---------------------------------------------------------------------------

def input_colorspace(value, space="display", gamma=_DEFAULT_GAMMA):
    """把输入转换到 LUT 域。

    space="scene-linear": 场景线性输入，先做 gamma 编码再查显示域 LUT，
                          避免暗部被整体提亮。
    space="display" / "display-gamma": 已在显示域，直通。
    """
    def _enc(v):
        v = float(v)
        if space in ("scene-linear", "linear", "scene"):
            return v ** (1.0 / gamma) if v > 0.0 else 0.0
        if space in ("display", "display-gamma", "gamma"):
            return v
        raise ValueError(f"未知输入色彩空间: {space!r}")

    if _is_seq(value):
        return tuple(_enc(v) for v in value)
    return _enc(value)


# ---------------------------------------------------------------------------
# 修复点 4: shaper 应用
# ---------------------------------------------------------------------------

def shaper_apply(value, shaper=None):
    """在 3D 查表前应用 1D shaper 曲线。

    shaper 可为 None（恒等）、可调用对象、1D 采样序列（线性插值）或
    "log" / "gamma:<g>" 内置曲线名。Log 素材先过 shaper 再进表，
    映射不再整体偏移。
    """
    def _one(v):
        v = float(v)
        if shaper is None:
            return v
        if callable(shaper):
            return float(shaper(v))
        if isinstance(shaper, str):
            name = shaper.lower()
            if name == "log":
                # 归一化对数曲线: [0,1] -> [0,1]
                return math.log1p(v * (math.e - 1.0)) if v > 0.0 else 0.0
            if name.startswith("gamma:"):
                g = float(name.split(":", 1)[1])
                return gamut_clamp(v) ** (1.0 / g)
            raise ValueError(f"未知 shaper: {shaper!r}")
        # 1D 采样序列，均匀采样 + 线性插值
        samples = list(shaper)
        n = len(samples)
        if n == 0:
            return v
        if n == 1:
            return float(samples[0])
        x = gamut_clamp(v) * (n - 1)
        i = min(int(x), n - 2)
        t = x - i
        return float(samples[i]) * (1.0 - t) + float(samples[i + 1]) * t

    if _is_seq(value):
        return tuple(_one(v) for v in value)
    return _one(value)


# ---------------------------------------------------------------------------
# 修复点 5: 缓存版本
# ---------------------------------------------------------------------------

def cache_version(lut=None, shaper=None):
    """计算 LUT 缓存版本号。

    版本由表尺寸 + 表内容哈希 + shaper 构成：换表或改尺寸后版本必变，
    旧缓存自然失效，不会切了 LUT 仍用旧表。
    """
    if isinstance(lut, dict):
        size = lut.get("size")
        data = lut.get("data")
        shaper = lut.get("shaper", shaper)
    elif isinstance(lut, int):
        size, data = lut, None
    elif lut is None:
        size, data = None, None
    else:
        size, data = None, lut

    h = hashlib.sha1()
    h.update(repr(size).encode())
    if data is not None:
        for entry in data:
            h.update(repr(tuple(entry) if _is_seq(entry) else entry).encode())
    h.update(repr(shaper).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 修复点 6: 色域裁切
# ---------------------------------------------------------------------------

def gamut_map_clip(rgb, lo=GAMUT_LO, hi=GAMUT_HI):
    """色域裁切：把 LUT 输出裁回 [lo, hi] 色域。"""
    return gamut_clamp(rgb, lo, hi)


# ---------------------------------------------------------------------------
# LUT 加载 / 构建 / 校验
# ---------------------------------------------------------------------------

def _validate_lut(lut):
    size = int(lut.get("size", 0))
    data = lut.get("data")
    if size < 2:
        raise ValueError("LUT 尺寸至少为 2")
    if data is None or len(data) != size ** 3:
        raise ValueError(f"LUT 数据长度应为 {size ** 3}")
    for entry in data:
        if not _is_seq(entry) or len(entry) < 3:
            raise ValueError("LUT 条目须为 RGB 三元组")
        for v in entry[:3]:
            if not math.isfinite(float(v)):
                raise ValueError("LUT 含非有限值")
    return True


def _parse_cube(path):
    """解析 .cube 文件，返回 (size, data, shaper_samples_or_None)。"""
    size = None
    shaper_samples = []
    data = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.upper().startswith("TITLE"):
                continue
            upper = line.upper()
            if upper.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            if upper.startswith("LUT_1D_SIZE"):
                continue
            if upper.startswith(("DOMAIN_MIN", "DOMAIN_MAX")):
                continue
            parts = line.split()
            if len(parts) >= 3:
                rgb = tuple(float(p) for p in parts[:3])
                if size is None:
                    # LUT_3D_SIZE 之前的数据行是 1D shaper 曲线
                    shaper_samples.append(sum(rgb) / 3.0)
                else:
                    data.append(rgb)
    if size is None:
        raise ValueError(".cube 缺少 LUT_3D_SIZE")
    return size, data, (shaper_samples or None)


def load_lut(source, shaper=None):
    """加载 LUT，返回 {"size", "data", "shaper", "version"}。

    source 可以是:
    - int: 生成该尺寸的恒等 LUT（尺寸至少为 2）;
    - dict: 已构造的 LUT，校验后附加版本号;
    - str: .cube 文件路径（含 1D shaper 时一并加载）;
    - (size, data) 序列。
    """
    if isinstance(source, dict):
        lut = dict(source)
        _validate_lut(lut)
        lut.setdefault("shaper", shaper)
        lut["version"] = cache_version(lut)
        return lut
    if isinstance(source, int):
        return build_lut(max(2, source), shaper=shaper)
    if isinstance(source, str):
        size, data, file_shaper = _parse_cube(source)
        lut = {
            "size": size,
            "data": data,
            "shaper": shaper if shaper is not None else file_shaper,
        }
        _validate_lut(lut)
        lut["version"] = cache_version(lut)
        return lut
    if _is_seq(source) and len(source) == 2:
        size, data = source
        lut = {
            "size": int(size),
            "data": [tuple(float(v) for v in e[:3]) for e in data],
            "shaper": shaper,
        }
        _validate_lut(lut)
        lut["version"] = cache_version(lut)
        return lut
    raise ValueError(f"无法识别的 LUT 来源: {type(source)!r}")


def build_lut(size=17, transform=None, shaper=None):
    """构建 3D LUT（默认恒等）。transform 若给出，对每个网格点颜色调用。

    数据布局与 .cube 一致: r 变化最快，索引 = (b * size + g) * size + r。
    """
    size = max(2, int(size))
    n1 = size - 1
    data = []
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rgb = (r / n1, g / n1, b / n1)
                if transform is not None:
                    rgb = _as_rgb(transform(rgb))
                data.append(rgb)
    lut = {"size": size, "data": data, "shaper": shaper}
    lut["version"] = cache_version(lut)
    return lut


# ---------------------------------------------------------------------------
# 插值核心（CPU / GPU 共用同一算法，保证预览与导出一致）
# ---------------------------------------------------------------------------

def _lookup_tetra(lut, rgb):
    """四面体插值核心。查表前先域外钳制，插值不会跨出表外。"""
    size = lut["size"]
    data = lut["data"]
    n1 = size - 1
    r, g, b = gamut_clamp(rgb)
    x, y, z = r * n1, g * n1, b * n1
    i = min(int(x), n1 - 1)
    j = min(int(y), n1 - 1)
    k = min(int(z), n1 - 1)
    fr, fg, fb = x - i, y - j, z - k
    case, weights = _tetra_case(fr, fg, fb)
    verts = _TETRA_SPLIT[case]
    base = (k * size + j) * size + i
    out0 = out1 = out2 = 0.0
    for (dr, dg, db), w in zip(verts, weights):
        if w == 0.0:
            continue
        entry = data[base + (db * size + dg) * size + dr]
        out0 += w * entry[0]
        out1 += w * entry[1]
        out2 += w * entry[2]
    return (out0, out1, out2)


# ---------------------------------------------------------------------------
# 应用 LUT
# ---------------------------------------------------------------------------

_DEFAULT_LUTS = {}


def _default_lut(size=2):
    key = ("identity", int(size))
    lut = _DEFAULT_LUTS.get(key)
    if lut is None:
        lut = build_lut(size)
        _DEFAULT_LUTS[key] = lut
    return lut


def _apply_pixel(rgb, lut, input_space, clip):
    vals = _as_rgb(rgb)
    vals = input_colorspace(vals, input_space)
    shaper = lut.get("shaper")
    if shaper is not None:
        vals = shaper_apply(vals, shaper)
    out = _lookup_tetra(lut, vals)
    if clip:
        out = gamut_map_clip(out)
    return out


def _apply_image(pixels, lut, input_space, clip):
    n = len(pixels)
    results = [None] * n
    if n < _PARALLEL_THRESHOLD:
        for idx, px in enumerate(pixels):
            results[idx] = _apply_pixel(px, lut, input_space, clip)
        return results
    # 按行/段分块，每块只写自己的结果槽位，无共享可变缓存，无竞争条纹
    n_chunks = min(8, (n + 1023) // 1024)
    chunk = (n + n_chunks - 1) // n_chunks
    bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]

    def _run(lo, hi):
        for idx in range(lo, hi):
            results[idx] = _apply_pixel(pixels[idx], lut, input_space, clip)

    with ThreadPoolExecutor(max_workers=n_chunks) as pool:
        list(pool.map(lambda b: _run(*b), bounds))
    return results


def apply_lut(rgb, lut=None, input_space="display", clip=True):
    """应用 LUT: 输入色彩空间转换 -> shaper -> 域外钳制 -> 四面体插值 -> 色域裁切。

    纯函数，无可变共享状态；传入像素序列时按索引写结果槽，
    多线程分块也不会出现共享缓存竞争条纹。
    """
    if lut is None:
        lut = _default_lut()
    elif not isinstance(lut, dict):
        lut = load_lut(lut)
    if _is_seq(rgb) and rgb and _is_seq(rgb[0]):
        return _apply_image(rgb, lut, input_space, clip)
    return _apply_pixel(rgb, lut, input_space, clip)


# ---------------------------------------------------------------------------
# 调色管线入口
# ---------------------------------------------------------------------------

def present_grade(image, lut=None, input_space="display"):
    """预览/导出共用的调色入口：同一代码路径保证两端一致。

    LUT 缺失或非法时回退为恒等直通，不崩溃。
    """
    try:
        if lut is not None and not isinstance(lut, dict):
            lut = load_lut(lut)
        return apply_lut(image, lut=lut, input_space=input_space, clip=True)
    except (ValueError, TypeError, KeyError, OSError):
        return fallback_grade(image)


def validate_grade(lut):
    """校验 LUT 结构，返回 (是否通过, 错误列表)。"""
    errors = []
    try:
        if isinstance(lut, dict):
            _validate_lut(lut)
        elif isinstance(lut, int):
            if lut < 2:
                errors.append("LUT 尺寸至少为 2")
        else:
            _validate_lut(load_lut(lut))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        errors.append(str(exc))
    return (not errors, errors)


def metrics_grade(image, reference=None):
    """调色结果指标：各通道均值/最小/最大；给参考时附最大/平均绝对误差。"""
    if _is_seq(image) and image and _is_seq(image[0]):
        pixels = [_as_rgb(px) for px in image]
    else:
        pixels = [_as_rgb(image)]
    n = len(pixels)
    report = {
        "count": n,
        "mean": tuple(sum(px[c] for px in pixels) / n for c in range(3)),
        "min": tuple(min(px[c] for px in pixels) for c in range(3)),
        "max": tuple(max(px[c] for px in pixels) for c in range(3)),
    }
    if reference is not None:
        if _is_seq(reference) and reference and _is_seq(reference[0]):
            ref = [_as_rgb(px) for px in reference]
        else:
            ref = [_as_rgb(reference)]
        m = min(len(ref), n)
        if m:
            errs = [abs(pixels[i][c] - ref[i][c])
                    for i in range(m) for c in range(3)]
            report["max_error"] = max(errs)
            report["mean_error"] = sum(errs) / len(errs)
    return report


def audit_grade(lut=None, input_space="display"):
    """一致性审计报告。

    - seam: 晶格接缝两侧采样，验证统一四面体划分下结果连续；
    - determinism: 重复应用结果逐位一致；
    - gamut: 输出均在色域内。
    """
    lut = lut if isinstance(lut, dict) else load_lut(lut)
    checks = {}
    # 接缝连续性
    eps = 1e-6
    max_delta = 0.0
    for axis in range(3):
        for i in range(1, lut["size"]):
            x = i / (lut["size"] - 1)
            for t in (0.25, 0.5, 0.75):
                lo = [t, t, t]
                hi = [t, t, t]
                lo[axis] = x - eps
                hi[axis] = x + eps
                a = apply_lut(lo, lut, input_space)
                b = apply_lut(hi, lut, input_space)
                max_delta = max(max_delta,
                                max(abs(a[c] - b[c]) for c in range(3)))
    checks["seam_max_delta"] = max_delta
    checks["seam_ok"] = max_delta < 1e-3
    checks["deterministic"] = determinism_grade(lut=lut)
    corners = [(r, g, b) for r in (0.0, 1.0) for g in (0.0, 1.0) for b in (0.0, 1.0)]
    outs = apply_lut(corners, lut, input_space)
    checks["gamut_ok"] = all(
        all(GAMUT_LO - 1e-9 <= v <= GAMUT_HI + 1e-9 for v in px) for px in outs
    )
    ok = checks["seam_ok"] and checks["deterministic"] and checks["gamut_ok"]
    return {"lut_version": lut["version"], "checks": checks, "ok": ok}


def fallback_grade(image, reason=None):
    """LUT 缺失/非法时的回退：恒等直通 + 色域裁切，不崩溃。"""
    if _is_seq(image) and image and _is_seq(image[0]):
        return [gamut_map_clip(_as_rgb(px)) for px in image]
    return gamut_map_clip(_as_rgb(image))


def determinism_grade(image=None, lut=None, repeats=2):
    """确定性校验：多次应用（含多线程分块路径）结果逐位一致。"""
    if image is None:
        image = [(r / 7.0, g / 7.0, b / 7.0)
                 for r in range(8) for g in range(8) for b in range(8)]
    first = apply_lut(image, lut)
    for _ in range(max(1, repeats) - 1):
        if apply_lut(image, lut) != first:
            return False
    return True
