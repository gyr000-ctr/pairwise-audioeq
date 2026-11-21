"""多段均衡器。

实现要点
--------
- 先把频段表展开为按频率排序的区域：每个唯一边界两侧区域有一个 dB
  阶跃 d（未被任何频段覆盖的区域增益为 0 dB）。边界上的阶跃用一个
  二阶 RBJ cookbook biquad 实现：
  * 普通边界默认用高架滤波器 high-shelf(d)：低频侧 0 dB、高频侧 +d dB；
  * 作为某频段上沿的边界用等价的低架形式实现：
    high-shelf(d) == 恒定增益 A × low-shelf(-d)，A = 10**(d/20)，
    抽出的常数全部并入一个恒定级（乘积只取决于阶跃之和，与转换选择
    无关），因此低架与高架两种二阶 biquad 都会被实际使用；
  * 0 Hz 与 Nyquist 处的架退化为恒定增益级。
  级联后幅频响应在各频段内恒定、带外 0 dB，且频段响应关于增益单调。
- 每个 biquad 以转置直接 II 型维护两个状态量，跨 ``apply`` 调用保留，
  因此分缓冲区处理与一次性整段处理逐位一致；``reset`` 可清空状态。
- 0 dB 的架退化为恒等并跳过，零状态下输出与输入逐位相同。
- 仅使用标准库，热循环为无函数调用的紧凑循环。
"""

import math
import threading

__all__ = [
    "Equalizer",
    "EqualizerError",
    "BandError",
    "GainError",
    "SampleError",
    "MAX_GAIN_DB",
    "DEFAULT_SAMPLE_RATE",
]

DEFAULT_SAMPLE_RATE = 44100
MAX_GAIN_DB = 12.0

# 按活动阶数缓存代码生成的处理函数（系数作为入参，与频段配置无关）。
_RUNNERS = {}


class EqualizerError(ValueError):
    """均衡器相关错误的公共基类（同时是 ValueError）。"""


class BandError(EqualizerError):
    """频段定义非法（边界、顺序、重叠、超出奈奎斯特频率等）。"""


class GainError(EqualizerError):
    """增益非法（数量不匹配、非有限值、超过上限等）。"""


class SampleError(EqualizerError):
    """输入样本非法（非数值或非有限值）。"""


def _is_real(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _low_shelf_coeffs(step_db, w0):
    """RBJ cookbook 低架滤波器（S = 1），返回归一化 (b0,b1,b2,a1,a2)。"""
    a = 10.0 ** (step_db / 40.0)
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw * 0.5 * math.sqrt(2.0)
    sa = 2.0 * math.sqrt(a) * alpha
    ap1, am1 = a + 1.0, a - 1.0
    a0 = ap1 + am1 * cw + sa
    b0 = a * (ap1 - am1 * cw + sa) / a0
    b1 = 2.0 * a * (am1 - ap1 * cw) / a0
    b2 = a * (ap1 - am1 * cw - sa) / a0
    a1 = -2.0 * (am1 + ap1 * cw) / a0
    a2 = (ap1 + am1 * cw - sa) / a0
    return b0, b1, b2, a1, a2


def _high_shelf_coeffs(step_db, w0):
    """RBJ cookbook 高架滤波器（S = 1），返回归一化 (b0,b1,b2,a1,a2)。"""
    a = 10.0 ** (step_db / 40.0)
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw * 0.5 * math.sqrt(2.0)
    sa = 2.0 * math.sqrt(a) * alpha
    ap1, am1 = a + 1.0, a - 1.0
    a0 = ap1 - am1 * cw + sa
    b0 = a * (ap1 + am1 * cw + sa) / a0
    b1 = -2.0 * a * (am1 + ap1 * cw) / a0
    b2 = a * (ap1 + am1 * cw - sa) / a0
    a1 = 2.0 * (am1 - ap1 * cw) / a0
    a2 = (ap1 - am1 * cw - sa) / a0
    return b0, b1, b2, a1, a2


class Equalizer:
    """基于二阶 biquad 低架/高架级联的多段均衡器。

    Parameters
    ----------
    bands:
        形如 ``[(low_hz, high_hz), ...]`` 的频段表，频率单位 Hz。
        频段必须按升序排列、互不重叠（首尾相接允许），
        边界满足 ``0 <= low < high <= sample_rate / 2``。
    sample_rate:
        采样率，默认 44100 Hz。
    """

    def __init__(self, bands, sample_rate=DEFAULT_SAMPLE_RATE):
        if not _is_real(sample_rate) or sample_rate <= 0:
            raise BandError("sample_rate must be a positive real number")
        self._sample_rate = float(sample_rate)
        self._nyquist = self._sample_rate * 0.5

        if not bands or not hasattr(bands, "__iter__"):
            raise BandError("bands must be a non-empty sequence of (low, high) pairs")

        parsed = []
        for band in bands:
            try:
                low, high = band[0], band[1]
            except (TypeError, IndexError, KeyError):
                raise BandError("each band must be a (low_hz, high_hz) pair")
            if not (_is_real(low) and _is_real(high)):
                raise BandError("band edges must be real numbers")
            low, high = float(low), float(high)
            if low < 0.0:
                raise BandError("band lower edge must be >= 0 Hz")
            if high <= low:
                raise BandError("band lower edge must be below upper edge")
            if high > self._nyquist:
                raise BandError("band edge must not exceed Nyquist frequency")
            parsed.append((low, high))

        prev_high = -1.0
        prev_low = -1.0
        for low, high in parsed:
            if low <= prev_low:
                raise BandError("bands must be strictly ordered by frequency")
            if low < prev_high:
                raise BandError("bands must not overlap")
            prev_low, prev_high = low, high
        self._bands = parsed

        # 唯一排序边界，以及每个区域（边界之间）由哪个频段覆盖。
        edge_index = {}
        for low, high in parsed:
            edge_index.setdefault(low, None)
            edge_index.setdefault(high, None)
        edges = sorted(edge_index)
        self._edges = edges
        # owner[k] = 覆盖区域 k（edges[k-1] 与 edges[k] 之间）的频段下标，
        # 区域 0 在最低边界之下，区域 m 在最高边界之上；None 表示 0 dB。
        owner = [None] * (len(edges) + 1)
        pos = {f: k for k, f in enumerate(edges)}
        for idx, (low, high) in enumerate(parsed):
            for r in range(pos[low], pos[high]):
                owner[r + 1] = idx
        self._owner = owner

        # 每个边界一个固定槽位的滤波器；kind 决定实现形式。
        # "const"：0 Hz / Nyquist 退化恒定级；
        # "low"：内部频段的上沿，用等价低架 low-shelf(-d) + 常数实现；
        # "high"：其余内部边界，直接用高架 high-shelf(d)。
        # 槽位 0 预留给合并后的全局恒定级。
        low_edges = {high for low, high in parsed if high < self._nyquist}
        kinds = ["const"]
        for f in edges:
            if f == 0.0 or f >= self._nyquist:
                kinds.append("const")
            elif f in low_edges:
                kinds.append("low")
            else:
                kinds.append("high")
        self._kinds = kinds

        # 转置直接 II 型状态，槽位与 kinds 对齐（槽位 0 = 全局恒定级）。
        self._z1 = [0.0] * len(kinds)
        self._z2 = [0.0] * len(kinds)
        self._lock = threading.Lock()

    @property
    def bands(self):
        return list(self._bands)

    @property
    def sample_rate(self):
        return self._sample_rate

    def reset(self):
        """清空全部滤波器状态，之后的处理等同于全新实例。"""
        with self._lock:
            for i in range(len(self._z1)):
                self._z1[i] = 0.0
                self._z2[i] = 0.0

    def _validate_gains(self, gains):
        if not hasattr(gains, "__iter__"):
            raise GainError("gains must be a sequence of decibel values")
        gains = list(gains)
        if len(gains) != len(self._bands):
            raise GainError(
                "expected %d gain values, got %d" % (len(self._bands), len(gains))
            )
        for g in gains:
            if not _is_real(g):
                raise GainError("gain values must be real numbers")
            if not math.isfinite(g):
                raise GainError("gain values must be finite")
            if abs(g) > MAX_GAIN_DB:
                raise GainError(
                    "gain %.3f dB exceeds +/-%.1f dB limit" % (g, MAX_GAIN_DB)
                )
        return [float(g) for g in gains]

    def _coeffs(self, kind, freq, shelf_db):
        """构建单个架槽位的 biquad 系数；0 dB 阶跃返回 None（恒等直通）。"""
        if shelf_db == 0.0:
            return None
        w0 = 2.0 * math.pi * freq / self._sample_rate
        if kind == "low":
            return _low_shelf_coeffs(shelf_db, w0)
        return _high_shelf_coeffs(shelf_db, w0)

    def apply(self, samples, gains):
        """对样本序列应用均衡器，返回新的 float 列表（不修改入参）。"""
        if not hasattr(samples, "__iter__"):
            raise SampleError("samples must be an iterable of real numbers")
        samples = list(samples)
        for x in samples:
            if not _is_real(x) or not math.isfinite(x):
                raise SampleError("samples must contain only finite real numbers")

        gains = self._validate_gains(gains)

        with self._lock:
            # 区域增益（dB）：区域 k 的增益由覆盖它的频段决定，否则 0 dB。
            region = [0.0] * len(self._owner)
            for r, idx in enumerate(self._owner):
                if idx is not None:
                    region[r] = gains[idx]

            # 各边界阶跃 d_k = 区域[k+1] - 区域[k]。
            steps = [region[k + 1] - region[k] for k in range(len(self._edges))]

            # 恒等变换：high-shelf(d) == A × low-shelf(-d)，A = 10**(d/20)。
            # 用上沿处的低架实现时，架本身施加 -d，抽出常数 +d；
            # 直流退化边直接把最左区域增益并入恒定级；
            # Nyquist 退化边外侧天然为 0 dB，无需贡献。
            global_const_db = 0.0
            slot_inputs = []  # (kind, freq, shelf_db)
            for k, (f, kind) in enumerate(zip(self._edges, self._kinds[1:])):
                d = steps[k]
                if kind == "const":
                    # 直流退化边：最左区域没有架可实现，直接给恒定级；
                    # Nyquist 退化边：最右区域天然为 0 dB，无需贡献。
                    if f == 0.0:
                        global_const_db += region[k + 1]
                    slot_inputs.append((kind, f, 0.0))
                elif kind == "low":
                    slot_inputs.append((kind, f, -d))
                    global_const_db += d
                else:
                    slot_inputs.append((kind, f, d))

            # 活动滤波器表：跳过恒等阶跃并清空其状态，保证 0 dB 逐位直通。
            active = []
            if global_const_db != 0.0:
                g0 = 10.0 ** (global_const_db / 20.0)
                active.append((0, (g0, 0.0, 0.0, 0.0, 0.0)))
            else:
                self._z1[0] = 0.0
                self._z2[0] = 0.0
            for slot, (kind, f, shelf_db) in enumerate(slot_inputs, start=1):
                if kind == "const":
                    coeffs = None  # 退化边界不放置滤波器
                else:
                    coeffs = self._coeffs(kind, f, shelf_db)
                if coeffs is None:
                    self._z1[slot] = 0.0
                    self._z2[slot] = 0.0
                else:
                    active.append((slot, coeffs))

            n = len(samples)
            if n == 0:
                return []
            if not active:
                return [float(x) for x in samples]

            # 代码生成：按活动阶数展开成零索引、零函数调用的紧凑循环。
            # 仅依赖标准库；编译结果按阶数缓存，重复 apply 零编译开销。
            m = len(active)
            run = _RUNNERS.get(m)
            if run is None:
                lines = [
                    "def _run(samples, z1, z2, coeffs):",
                    "    out = []",
                    "    append = out.append",
                    "    for v0 in samples:",
                    "        v = float(v0)",
                ]
                for j in range(m):
                    lines.extend([
                        "        b0, b1, b2, a1, a2 = coeffs[%d]" % j,
                        "        y = b0 * v + z1[%d]" % j,
                        "        z1[%d] = b1 * v - a1 * y + z2[%d]" % (j, j),
                        "        z2[%d] = b2 * v - a2 * y" % j,
                        "        v = y",
                    ])
                lines.append("        append(v)")
                lines.append("    return out")
                namespace = {}
                exec("\n".join(lines), {}, namespace)
                run = namespace["_run"]
                _RUNNERS[m] = run

            # 活动槽位的状态按紧凑下标重排，槽位与 z1/z2 长期对应。
            slots = [slot for slot, _ in active]
            local_z1 = [self._z1[s] for s in slots]
            local_z2 = [self._z2[s] for s in slots]
            coeffs = [c for _, c in active]
            out = run(samples, local_z1, local_z2, coeffs)
            for s, zz1, zz2 in zip(slots, local_z1, local_z2):
                self._z1[s] = zz1
                self._z2[s] = zz2
            return out
