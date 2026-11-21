"""多段均衡器补充验收测试（隐藏边界：性能/并发/非有限/频段边界/增益上限）。"""

import math
import threading
import time

import pytest

import audioeq
from audioeq import BandError, Equalizer, GainError, SampleError

SR = 44100


def _sine(freq, n, sr=SR, amp=0.5):
    return [amp * math.sin(2.0 * math.pi * freq * k / sr) for k in range(n)]


def _rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


# ---------- 0 dB 原样 ----------

def test_eq_zero_db_is_exact_identity():
    eq = Equalizer([(0, 200), (200, 1000)])
    src = [0.3 * math.sin(k * 0.1) for k in range(512)]
    out = eq.apply(src, [0.0, 0.0])
    assert out == src
    assert all(type(v) is float for v in out)


def test_eq_empty_input():
    eq = Equalizer([(0, 200)])
    assert eq.apply([], [0.0]) == []


def test_eq_does_not_mutate_input():
    eq = Equalizer([(0, 200)])
    src = [0.5, -0.25, 0.125]
    snapshot = list(src)
    eq.apply(src, [6.0])
    assert src == snapshot


# ---------- 二阶 biquad 低架/高架的频率形状 ----------

def test_eq_low_shelf_dc_and_high_freq_gains():
    # 单条从 0 起的频段：直流处应得全部增益，远高于边频处趋近 0 dB。
    eq = Equalizer([(0, 500)])
    g_db = 12.0
    target = 10.0 ** (g_db / 20.0)
    dc = eq.apply([1.0] * 6000, [g_db])
    assert dc[-1] == pytest.approx(target, rel=2e-3)
    # 18 kHz 距 500 Hz 边频 5 个多倍频程，低架尾部在此约 0.4 dB
    # （RBJ S=1 架以 6 dB/倍频程回落的固有特性）。
    ref = _sine(18000, 8000)
    hi = eq.apply(_sine(18000, 8000), [g_db])
    assert _rms(hi) / _rms(ref) == pytest.approx(1.0, abs=0.06)


def test_eq_high_shelf_gain_shape():
    # 高架型边界：低频几乎不动，高频被提升。
    eq = Equalizer([(1000, SR / 2)])
    g_db = 9.0
    target = 10.0 ** (g_db / 20.0)
    hi = eq.apply(_sine(12000, 8000), [g_db])
    assert _rms(hi) / _rms(_sine(12000, 8000)) == pytest.approx(target, rel=2e-2)
    lo = eq.apply(_sine(60, 8000), [g_db])
    assert _rms(lo) / _rms(_sine(60, 8000)) == pytest.approx(1.0, abs=0.03)


def test_eq_band_response_is_monotonic_in_gain():
    eq = Equalizer([(0, 300)])
    prev = -math.inf
    for g in (-12.0, -6.0, 0.0, 6.0, 12.0):
        out = eq.apply(_sine(120, 6000), [g])
        level = _rms(out)
        assert level >= prev - 1e-12
        prev = level


def test_eq_mid_band_is_band_limited():
    # 中间频段：通带内明显提升，远低于和远高于频段处均趋近 0 dB
    # （S=1 架过渡带约一个倍频程，测量点取在边频 4 倍频程之外）。
    eq = Equalizer([(800, 1600)])
    g = 10.0
    inside = _rms(eq.apply(_sine(1150, 8000), [g])) / _rms(_sine(1150, 8000))
    below = _rms(eq.apply(_sine(80, 8000), [g])) / _rms(_sine(80, 8000))
    above = _rms(eq.apply(_sine(10000, 8000), [g])) / _rms(_sine(10000, 8000))
    assert inside > 1.7
    assert below == pytest.approx(1.0, abs=0.03)
    assert above == pytest.approx(1.0, abs=0.03)


# ---------- 滤波状态：跨缓冲区一致、可重置、无爆音 ----------

def test_eq_chunked_processing_matches_whole_buffer():
    eq = Equalizer([(0, 300), (300, 2000), (2000, SR / 2)])
    src = _sine(440, 3001, amp=0.7)
    gains = [6.0, -4.0, 3.0]
    whole = Equalizer([(0, 300), (300, 2000), (2000, SR / 2)]).apply(src, gains)

    eq_chunks = Equalizer([(0, 300), (300, 2000), (2000, SR / 2)])
    glued = []
    for lo in range(0, len(src), 137):
        glued.extend(eq_chunks.apply(src[lo:lo + 137], gains))
    assert glued == whole  # 同系数同状态，逐位一致


def test_eq_reset_restores_initial_state():
    bands = [(0, 400), (400, 2000)]
    gains = [8.0, -5.0]
    src = _sine(300, 600)
    a = Equalizer(bands).apply(src, gains)
    b = Equalizer(bands)
    b.apply(_sine(1000, 900), gains)
    b.reset()
    assert b.apply(src, gains) == a


def test_eq_no_pops_bounded_sample_jumps():
    eq = Equalizer([(0, 500), (500, 4000)])
    src = [0.0] * 500 + [0.9] * 500 + [-0.9] * 500 + [0.0] * 500
    out = eq.apply(src, [12.0, -12.0])
    max_in = max(abs(b - a) for a, b in zip(src, src[1:]))
    max_out = max(abs(b - a) for a, b in zip(out, out[1:]))
    # 稳定的双二阶级联：输出样本差不应超过输入跳变的一个有界倍数。
    assert max_out <= 6.0 * max_in + 1e-12
    # 输出全程有限；幅值上界取各架绝对增益之积（含 50% 瞬态余量）。
    peak_bound = 1.5 * max(1.0, max(abs(v) for v in src)) * 10.0 ** (24.0 / 20.0)
    assert all(math.isfinite(v) and abs(v) < peak_bound for v in out)


def test_eq_impulse_response_decays():
    eq = Equalizer([(0, 500), (500, 4000), (4000, SR / 2)])
    src = [1.0] + [0.0] * 2000
    out = eq.apply(src, [10.0, -10.0, 6.0])
    assert math.isfinite(out[-1])
    assert abs(out[-1]) < 1e-6  # 稳定滤波器冲激响应衰减


# ---------- 可区分的错误类型 ----------

def test_eq_error_classes_are_distinct_value_errors():
    for exc in (BandError, GainError, SampleError):
        assert issubclass(exc, ValueError)
    assert len({BandError, GainError, SampleError}) == 3


def test_eq_band_validation():
    with pytest.raises(BandError):
        Equalizer([(100, 0)])
    with pytest.raises(BandError):
        Equalizer([(0, 100), (50, 200)])
    with pytest.raises(BandError):
        Equalizer([(200, 100)])
    with pytest.raises(BandError):
        Equalizer([(-1.0, 100.0)])
    with pytest.raises(BandError):
        Equalizer([(0.0, SR)])  # 超过 Nyquist
    with pytest.raises(BandError):
        Equalizer([])
    with pytest.raises(BandError):
        Equalizer([(100, 100)])


def test_eq_band_boundaries_dc_and_nyquist_accepted():
    eq = Equalizer([(0.0, 500.0), (500.0, SR / 2)])
    out = eq.apply([0.3, -0.3, 0.1], [6.0, -6.0])
    assert len(out) == 3 and all(math.isfinite(v) for v in out)


def test_eq_gain_validation_distinguishable():
    eq = Equalizer([(0, 200), (200, 1000)])
    with pytest.raises(GainError):
        eq.apply([0.1], [200.0, 0.0])
    with pytest.raises(GainError):
        eq.apply([0.1], [0.0])  # 数量不匹配
    with pytest.raises(GainError):
        eq.apply([0.1], [float("nan"), 0.0])
    with pytest.raises(GainError):
        eq.apply([0.1], [float("inf"), 0.0])
    # 增益上限本身可接受（闭区间）。
    assert len(eq.apply([0.1], [12.0, -12.0])) == 1
    with pytest.raises(GainError):
        eq.apply([0.1], [12.5, 0.0])
    with pytest.raises(GainError):
        eq.apply([0.1], [0.0, -12.5])


def test_eq_nonfinite_samples_rejected():
    eq = Equalizer([(0, 200)])
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(SampleError):
            eq.apply([0.1, bad, 0.2], [0.0])
    # 与增益错误可区分。
    try:
        eq.apply([0.1, float("nan")], [0.0])
    except SampleError:
        pass
    else:  # pragma: no cover
        raise AssertionError("non-finite sample must raise SampleError")


# ---------- 性能 ----------

def test_eq_performance_1m_under_300ms():
    n = 1_000_000
    eq = Equalizer([(0, 300), (300, 2000), (2000, SR / 2)])
    src = [0.5 * math.sin(k * 0.05) for k in range(8192)]
    gains = [6.0, -3.0, 4.0]
    eq.apply(src, gains)  # 预热
    big = [0.5 * ((k * 1103515245 + 12345) & 0x7fff) / 0x4000 - 1.0
           for k in range(n)]
    t0 = time.perf_counter()
    out = eq.apply(big, gains)
    dt = time.perf_counter() - t0
    assert len(out) == n
    assert dt < 0.300, dt


def test_eq_performance_100k():
    n = 100_000
    eq = Equalizer([(0, 400), (400, SR / 2)])
    big = [0.3 * math.sin(k * 0.07) for k in range(n)]
    t0 = time.perf_counter()
    out = eq.apply(big, [5.0, -5.0])
    dt = time.perf_counter() - t0
    assert len(out) == n
    assert dt < 0.100, dt


# ---------- 并发与确定性 ----------

def test_eq_concurrent_equalizers_are_independent():
    bands = [(0, 400), (400, 3000)]
    gains = [6.0, -4.0]
    src = _sine(440, 2048)
    expected = Equalizer(bands).apply(src, gains)
    results = [None] * 8

    def worker(i):
        eq = Equalizer(bands)
        results[i] = eq.apply(src, gains)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r == expected for r in results)


def test_eq_shared_equalizer_thread_safety():
    eq = Equalizer([(0, 500), (500, SR / 2)])
    src = _sine(220, 1024)
    errors = []

    def worker():
        try:
            for _ in range(10):
                out = eq.apply(src, [3.0, -3.0])
                assert len(out) == len(src)
                assert all(math.isfinite(v) for v in out)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_eq_deterministic():
    a = Equalizer([(0, 300), (300, 2000)]).apply(_sine(440, 512), [4.0, -2.0])
    b = Equalizer([(0, 300), (300, 2000)]).apply(_sine(440, 512), [4.0, -2.0])
    assert a == b


def test_eq_gap_bands_region_outside_is_unity():
    # 频段不连续时，未覆盖区域应为 0 dB（高架/低架阶跃法的天然性质）。
    eq = Equalizer([(2000, 4000)])
    src = _sine(100, 6000)
    out = eq.apply(src, [10.0])
    assert _rms(out) / _rms(src) == pytest.approx(1.0, abs=0.03)
    # 间隙区域（800 与 4000 两个频段之间）同样保持 0 dB；
    # 取间隙中心并对 S=1 架的过渡带尾部留容差。
    eq2 = Equalizer([(200, 800), (4000, 8000)])
    gap = _sine(2000, 8000)
    out2 = eq2.apply(gap, [8.0, 8.0])
    assert _rms(out2) / _rms(gap) == pytest.approx(1.0, abs=0.10)
