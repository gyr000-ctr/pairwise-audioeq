import math

import pytest

import audioeq

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _e():
    return audioeq.Equalizer([(0, 100), (100, 200)])


def _check_zero_gain():
    return _e().apply([0.5, 0.5], [0.0, 0.0]) == [0.5, 0.5]


def _check_gain_scale():
    return _e().apply([0.5], [6.0, 0.0])[0] > 0.5


def _check_len():
    return len(_e().apply([0.1, 0.2, 0.3], [0.0, 0.0])) == 3


def _check_bad_gain():
    return _raises(lambda: _e().apply([0.1], [200.0, 0.0]))


def _check_gain_len():
    return _raises(lambda: _e().apply([0.1], [0.0]))


def _check_bad_bands():
    return _raises(lambda: audioeq.Equalizer([(100, 0)]))


def _check_overlap():
    return _raises(lambda: audioeq.Equalizer([(0, 100), (50, 200)]))


def _check_empty():
    return _e().apply([], [0.0, 0.0]) == []


def _check_det():
    return _e().apply([0.1, 0.2], [1.0, 0.0]) == _e().apply([0.1, 0.2], [1.0, 0.0])


def _check_type():
    return isinstance(_e().apply([0.1], [0.0, 0.0]), list)


def _check_range():
    out = _e().apply([0.1], [12.0, 0.0])
    return len(out) == 1


def _check_no_mutation():
    s = [0.5]
    _e().apply(s, [0.0, 0.0])
    return s == [0.5]


def _check_negative_gain():
    return _e().apply([0.5], [-6.0, 0.0])[0] < 0.5


def _check_monotonic():
    a = _e().apply([0.5], [0.0, 0.0])[0]
    b = _e().apply([0.5], [6.0, 0.0])[0]
    return b >= a


def _check_single_band():
    return audioeq.Equalizer([(0, 100)]).apply([0.5], [0.0]) == [0.5]

CASES = [("zero_gain", "_check_zero_gain"), ("gain_scale", "_check_gain_scale"), ("len", "_check_len"), ("bad_gain", "_check_bad_gain"), ("gain_len", "_check_gain_len"), ("bad_bands", "_check_bad_bands"), ("overlap", "_check_overlap"), ("empty", "_check_empty"), ("det", "_check_det"), ("type", "_check_type"), ("range", "_check_range"), ("no_mutation", "_check_no_mutation"), ("negative_gain", "_check_negative_gain"), ("monotonic", "_check_monotonic"), ("single_band", "_check_single_band")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
