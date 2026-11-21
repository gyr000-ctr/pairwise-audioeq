"""audioeq 核心 API（初始未实现）。"""


class Equalizer:
    def __init__(self, bands):
        """按频段构造"""
        raise NotImplementedError

    def apply(self, samples, gains):
        """应用增益"""
        raise NotImplementedError
