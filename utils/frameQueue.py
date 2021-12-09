from typing import List
from vo_pipeline.frameState import FrameState


class FrameQueue:
    """Ring buffer based on a python list """
    def __init__(self, size: int):
        assert size > 0
        self.queue: List[FrameState] = [None] * size
        self.head = -1
        self.tail = 0
        self.size = size
        self._is_new = True
        self._i = -1
        self._idx = -1

    def add(self, frame_state: FrameState):
        self.head = (self.head + 1) % self.size
        self.tail = (
            self.tail + 1
        ) % self.size if self.head == self.tail and not self._is_new else self.tail
        self._is_new = False
        self.queue[self.head] = frame_state

    def get(self, idx: int):
        assert idx >= 0
        return self.queue[self.head - idx]

    def get_head(self) -> FrameState:
        return self.queue[self.head]

    def __iter__(self):
        return self

    def __next__(self):
        self._i += 1
        self._idx = (self.tail + self._i) % self.size
        if (self._idx - 1) % self.size == self.head:
            self._i = -1
            self._idx = -1
            raise StopIteration()
        return self.queue[self._idx]