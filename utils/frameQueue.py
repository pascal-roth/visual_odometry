from typing import List, Deque
from vo_pipeline.frameState import FrameState
from collections import deque


class FrameQueue:
    """Ring buffer based on a python list """
    def __init__(self, size: int):
        assert size > 0
        self.queue: List[FrameState] = []
        self.size = size
        self._i = -1

    @property
    def length(self) -> int:
        return len(self.queue)


    def add(self, frame_state: FrameState):
        self.queue.append(frame_state)
        if len(self.queue) > self.size:
            self.queue.pop(0)

    def get(self, idx: int):
        assert idx >= 0
        return self.queue[len(self.queue) - idx - 1]

    def get_head(self) -> FrameState:
        return self.get(0)

    def __iter__(self):
        return self

    def __next__(self):
        self._i += 1
        if self._i < len(self.queue):
            return self.queue[self._i]
        else:
            self._i = -1
            raise StopIteration()