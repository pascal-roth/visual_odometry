from typing import List
from vo_pipeline.frameState import FrameState


class FrameQueue:
    def __init__(self, size: int):
        assert size > 0
        self.queue: List[FrameState] = [None] * size
        self.head = -1
        self.size = size

    def add(self, frame_state: FrameState):
        self.head = (self.head + 1) % self.size
        self.queue[self.head] = frame_state

    def get(self, idx: int):
        assert idx >= 0
        return self.queue[self.head - idx]
    
    def get_head(self) -> FrameState:
        return self.queue[self.head]