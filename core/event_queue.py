# core/event_queue.py
from queue import Queue  # Can be swapped to PriorityQueue later if needed

class EventQueue:
    """Central in-memory bus for Market/Signal/Order/Fill events."""
    def __init__(self):
        self._q = Queue()

    def put(self, event):
        self._q.put(event)

    def get(self):
        return self._q.get(block=False)

    def empty(self):
        return self._q.empty()
