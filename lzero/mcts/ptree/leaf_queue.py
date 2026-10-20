from dataclasses import dataclass


@dataclass
class LeafTask:
    task_id: int
    node: object
    env_snapshot: object
    search_path: tuple

class LeafQueue:
    def __init__(self, max_size: int = 16):
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self.max_size = max_size
        self._queue = []
        self._node_ids = set()

    def add(self, task: LeafTask) -> bool:
        if len(self._queue) >= self.max_size:
            return False

        node_id = id(task.node)

        if node_id in self._node_ids:
            return False

        self._queue.append(task)
        self._node_ids.add(node_id)

        return True

    def pop_batch(self, size: int) -> list[LeafTask]:
        if size <= 0:
            raise ValueError("size must be positive")

        batch = self._queue[:size]
        self._queue = self._queue[size:]

        for task in batch:
            self._node_ids.remove(id(task.node))

        return batch

    def is_full(self) -> bool:
        return len(self._queue) >= self.max_size

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()
        self._node_ids.clear()




















