from typing import Generic, TypeVar

import numpy as np

State_ = TypeVar("State_")


class PrioQ(Generic[State_]):
    """Approximate priority queue with a fixed length buffer."""

    def __init__(self, capacity: int):
        self.ids = np.array([], dtype=np.int32)
        self.priority = np.array([], dtype=np.float32)
        self.values = []
        self.capacity = capacity

    @property
    def size(self):
        return len(self.ids)

    def add_or_update(
        self, b_id: np.ndarray, b_priority: np.ndarray, b_state: list[State_]
    ):
        # Update if it exists. Append if it doesn't.
        id_to_add = []
        prio_to_add = []
        val_to_add = []
        for ii, id_ in enumerate(b_id):
            if id_ in self.ids:
                idx = np.flatnonzero(self.ids == id_)[0]
                self.priority[idx] = b_priority[ii]
                self.values[idx] = b_state[ii]
            else:
                # Add it to the end.
                id_to_add.append(id_)
                prio_to_add.append(b_priority[ii])
                val_to_add.append(b_state[ii])

        # Add the new values.
        id_to_add = np.array(id_to_add, dtype=np.int32)
        prio_to_add = np.array(prio_to_add, dtype=np.float32)

        self.ids = np.concatenate([self.ids, id_to_add], axis=0)
        self.priority = np.concatenate([self.priority, prio_to_add], axis=0)
        self.values.extend(val_to_add)

        # Keep the priorities sorted.
        idxs_sorted = np.argsort(self.priority)[::-1]

        self.ids = self.ids[idxs_sorted]
        self.priority = self.priority[idxs_sorted]
        self.values = [self.values[ii] for ii in idxs_sorted]

        # Truncate if too large.
        if self.size > self.capacity:
            self.ids = self.ids[: self.capacity]
            self.priority = self.priority[: self.capacity]
            self.values = self.values[: self.capacity]

    def get_top_k(self, k: int) -> tuple[np.ndarray, np.ndarray, list[State_]]:
        assert k <= self.size

        return self.ids[:k], self.priority[:k], self.values[:k]
