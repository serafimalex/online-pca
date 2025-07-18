class MaxHeap:
    def __init__(self, items=None):
        """
        items: dict mapping key -> value
        """
        self.heap = []    # list of (neg_value, key)
        self.pos = {}     # key -> index in heap
        if items:
            self._build(items)

    def _build(self, items):
        # initialize heap array and positions
        self.heap = [(-value, key) for key, value in items.items()]
        self.pos = {key: i for i, (_, key) in enumerate(self.heap)}
        # heapify
        for i in reversed(range(len(self.heap) // 2)):
            self._sift_down(i)

    def peek(self):
        """Return (value, key) of max element"""
        if not self.heap:
            return None
        neg_val, key = self.heap[0]
        return -neg_val, key

    def update(self, key, new_value):
        """
        Update the value for the given key and reheapify.
        """
        i = self.pos[key]
        old_neg = self.heap[i][0]
        new_neg = -new_value
        self.heap[i] = (new_neg, key)
        # decide to sift up or down
        if new_neg < old_neg:
            self._sift_up(i)
        else:
            self._sift_down(i)

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] < self.heap[parent]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pos[self.heap[i][1]] = i
        self.pos[self.heap[j][1]] = j


class TwoLevelHeap:
    def __init__(self, matrix):
        """
        matrix: list of lists, size N x M
        """
        self.N = len(matrix)
        self.M = len(matrix[0]) if self.N > 0 else 0
        # store the original values
        self.matrix = [row[:] for row in matrix]
        # per-row heaps
        self.row_heaps = []  # list of MaxHeap, one per row
        row_maxes = {}
        for i, row in enumerate(self.matrix):
            items = {j: row[j] for j in range(self.M)}
            heap = MaxHeap(items)
            self.row_heaps.append(heap)
            max_val, _ = heap.peek()
            row_maxes[i] = max_val
        # global heap over row maxima
        self.global_heap = MaxHeap(row_maxes)

    def update_row(self, i, new_values):
        """Update entire row i with new_values (list of M values)"""
        if i < 0 or i >= self.N:
            raise IndexError(f"Row index {i} out of range [0, {self.N-1}]")
        if len(new_values) != self.M:
            raise ValueError(f"new_values must have length {self.M}")

        # 1. Update matrix and rebuild row heap
        self.matrix[i] = new_values
        self.row_heaps[i] = MaxHeap({j: val for j, val in enumerate(new_values)})  # Rebuild row heap
        
        # 2. Update global heap with new row max
        new_max, _ = self.row_heaps[i].peek()
        self.global_heap.update(i, new_max)

    def update_col(self, j, new_values):
        """Update entire column j with new_values (list of N values)"""
        if j < 0 or j >= self.M:
            raise IndexError(f"Column index {j} out of range [0, {self.M-1}]")
        if len(new_values) != self.N:
            raise ValueError(f"new_values must have length {self.N}")

        for i in range(self.N):
            # Skip if value unchanged (optimization)
            if self.matrix[i][j] == new_values[i]:
                continue
                
            # Update cell and row heap
            self.matrix[i][j] = new_values[i]
            old_row_max, _ = self.row_heaps[i].peek()
            self.row_heaps[i].update(j, new_values[i])
            new_row_max, _ = self.row_heaps[i].peek()

            # Update global heap if row max changed
            if new_row_max != old_row_max:
                self.global_heap.update(i, new_row_max)

    def update(self, i, j, new_value):
        """
        Update cell (i, j) to new_value.
        """
        if not (0 <= i < self.N and 0 <= j < self.M):
            raise IndexError("Cell index out of range")
        # update stored matrix
        self.matrix[i][j] = new_value
        # update row heap
        row_heap = self.row_heaps[i]
        old_row_max, _ = row_heap.peek()
        row_heap.update(j, new_value)
        new_row_max, _ = row_heap.peek()
        # if row max changed, update global heap
        if new_row_max != old_row_max:
            self.global_heap.update(i, new_row_max)

    def get_max(self):
        """Return (value, row, col) of the current global maximum"""
        max_val, i = self.global_heap.peek()
        _, j = self.row_heaps[i].peek()
        return max_val, i, j
