"""
Interface integration tests for MCTS Node and LeafQueue.

This does not implement batched search or GPU inference.
"""

from types import SimpleNamespace

from lzero.mcts.ptree.ptree_az import Node
from lzero.mcts.ptree.leaf_queue import LeafQueue, LeafTask


def test_mcts_nodes_can_be_queued():
    root = Node()
    leaf_a = Node(parent=root)
    leaf_b = Node(parent=root)

    env_a = SimpleNamespace(state="A")
    env_b = SimpleNamespace(state="B")

    queue = LeafQueue(max_size=2)

    task_a = LeafTask(1, leaf_a, env_a, (root, leaf_a))
    task_b = LeafTask(2, leaf_b, env_b, (root, leaf_b))

    assert queue.add(task_a)
    assert queue.add(task_b)
    assert queue.is_full()

    batch = queue.pop_batch(2)

    assert len(batch) == 2
    assert batch[0].node is leaf_a
    assert batch[1].node is leaf_b
    assert batch[0].env_snapshot is env_a
    assert batch[1].env_snapshot is env_b
    assert batch[0].search_path == (root, leaf_a)
    assert batch[1].search_path == (root, leaf_b)
    assert queue.is_empty()


def test_queue_does_not_change_mcts_statistics():
    root = Node()
    leaf = Node(parent=root)
    queue = LeafQueue(max_size=4)

    task = LeafTask(1, leaf, object(), (root, leaf))

    assert queue.add(task)
    assert not queue.add(task)

    batch = queue.pop_batch(4)

    assert len(batch) == 1
    assert root.visit_count == 0
    assert leaf.visit_count == 0
    assert root.value == 0
    assert leaf.value == 0
