"""
Tests for the MCTS batch evaluation bridge.

Uses a fake policy function, not the GPU.
"""

from types import SimpleNamespace

import pytest

from lzero.mcts.ptree.ptree_az import MCTS, Node
from lzero.mcts.ptree.leaf_queue import LeafQueue, LeafTask


def make_task(task_id):
    root = Node()
    leaf = Node(parent=root)
    env = SimpleNamespace(state=task_id)

    return LeafTask(
        task_id, leaf, env, (root, leaf)
    )


def test_batch_evaluation_preserves_mapping():
    mcts = object.__new__(MCTS)
    queue = LeafQueue(max_size=4)

    task_a = make_task(1)
    task_b = make_task(2)

    assert queue.add(task_a)
    assert queue.add(task_b)

    calls = []

    def fake_batch_policy(env_list):
        calls.append(list(env_list))
        return [
            ({0: 0.7}, 0.1),
            ({1: 0.8}, 0.2),
        ]

    results = mcts._evaluate_queued_leaves(
        queue, fake_batch_policy, batch_size=4
    )

    assert len(calls) == 1
    assert calls[0] == [
        task_a.env_snapshot,
        task_b.env_snapshot,
    ]
    assert len(results) == 2
    assert results[0] == (task_a, ({0: 0.7}, 0.1))
    assert results[1] == (task_b, ({1: 0.8}, 0.2))
    assert queue.is_empty()


def test_empty_queue_skips_inference():
    mcts = object.__new__(MCTS)
    queue = LeafQueue(max_size=4)

    def should_not_run(env_list):
        raise AssertionError("Policy should not run")

    results = mcts._evaluate_queued_leaves(
        queue, should_not_run, batch_size=4
    )

    assert results == []


def test_invalid_batch_size():
    mcts = object.__new__(MCTS)
    queue = LeafQueue(max_size=4)

    with pytest.raises(ValueError):
        mcts._evaluate_queued_leaves(
            queue, lambda envs: [], batch_size=0
        )


def test_mismatched_result_count():
    mcts = object.__new__(MCTS)
    queue = LeafQueue(max_size=4)

    assert queue.add(make_task(1))

    with pytest.raises(RuntimeError):
        mcts._evaluate_queued_leaves(
            queue, lambda envs: [], batch_size=4
        )
