"""
Tests for separated MCTS Evaluation and Expansion.
"""

from types import SimpleNamespace
from lzero.mcts.ptree.ptree_az import MCTS, Node


def test_expand_from_result():
    """Only legal actions should be expanded."""
    mcts = object.__new__(MCTS)
    root = Node()
    env = SimpleNamespace(legal_actions=[0, 2])

    policy = {
        0: 0.25,
        1: 0.50,
        2: 0.75
    }

    value = mcts._expand_leaf_from_result(
        root, env, policy, 0.4
    )

    assert value == 0.4
    assert set(root.children.keys()) == {0, 2}
    assert root.children[0].prior_p == 0.25
    assert root.children[2].prior_p == 0.75
    assert root.children[0].parent is root
    assert root.visit_count == 0


def test_serial_expansion():
    """Serial evaluation should call the policy function once."""
    mcts = object.__new__(MCTS)
    root = Node()
    env = SimpleNamespace(legal_actions=[3])

    calls = []

    def fake_policy(e):
        calls.append(e)
        return {3: 0.8, 4: 0.2}, -0.25

    value = mcts._expand_leaf_node(
        root, env, fake_policy
    )

    assert len(calls) == 1
    assert calls[0] is env
    assert value == -0.25
    assert set(root.children.keys()) == {3}
    assert root.children[3].prior_p == 0.8
