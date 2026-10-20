"""
Unit tests for the batched MCTS controller skeleton.

These tests verify the controller interface only.
They do not measure GPU speed or implement batched MCTS.
"""

import pytest
from unittest.mock import patch

from lzero.mcts.ptree.ptree_az import MCTS


def test_batch_size_one_uses_serial_mcts():
    """Batch size 1 should call the existing serial MCTS."""

    mcts = object.__new__(MCTS)

    def fake_serial(self, config, policy_fn, temperature, sample):
        assert policy_fn("test_env") == ("policy", 0.5)
        assert temperature == 1.0
        assert sample is False
        return 3, [0, 0, 0, 1, 0, 0, 0]

    with patch.object(MCTS, "get_next_action", fake_serial):
        action, probs = mcts.get_next_action_batched(
            {},
            lambda env_list: [("policy", 0.5)],
            batch_size=1,
            sample=False
        )

    assert action == 3
    assert probs[3] == 1


def test_batch_size_sixteen_runs_real_search():
    """Batch size 16 can run a real Connect4 MCTS search."""
    import numpy as np
    from easydict import EasyDict
    from zoo.board_games.connect4.envs.connect4_env import Connect4Env

    env = Connect4Env(Connect4Env.default_config())
    env.reset()

    mcts = MCTS(EasyDict(num_simulations=4), env)
    batch_calls = []

    def fake_batch_policy(env_list):
        batch_calls.append(len(env_list))
        return [
            (
                {a: 1.0 / len(e.legal_actions)
                 for a in e.legal_actions},
                0.0
            )
            for e in env_list
        ]

    action, probs = mcts.get_next_action_batched(
        {"start_player_index": 0, "init_state": None},
        fake_batch_policy,
        batch_size=16,
        temperature=1.0,
        sample=False,
    )

    assert action in range(7)
    assert len(probs) == 7
    assert np.isclose(probs.sum(), 1.0)
    assert batch_calls == [1, 4]


def test_invalid_batch_size():
    """Batch size must be at least 1."""

    mcts = object.__new__(MCTS)

    with pytest.raises(ValueError):
        mcts.get_next_action_batched(
            {},
            lambda env_list: [],
            batch_size=0
        )
