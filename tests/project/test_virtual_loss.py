
import importlib.util
from pathlib import Path

import pytest


# Load virtual_loss.py directly to avoid importing all LightZero dependencies.
MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "lzero"
    / "mcts"
    / "ptree"
    / "virtual_loss.py"
)

spec = importlib.util.spec_from_file_location("virtual_loss", MODULE_PATH)
virtual_loss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(virtual_loss)

VirtualLossManager = virtual_loss.VirtualLossManager


class Node:
    """Minimal MCTS node for independent unit testing."""

    def __init__(self, parent=None):
        self._parent = parent
        self._visit_count = 0
        self._value_sum = 0

    @property
    def visit_count(self):
        return self._visit_count

    @property
    def value(self):
        if self._visit_count == 0:
            return 0
        return self._value_sum / self._visit_count

    def update(self, value):
        self._visit_count += 1
        self._value_sum += value



def test_apply_virtual_loss():
    manager = VirtualLossManager()
    root = Node()
    leaf = Node(parent=root)
    leaf.update(0.5)

    token = manager.apply((root, leaf))

    assert root.visit_count == 0
    assert leaf.visit_count == 2
    assert leaf.value == pytest.approx(-0.25)


def test_revert_virtual_loss():
    manager = VirtualLossManager()
    root = Node()
    leaf = Node(parent=root)
    leaf.update(0.5)

    token = manager.apply((root, leaf))
    manager.revert(token)

    assert leaf.visit_count == 1
    assert leaf.value == pytest.approx(0.5)
    assert leaf._value_sum == pytest.approx(0.5)


def test_shared_nodes():
    manager = VirtualLossManager()
    root = Node()
    shared = Node(parent=root)
    leaf_a = Node(parent=shared)
    leaf_b = Node(parent=shared)

    token_a = manager.apply((root, shared, leaf_a))
    token_b = manager.apply((root, shared, leaf_b))

    assert shared.visit_count == 2
    assert shared._value_sum == pytest.approx(-2.0)

    manager.revert(token_a)

    assert shared.visit_count == 1
    assert leaf_a.visit_count == 0
    assert leaf_b.visit_count == 1

    manager.revert(token_b)

    assert shared.visit_count == 0
    assert shared._value_sum == pytest.approx(0)


def test_double_revert():
    manager = VirtualLossManager()
    root = Node()
    leaf = Node(parent=root)

    token = manager.apply((root, leaf))
    manager.revert(token)

    with pytest.raises(ValueError):
        manager.revert(token)

    assert leaf.visit_count == 0
    assert leaf._value_sum == 0


def test_invalid_token():
    manager = VirtualLossManager()

    with pytest.raises(TypeError):
        manager.revert(None)


def test_token_from_another_manager():
    manager_a = VirtualLossManager()
    manager_b = VirtualLossManager()
    root = Node()
    leaf = Node(parent=root)

    token = manager_a.apply((root, leaf))

    with pytest.raises(ValueError):
        manager_b.revert(token)

    assert leaf.visit_count == 1
    manager_a.revert(token)


def test_context_manager_reverts_on_exception():
    manager = VirtualLossManager()
    root = Node()
    leaf = Node(parent=root)

    with pytest.raises(RuntimeError):
        with manager.context((root, leaf)):
            assert leaf.visit_count == 1
            raise RuntimeError("evaluation failed")

    assert leaf.visit_count == 0
    assert leaf._value_sum == 0


@pytest.mark.parametrize(
    "loss",
    [0, -1, float("inf"), float("nan")]
)
def test_invalid_loss(loss):
    manager = VirtualLossManager()
    root = Node()
    leaf = Node(parent=root)

    with pytest.raises(ValueError):
        manager.apply((root, leaf), loss=loss)



def test_empty_path():
    manager = VirtualLossManager()

    with pytest.raises(ValueError):
        manager.apply(())



def test_duplicate_nodes():
    manager = VirtualLossManager()
    root = Node()
    leaf = Node(parent=root)

    with pytest.raises(ValueError):
        manager.apply((root, leaf, leaf))
