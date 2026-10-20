
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class VirtualLossToken:
    """Immutable record of one virtual loss operation."""
    nodes: tuple
    loss: float


class VirtualLossManager:
    """Manage virtual losses applied to MCTS search paths."""

    def __init__(self):
        self._active_tokens = {}

    def apply(self, search_path, loss=1.0):
        """Apply virtual loss and return a token for reverting."""
        if not isinstance(loss, (int, float)) or not isfinite(loss) or loss <= 0:
            raise ValueError("loss must be a positive finite number")

        path = tuple(search_path)

        if not path:
            raise ValueError("search_path must not be empty")

        nodes = path[1:]

        if len({id(node) for node in nodes}) != len(nodes):
            raise ValueError("search_path contains duplicate nodes")

        for node in nodes:
            if not hasattr(node, "_visit_count") or not hasattr(node, "_value_sum"):
                raise TypeError("search_path contains an invalid MCTS node")

        token = VirtualLossToken(nodes=nodes, loss=loss)

        for node in nodes:
            node._visit_count += 1
            node._value_sum -= loss

        self._active_tokens[id(token)] = token
        return token

    def revert(self, token):
        """Revert exactly one previously applied virtual loss."""
        if not isinstance(token, VirtualLossToken):
            raise TypeError("invalid virtual loss token")

        if self._active_tokens.get(id(token)) is not token:
            raise ValueError("virtual loss token is not active")

        for node in token.nodes:
            node._visit_count -= 1
            node._value_sum += token.loss

        
        del self._active_tokens[id(token)]

    #@contextmanager
    def context(self, search_path, loss=1.0):
        """Always revert virtual loss when the block exits."""
        token = self.apply(search_path, loss)
        try:
            yield token
        finally:
            self.revert(token)
