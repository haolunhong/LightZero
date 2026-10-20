"""
Real GPU single-tree Batched MCTS demonstration.
12 simulations, batch size 4, Connect4.
"""
import runpy
import numpy as np

# Load the already verified real CUDA / AlphaZero model.
ctx = runpy.run_path(
    "experiments/connect4_gpu_batch_smoke.py"
)

mcts = ctx["mcts"]
gpu_batch_policy = ctx["gpu_batch_policy"]
forward_calls = ctx["forward_calls"]

forward_calls.clear()
mcts._num_simulations = 12

action, probs = mcts.get_next_action_batched(
    {"start_player_index": 0, "init_state": None},
    gpu_batch_policy,
    batch_size=4,
    temperature=1.0,
    sample=False,
)

assert 0 <= action <= 6
assert len(probs) == 7
assert np.isclose(probs.sum(), 1.0)
assert forward_calls == [1, 4, 4, 4]

print("\n=== MULTI-ROUND BATCHED MCTS ===")
print("Environment: Connect4")
print("Simulations: 12")
print("Batch size: 4")
print("Selected action:", action)
print("Action probabilities:", probs)
print("GPU forward batch sizes:", forward_calls)
print("SUCCESS: Real multi-round GPU Batched MCTS!")
