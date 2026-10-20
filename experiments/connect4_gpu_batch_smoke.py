import torch
from easydict import EasyDict

from zoo.board_games.connect4.envs.connect4_env import Connect4Env
from lzero.model.alphazero_model import AlphaZeroModel
from lzero.mcts.ptree.ptree_az import MCTS, Node
from lzero.mcts.ptree.leaf_queue import LeafTask, LeafQueue


assert torch.cuda.is_available(), "CUDA is unavailable"
device = torch.device("cuda")

# 使用 LightZero 真正的 AlphaZero 神經網路（尚未訓練）
model = AlphaZeroModel(
    observation_shape=(3, 6, 7),
    action_space_size=7,
    num_res_blocks=1,
    num_channels=64,
).to(device).eval()

env = Connect4Env(Connect4Env.default_config())
env.reset()

mcts = MCTS(EasyDict(num_simulations=4), env)
root = Node()
forward_calls = []


@torch.inference_mode()
def gpu_batch_policy(env_list):
    states = torch.stack([
        torch.from_numpy(e.current_state()[1])
        for e in env_list
    ]).to(device=device, dtype=torch.float32)

    probs, values = model.compute_policy_value(states)
    torch.cuda.synchronize()

    forward_calls.append(len(env_list))
    probs = probs.cpu().numpy()
    values = values.reshape(-1).cpu().numpy()

    return [
        (
            {a: float(probs[i, a]) for a in e.legal_actions},
            float(values[i])
        )
        for i, e in enumerate(env_list)
    ]


# 1. 初始化並擴展根節點
root_policy, root_value = gpu_batch_policy([env])[0]
mcts._expand_leaf_from_result(
    root, env, root_policy, root_value
)

# 2. 從同一棵樹收集 4 個不同葉節點
queue = LeafQueue(max_size=4)
pending = set()

for i in range(4):
    result = mcts._select_leaf_for_batch(
        root,
        {"start_player_index": 0, "init_state": None},
        pending
    )
    assert result is not None, "Cannot find another leaf"

    node, leaf_env, path, done = result
    assert not done, "This smoke test expects nonterminal leaves"

    assert queue.add(LeafTask(i, node, leaf_env, path))
    pending.add(id(node))

print("Queued leaves:", len(queue))

# 3. 一次執行 GPU Batch Evaluation
batch_results = mcts._evaluate_queued_leaves(
    queue, gpu_batch_policy, batch_size=4
)

assert len(batch_results) == 4

# 4. 各自擴展並反向傳播
for task, (policy, value) in batch_results:
    mcts._expand_leaf_from_result(
        task.node, task.env_snapshot, policy, value
    )

    # 此測試使用 self_play_mode，沿用原始 MCTS 正負號規則
    task.node.update_recursive(-value, negate=True)
    pending.remove(id(task.node))

assert root.visit_count == 4
assert not pending
assert queue.is_empty()

print("GPU:", torch.cuda.get_device_name(0))
print("Forward batch sizes:", forward_calls)
print("Root visit count:", root.visit_count)
print("Expanded leaves:", sum(
    1 for task, _ in batch_results if task.node.children
))
print("SUCCESS: Real GPU batch evaluation and MCTS backpropagation!")
