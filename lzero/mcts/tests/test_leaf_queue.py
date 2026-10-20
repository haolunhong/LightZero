import importlib.util
import sys
import unittest
from pathlib import Path


# 直接載入 leaf_queue.py，避免依賴完整 LightZero 環境
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ptree"
    / "leaf_queue.py"
)

spec = importlib.util.spec_from_file_location(
    "leaf_queue", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

LeafTask = module.LeafTask
LeafQueue = module.LeafQueue


class TestLeafQueue(unittest.TestCase):

    def test_add_and_duplicate(self):
        queue = LeafQueue(max_size=2)
        node = object()

        task = LeafTask(
            task_id=1,
            node=node,
            env_snapshot=object(),
            search_path=(node,)
        )

        self.assertTrue(queue.add(task))
        self.assertFalse(queue.add(task))
        self.assertEqual(len(queue), 1)

    def test_pop_batch(self):
        queue = LeafQueue(max_size=4)
        nodes = [object() for _ in range(3)]

        for i, node in enumerate(nodes):
            task = LeafTask(
                task_id=i,
                node=node,
                env_snapshot=object(),
                search_path=(node,)
            )
            self.assertTrue(queue.add(task))

        # 取出前兩筆，確認 FIFO 順序
        batch1 = queue.pop_batch(2)
        self.assertEqual(
            [task.task_id for task in batch1],
            [0, 1]
        )
        self.assertEqual(len(queue), 1)

        # 要求取出 10 筆，但實際只剩 1 筆
        batch2 = queue.pop_batch(10)
        self.assertEqual(
            [task.task_id for task in batch2],
            [2]
        )
        self.assertTrue(queue.is_empty())

        # 已取出的 Node 可以重新加入
        new_task = LeafTask(
            task_id=99,
            node=nodes[0],
            env_snapshot=object(),
            search_path=(nodes[0],)
        )
        self.assertTrue(queue.add(new_task))

    def test_capacity_and_clear(self):
        queue = LeafQueue(max_size=2)

        self.assertTrue(queue.is_empty())
        self.assertFalse(queue.is_full())
        self.assertEqual(len(queue), 0)

        node1 = object()
        node2 = object()
        node3 = object()

        task1 = LeafTask(1, node1, object(), (node1,))
        task2 = LeafTask(2, node2, object(), (node2,))
        task3 = LeafTask(3, node3, object(), (node3,))

        # 加入兩筆任務，Queue 應該已滿
        self.assertTrue(queue.add(task1))
        self.assertTrue(queue.add(task2))

        self.assertTrue(queue.is_full())
        self.assertFalse(queue.is_empty())
        self.assertEqual(len(queue), 2)

        # Queue 已滿，第三筆不能加入
        self.assertFalse(queue.add(task3))
        self.assertEqual(len(queue), 2)

        # 清空 Queue
        queue.clear()

        self.assertTrue(queue.is_empty())
        self.assertFalse(queue.is_full())
        self.assertEqual(len(queue), 0)

        # 清空後，原本的 Node 可以重新加入
        self.assertTrue(queue.add(task1))
        self.assertEqual(len(queue), 1)

    def test_invalid_parameters(self):
        # Queue 容量不能是 0 或負數
        with self.assertRaises(ValueError):
            LeafQueue(max_size=0)

        with self.assertRaises(ValueError):
            LeafQueue(max_size=-1)

        queue = LeafQueue(max_size=2)

        # 批次取出數量不能是 0 或負數
        with self.assertRaises(ValueError):
            queue.pop_batch(0)

        with self.assertRaises(ValueError):
            queue.pop_batch(-1)

        # 空 Queue 取出任務，應回傳空清單
        self.assertEqual(queue.pop_batch(2), [])

        # 空 Queue 執行 clear() 不應出錯
        queue.clear()
        self.assertTrue(queue.is_empty())


if __name__ == "__main__":
    unittest.main()