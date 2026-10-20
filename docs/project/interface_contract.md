# GPU Batched MCTS 模組介面規格

## 一、開發目標

以 LightZero AlphaZero MCTS 為基礎，
實作同一棵搜尋樹內的 GPU 批次化葉節點評估。

本文件為 2026/10/30 專題框架階段的設計規格，
不代表批次化功能已完成。

## 二、工作分工

### 洪浩倫
- MCTS 搜尋主程式
- get_next_action_batched() 介面
- Selection、Expansion、Backpropagation 整合
- GPU Batch Evaluation 串接
- 正式 GPU 效能實驗

### 張心人
- LeafTask 資料結構
- LeafQueue 模組
- Virtual Loss 模組
- 單元測試
- 技術文件與實驗分析

## 三、LeafTask 資料結構

預計包含：

- task_id: int
- node: object
- env_snapshot: object
- search_path: tuple

env_snapshot 必須代表獨立的葉節點環境。
search_path 保存從根節點到葉節點的節點參照。

## 四、LeafQueue 介面

預計提供：

- add(task) -> bool
- pop_batch(size) -> list
- is_full() -> bool
- is_empty() -> bool
- clear() -> None
- __len__() -> int

要求：
- 支援部分批次
- 避免重複排入相同節點
- 保持任務與評估結果的對應順序
- 不直接修改 MCTS 節點統計

注意：
任務取出 Queue 後，仍可能正在等待 GPU 評估，
MCTS 必須另外管理待處理狀態，避免重複選擇。

## 五、MCTS 批次搜尋介面

預計新增：

get_next_action_batched(
    state_config,
    policy_forward_fn_batch,
    batch_size=16,
    temperature=1.0,
    sample=True
)

流程：

1. 建立根節點
2. Selection 選擇葉節點
3. 建立 LeafTask
4. 加入 LeafQueue
5. 暫時管理等待中的搜尋路徑
6. 收集 Batch
7. GPU 批次推論
8. Expansion
9. Backpropagation
10. 輸出動作與機率

## 六、GPU 推論介面

沿用 LightZero 現有：

_policy_value_fn_batch(env_list)

每個輸入環境必須對應一筆輸出結果。

## 七、Virtual Loss

預計提供：

- apply(path)
- revert(path)

必須避免永久污染 visit_count 和 value_sum，
並處理推論失敗時的暫時狀態復原。

## 八、整合測試重點

- 環境副本互相獨立
- 不會重複評估等待中的相同葉節點
- 批次輸出對應正確
- 終局節點處理正確
- Backpropagation 的正負號正確
- 搜尋模擬次數符合設定
- Batch Size 1 可以正常執行
