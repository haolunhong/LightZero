# 基於 LightZero 的 GPU 批次化評估 MCTS 研究

> **2026-10-21 初審進度：已完成同一棵 MCTS 搜尋樹的 GPU 批次搜尋可執行原型；Virtual Loss、正式 Policy 串接與效能比較尚待完成。**

## 1. 研究目的

以 LightZero 的 AlphaZero MCTS 為基礎，研究如何在同一棵 MCTS 搜尋樹中收集多個葉節點，使用 GPU 批次推論減少個別評估呼叫，並比較不同 Batch Size 對搜尋效率與正確性的影響。

## 2. 研究架構

主要模組：MCTS Search Controller、LeafQueue、Virtual Loss（待整合）、GPU Batch Evaluation、Expansion 與 Backpropagation。

- [系統架構與目前進度](architecture.md)
- [模組介面契約](interface_contract.md)

## 3. 已完成並有執行紀錄的工作

- WSL2 Ubuntu、CUDA、PyTorch 與 RTX 5070 GPU 環境驗證。
- LightZero 原始碼安裝與 AlphaZero Connect4 基準功能測試。
- 序列式相容基準分支及 GitHub 雙人協作、LeafQueue PR 合併。
- MCTS 的 Evaluation／Expansion 拆分，以及 LeafQueue 的資料結構與介面測試。
- 同一棵 Connect4 搜尋樹成功收集 4 個不同葉節點。
- 真實 GPU 單輪批次評估：Forward batch sizes `[1, 4]`、Root visit count `4`。
- **多輪 `get_next_action_batched()` 原型成功**：12 次 Simulation、Batch Size 4，Forward batch sizes `[1, 4, 4, 4]`，並回傳合法動作與機率分布。
- 研究團隊開發機回報 **15 項單元及整合測試通過**。

以上屬於**功能驗證**；模型尚未訓練，尚未證實 GPU 加速倍率或搜尋品質提升。

## 4. 尚未完成

- Virtual Loss 模組與多輪搜尋控制器的正式整合。
- 原 AlphaZero Policy 呼叫路徑切換至新的 Batched MCTS Controller。
- 終局／平手、不同盤面及玩家視角的 Value 正負號驗證。
- 推論例外時的 Queue／pending 任務恢復、部分 Batch 邊界測試。
- 正式 GPU Benchmark 與統計分析。

## 5. 成員分工

### 洪浩倫

MCTS 主程式、單樹 Selection／搜尋迴圈、GPU Batch Evaluation 串接、Expansion／Backpropagation、GPU 效能量測。

### 張心人

LeafTask／LeafQueue、Virtual Loss、相關測試、技術文件及實驗分析；由兩人共同進行整合驗證及成果整理。

## 6. 執行測試

```bash
# 全部專題單元及介面測試
python -m pytest -q tests/project/ lzero/mcts/tests/test_leaf_queue.py

# 真實 Connect4 + GPU 單輪批次 Smoke Test（需 CUDA）
python experiments/connect4_gpu_batch_smoke.py
```

已回報多輪搜尋 Smoke Test：12 次 Simulation／Batch Size 4／GPU Forward `[1, 4, 4, 4]`。目前該測試由開發機直接呼叫 `get_next_action_batched()`；尚未成為正式 AlphaZero Policy 的預設呼叫流程。

## 7. 預計實驗

先驗證 Batch Size `1、4、8、16`；後續視正確性及 GPU 資源評估是否延伸至 `32、64、128`。

比較搜尋時間、每秒模擬／評估節點數、GPU Forward 呼叫次數及耗時、GPU 使用情況與搜尋正確性。

## 8. 專題時程

- **2026/10/21**：初審討論與多輪批次搜尋原型展示。
- **2026/10/30**：系統框架、關鍵模組、整合證據及階段文件交付。
- **2026/11**：完整整合、正確性與初步效能實驗。
- **2026/12**：系統性效能比較、分析與最終研究成果。
