# 基於 LightZero 的 GPU 批次化評估 MCTS 研究

## 1. 研究目的

本研究以 LightZero 的 AlphaZero MCTS 為基礎，
探討如何在同一棵 MCTS 搜尋樹中收集多個葉節點，
透過 GPU 批次推論降低神經網路評估成本，
並分析批次大小對搜尋效率的影響。

## 2. 研究架構

主要模組：

- MCTS Search Controller
- LeafQueue
- Virtual Loss
- GPU Batch Evaluation
- Expansion and Backpropagation

詳細說明：
- [系統架構](architecture.md)
- [模組介面](interface_contract.md)

## 3. 目前已完成

- WSL2 Ubuntu 開發環境
- RTX 5070 CUDA 與 PyTorch 驗證
- LightZero 原始碼安裝
- AlphaZero Connect4 功能測試
- 序列式 MCTS 基準版本
- 批次搜尋控制器骨架
- 控制器單元測試（3 tests passed）

## 4. 尚未完成

- LeafQueue 正式整合
- Virtual Loss 正式整合
- 同一棵搜尋樹內的 GPU 批次化搜尋
- 正式 GPU 效能實驗

## 5. 成員分工

### 洪浩倫
- MCTS 主程式
- GPU Batch Evaluation 整合
- 搜尋流程及反向傳播
- GPU 效能測試

### 張心人
- LeafQueue 模組
- Virtual Loss 模組
- 單元測試
- 技術文件及實驗結果分析

## 6. 執行測試

```bash
python -m pytest -q tests/project/test_mcts_controller.py
```

## 7. 預計實驗

Batch Size：1、16、32、64、128。

預計分析：
- 搜尋時間
- 每秒評估節點數
- 神經網路推論次數
- GPU 使用情況
- 搜尋正確性

## 8. 專題時程

- 2026/10/30：完成研究與系統框架
- 2026/11：主要模組實作及整合
- 2026/12：效能實驗與最終成果
