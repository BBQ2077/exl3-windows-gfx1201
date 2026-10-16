# 精簡執行版驗證結果

## GPU 與套件

- GPU：AMD Radeon RX 9070，`gfx1201`。
- Python：內附 CPython 3.12.0。
- PyTorch：`2.13.0+rocm10.1.0`。
- Triton Windows：`3.7.1.post27`。

## 驗證項目

| 項目 | 實測結果 |
|---|---|
| Triton 向量加法 | 257、4097 元素均通過 |
| PyTorch FP16 矩陣乘法 | 通過 |
| ExLlamaV3 RMSNorm | 通過，最大絕對誤差 0 |
| 模型基本回答 | 中文、JSON、Python 三項通過 |
| 固定生成測試 | 程式碼平均 104.16 tok/s；文章平均 48.50 tok/s，各 2 次、各次 256 tokens |
| 預設聊天入口 | 32,768 上下文、3,470-token 提示詞、29,290-token 回應容量 |
| 思考與 DFlash2 | 思考內容、結束標記、草稿接受統計皆存在 |
| 預設入口的算術題 | 正確回答 21 |
| 模型執行環境稽核 | 程式套件與 DLL 來自本執行版；Windows、AMD 驅動及 Defender 模組依其系統位置載入 |
| 外部編譯程序 | 模型測試及 GPU 探測期間均為 0 |
| PE 匯入依賴 | 檢查 151 個 DLL／PYD／EXE，未解析依賴為 0 |
| 換路徑測試 | 將執行版搬到含空白路徑，重新執行 GPU 數值探測通過，再搬回交付路徑 |

## 結果檔案

執行 `tests/` 內的驗證腳本後，結果寫在 `logs/`：

- `logs/generation-portable-dflash2.json`
- `logs/default-chat-check.json`
- `logs/default-chat-audit.json`
- `logs/relocation-check.json`
- `logs/static-dll-dependencies.json`
- `logs/runtime-audit-*.json`

模型測試使用共用的 Qwen3.8-27B EXL3 主模型及相容的 DFlash2 草稿模型。這份二進位核心供這個 ExLlamaV3 版本支援的模型共用。
