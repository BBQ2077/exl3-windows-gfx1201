# EXL3 Runtime for AMD RX 9070（gfx1201）— 原生 Windows

在 **原生 Windows** 上用 AMD RX 9070 系列顯卡跑 [ExLlamaV3](https://github.com/turboderp-org/exllamav3)（EXL3 量化模型）的免安裝執行環境。不需要 WSL、Linux、Visual Studio、ROCm／HIP SDK，也不用自己編譯：解壓縮後就能用。

- 內附 CPython 3.12.0、PyTorch `2.13.0+rocm10.1.0`（AMD TheRock）、Triton Windows `3.7.1.post27`、ROCm 10.1 執行庫。
- ExLlamaV3 1.6.2 的 C++／HIP 擴充已針對 `gfx1201` 預先編譯（`native/exllamav3_ext.pyd`）。
- `server.py`：OpenAI 相容 API 伺服器，附 llama.cpp 網頁介面，支援思考模式、工具呼叫、DFlash2 投機解碼、多輪前文快取、看圖。
- 推薦搭配 **[GGUFRun](https://github.com/BBQ2077/GGUFRun)** 使用：在圖形介面選模型、調參數、一鍵啟動。
- 可同時處理多個請求（`--parallel` 1～8，預設 2）：每一路有自己的串流、可各自取消，主模型與草稿各只載一份。
- 加速與快取：RDNA4 多列 GEMV 快路徑（一次 8～64 列）、DFlash2 拒絕取樣、前文快取可放系統 RAM 與 SSD。

## 適用顯卡

ExLlamaV3 的 HIP 核心是針對 `gfx1201` 預先編譯的。啟動時偵測到其他架構只會印出 `WARNING`，照常往下跑，能不能用以實際載入結果為準：

| 顯卡 | 架構 | 狀態 |
|---|---|---|
| Radeon RX 9070 XT／RX 9070／RX 9070 GRE | gfx1201 | ✅（實測：RX 9070 16 GB） |
| Radeon AI PRO R9700（32 GB） | gfx1201 | ✅ |
| Radeon RX 9060 XT／RX 9060 | gfx1200 | 未測試 |
| RX 7000 系列（gfx110x） | gfx110x | 未測試 |

同為 gfx1201 的不同品牌板卡（華碩、技嘉、撼訊…）都能用，不需要重新編譯。

## 系統需求

- Windows 10／11 64 位元。
- AMD Adrenalin 顯示卡驅動：使用近期版本（實測驅動 `32.0.32015.2008`）。runtime 只用系統裝好的驅動，其他 ROCm 元件都自帶。
- 系統 RAM：建議 32 GB（多輪快取預設用 4 GB RAM；視覺塔放 RAM 時另外佔一點）。
- 多路同時生成時，`-cs` 是**每一路**的 context，總 KV 容量＝路數 × `-cs`（例：8 路 × 8192 ＝ 131072 token）。
  顯存不夠時先降 `-cs` 或路數；模型權重仍然只有一份。
- 硬碟：runtime 解壓後約 2.5 GB；Qwen3.8-27B 3.0bpw 模型約 12 GB＋草稿 0.9 GB。

## 下載與安裝

到本倉庫的 **[Releases](../../releases)** 下載兩個檔案，**解壓到同一個資料夾**：

| 檔案 | 內容 | 大小 | 什麼時候要重新下載 |
|---|---|---|---|
| `EXL3-gfx1201-A-python-*.zip` | 基礎環境：`python/`（Python、PyTorch、ROCm、Triton） | 約 0.9 GB | 很少；只有 PyTorch／ROCm 升級時 |
| `EXL3-gfx1201-B-app-*.zip` | 程式本體：ExLlamaV3、預編譯核心、`server.py`、網頁介面、測試 | 約 30 MB | 每次更新 |

更新時通常只要下載新的 B 包覆蓋舊檔；Release 說明會註明要搭配哪一版 A 包。

B 包的內容和本倉庫的檔案相同，所以也可以直接 `git clone` 本倉庫代替 B 包，再把 A 包解壓進 clone 下來的資料夾（`python/` 已列在 `.gitignore`）；之後用 `git pull` 更新。

解壓後在資料夾內執行 `verify.cmd` 檢查：

```bat
verify.cmd
```

看到 `RUNTIME_FILES_VERIFIED`（檔案完整）和 `PRECOMPILED_RUNTIME AMD Radeon RX 9070 ...`（GPU 可用）就是好了。第一次執行 Triton 會建立 GPU 快取，稍慢是正常的。

## 推薦：搭配 GGUFRun 使用

[GGUFRun](https://github.com/BBQ2077/GGUFRun) 是 Windows 本機 LLM 控制台，會自動偵測這份 runtime，選到 EXL3 模型時改用它啟動。資料夾這樣放：

```text
GGUFRun/
  gguf-ui.py、start-ui.bat …
  RUNTIMES/
    exl3-gfx1201/                         ← A 包＋B 包解壓到這裡
  LLM-MODELS-EXL3/
    Qwen3.8-27B-EXL3-SC3.0bpw-H4-V4/      ← 主模型（一個模型一個資料夾）
    Qwen3.8-27B-DFlash2-EXL3-3.0bpw/      ← DFlash2 草稿模型
```

開啟 GGUFRun 後：

1. 主模型下拉選單選 `[EXL3] Qwen3.8-27B-EXL3-SC3.0bpw-H4-V4`，下方會顯示權重／顯存檢查結果。
2. 模式選「投機解碼」，草稿模型保持「自動」，會自動配對 DFlash2 草稿。
3. 「更多設定」可調 Context、KV 型別（q4_0 最省顯存）、思考模式、**視覺（off／gpu／ram）**。
4. 按啟動，等日誌出現 ready 後開網頁即可聊天；其他程式可接 `http://127.0.0.1:<埠>/v1` 的 OpenAI API。

## 模型下載

### Qwen3.8-27B（主模型）

來源：[turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)。不同量化放在不同**分支**，下載時要指定分支：

| 分支 | 大小 | 適合 |
|---|---|---|
| `SC_3.00bpw_H4_V4` | 12.1 GiB | **16 GB 顯卡推薦**；視覺塔也量化過（0.42 GiB） |
| `SC_3.00bpw_H4` | 12.5 GiB | 同上，視覺塔未量化（0.86 GiB） |
| `SC_2.20bpw_H3_V3` | 9.6 GiB | 想留更多顯存給長 context |
| `SC_4.00bpw_H5_V6` | 15.3 GiB | 32 GB 的 R9700；16 GB 卡放不下 |

用官方 Hugging Face CLI 下載（`pip install -U huggingface_hub` 後有 `hf` 指令）：

```bat
hf download turboderp/Qwen3.8-27B-exl3 --revision SC_3.00bpw_H4_V4 --local-dir LLM-MODELS-EXL3\Qwen3.8-27B-EXL3-SC3.0bpw-H4-V4
```

也可以在模型頁的「Files and versions」切換分支後，用瀏覽器逐檔下載，全部放進同一個資料夾。

### DFlash2 草稿模型（投機解碼，選配但強烈建議）

來源：[parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw](https://huggingface.co/parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw)（0.89 GB）。有它寫程式碼約快 3～4 倍（實測約 30 → 104～122 tok/s），輸出內容不變（每個草稿 token 都由主模型驗證）。

```bat
hf download parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw --local-dir LLM-MODELS-EXL3\Qwen3.8-27B-DFlash2-EXL3-3.0bpw
```

### 其他 EXL3 模型

架構列在 `supported-architectures.json` 的 EXL3 模型都能載入。EXL3 的權重和 KV cache 全部放顯卡，**模型大小＋context 必須塞得進顯存**（不能像 GGUF 一樣分層放 CPU）。

## 不用 GGUFRun：直接使用

### 啟動 API 伺服器＋網頁

```bat
server.cmd -m "D:\Models\Qwen3.8-27B-EXL3-SC3.0bpw-H4-V4" -dm "D:\Models\Qwen3.8-27B-DFlash2-EXL3-3.0bpw" ^
  --port 18437 -cs 32768 -cq 4 -rcs 4 -chunk_size 1024 -ndt 7 --thinking on --vision ram
```

日誌出現 `EXL3_SERVER_READY` 後，瀏覽器開 `http://127.0.0.1:18437/` 就是聊天網頁；OpenAI 相容 API 在 `http://127.0.0.1:18437/v1/chat/completions`。伺服器只監聽本機 `127.0.0.1`。

每次請求期間，日誌每 3 秒印一行進度（比照 llama-server），結束時印總結：

```
[EXL3] 預填 1000/2000 tok（50%）｜3.2 s｜312.2 tok/s
[EXL3] 生成 44 tok｜平均 14.5 tok/s｜近 3 秒 14.2 tok/s｜草稿接受 30/40（75%）
[EXL3] 完成：預填 2000 tok／6.48 s（308.8 tok/s）｜快取重用 0 tok｜生成 120 tok／8.38 s（14.3 tok/s）
```

常用參數：

| 參數 | 說明 |
|---|---|
| `-m` | 主模型資料夾 |
| `-dm`／`-ndt` | DFlash2 草稿模型資料夾／一次草稿幾個 token（建議 7） |
| `-cs` | context 長度（token） |
| `-cq` | KV cache 量化位元，例如 `4`、`8`，或 K、V 分開 `4,6`；不填＝FP16 |
| `-rcs` | 多輪前文快取用的系統 RAM（GB） |
| `-chunk_size` | 讀提示詞時一次送進 GPU 的 token 數；開草稿時建議 1024 |
| `--thinking` | `on`／`off`／`auto` 思考模式 |
| `--reasoning-effort` | 思考強度（模型模板有支援時） |
| `--vision` | `off`／`gpu`／`ram`，見下節 |
| `-temp -topp -topk -minp` | 預設取樣參數（請求可覆蓋） |

### 多路同時生成與加速選項

```bat
server.cmd -m "D:\Models\Qwen3.8-27B-EXL3-SC3.0bpw-H4-V4" -dm "D:\Models\Qwen3.8-27B-DFlash2-EXL3-3.0bpw" ^
  --parallel 2 -cs 65536 -cq 4 -rcs 4 --thinking on
```

| 參數 | 說明 |
|---|---|
| `--parallel N` | 同時生成的請求數，1～8，預設 2。每一個請求獨立串流、獨立取消；滿了再送會排隊。 |
| `--auto-draft` | 依「目前同時幾路」分別實測草稿長度與速度，自動挑最快的一檔（0～`-ndt`；0＝暫停草稿）。要快 5% 以上才換，並定期重測。 |
| `--gemv-wmma 0/1` | 多列 GEMV 快路徑，預設開。懷疑數值異常時關掉比對。 |
| `--gemv-max-m 32/64` | 快路徑一次最多幾列，預設 64。改 32 時 33 列以上走較慢的舊路徑（greedy 逐字一致優先）。 |
| `--gemv-wmma-tn 1/2/4` | 快路徑裡每個工作組一次算幾塊相鄰輸出，共用同一份輸入片段。預設 1（與舊核心相同）；2／4 只在 1～4 bit 權重生效。實測 8 路時 4 略快、路數少時反而慢，見下方效能表。 |
| `--dflash2-host-cb` | 實驗：DFlash2 的選擇查表放系統 RAM，省約 240 MB 顯存。 |
| `--qc-staging-sliced` | 實驗：量化 KV 分段搬運，壓低長 Context 的顯存尖峰。 |
| `-ccs GB` | 前文快取放系統 RAM（第二層）。同一段長提示詞再出現時不用重讀。 |
| `-dcd 資料夾` | 前文快取放 SSD（第三層，需 `-ccs > 0`），server 重開後仍有效。 |
| `-dcs GB` | SSD 快取上限，所有模型共用，預設 50 GB。 |
| `-hdt N` | 輸出照抄前文時（改程式碼、複述文件），一次抄 N 個 token 當草稿驗證。留空／0＝關。 |
| `-ngram N` | 不用草稿模型，改從前文找相同片段當草稿；只在沒載草稿模型時生效。 |

核心層的開關（`EXL3_DEC_Q4W`、`EXL3_QC_STAGING`、`EXL3_DEC_GROUP` 等）與自己編譯核心的步驟見 [`BUILD.md`](BUILD.md)。

### 終端機聊天

`models.json` 的 `main`、`draft` 指向主模型和草稿（相對路徑以 runtime 資料夾為準；預設值就是 GGUFRun 的資料夾配置）。之後雙擊 `chat-qwen38.cmd`：32K context、Q4 KV、DFlash2、思考開啟。`/think` 切換思考、`/clear` 清空、`/x` 結束。

其他模型用通用入口，自己指定聊天模板：

```bat
chat.cmd -m "D:\Models\另一個EXL3模型" -mode qwen35 -cs 32768 -cq 4 -think -tps
chat.cmd -m "D:\Models\另一個EXL3模型" -modes
```

## 看圖（視覺）

Qwen3.8-27B 自帶視覺塔。`--vision`（GGUFRun 的「視覺」選項）三選一：

| 選項 | 視覺塔放哪 | 常駐顯存（V4 模型） | 單張圖處理時間（實測） |
|---|---|---|---|
| `off` | 不載入 | 0 | — |
| `gpu` | 顯卡 | 0.25 GiB（未量化版約 0.9 GiB） | 640×800：0.22 秒；958×1198：0.28 秒 |
| `ram` | 系統 RAM，處理圖片時才串流上顯卡 | **0.01 GiB** | 640×800：0.21 秒；958×1198：1.45 秒 |

開啟後網頁會出現附加圖片的按鈕；API 用 OpenAI 格式的 `image_url`（`data:image/...;base64,...`）。同一張圖在多輪對話中只處理一次。視覺運算一定要 GPU 執行，所以 `ram` 是「權重放 RAM」而不是純 CPU 計算。

## 效能與顯存（RX 9070 16 GB 實測）

主模型 Qwen3.8-27B SC3.0bpw＋DFlash2 3.0bpw 草稿、Q4 KV：

| 項目 | 結果 |
|---|---|
| 寫程式碼（500 tokens） | 約 104～122 tok/s，草稿接受率約 76% |
| 一般文章 | 約 45～50 tok/s |
| 不開草稿 | 約 30 tok/s |
| 載入時間 | 約 20～25 秒 |

多路同時生成（2.20bpw 主模型、DFlash2 3.0bpw 草稿、Q4 KV、temp 0.7、8 路各 600 tokens、多列快路徑 64 列；
共同時間窗內的合計吞吐）：

| 設定 | 8 路合計 | 每路 | 生成中顯存 |
|---|---|---|---|
| 8 路、不開草稿（`-cs 16384`） | 206.3 tok/s | 約 25.8 tok/s | 14.69 GiB |
| 8 路、`-ndt 3`（`-cs 8192`） | **245.9 tok/s** | 約 30.8 tok/s | 14.64 GiB |
| 8 路、`--auto-draft`（`-cs 8192`） | 214.1 tok/s | 約 26.8 tok/s | 14.58 GiB |
| 8 路、`-ndt 7`（`-cs 8192`） | 194.3 tok/s | 約 24.3 tok/s | 14.59 GiB |

路數多時草稿深度不是越大越好：8 路 × 草稿 7 ＝ 64 列，驗證成本已經追上收益。`--auto-draft` 會依路數
自己選；也可以手動固定。2 路時（2.20bpw、`-cs 65536`）`--auto-draft` 實測落在草稿 1～3，合計約 79 tok/s。

同一組 8 路設定改成 temperature 0（提示詞相同，8 路輸出幾乎一致）時，草稿接受率掉到 19%，
合計只有 119.8 tok/s（每路 15.3 tok/s）：低溫搭配深度 7 不划算，把 `-ndt` 降到 1～3 或開
`--auto-draft` 比較合適。

以上是單機單次測量的結果，會隨模型量化、Context、取樣參數與背景程式變動；不同量化或溫度不要直接相比。

**顯存快滿時速度會掉很多**：16 GB 卡開 65536 context，模型＋KV 約佔 13 GiB；若 Chrome、桌面等程式又佔了 2.5 GiB 以上，總量逼近 16 GB，Windows 會把部分資料移到共享記憶體，速度可能掉到 40 tok/s 左右。這時：

- context 改成 32768 或 16384；
- 視覺用 `ram` 或 `off`；
- 關掉瀏覽器硬體加速等吃顯存的程式。

## 常見問題

- **`WARNING: kernels were precompiled for gfx1201; detected ...`**：顯卡不是 gfx1201，runtime 照常啟動；若後面出現 `invalid device function` 或 `no kernel image` 一類錯誤，代表核心不相容，需要針對該架構重新編譯。
- **`verify.cmd` 檔案驗證失敗**：A、B 包版本不搭或解壓不完整，重新下載對應版本。
- **防毒軟體攔截**：`python.exe`、`.pyd`、ROCm DLL 都是官方套件原檔，可將資料夾加入例外。
- **路徑有空白**：可以，已測試搬到含空白的路徑仍正常。
- **埠被佔用**：換 `--port`；GGUFRun 會偵測上次留下的 EXL3 server 並詢問是否停止。
- **多路時載入失敗或掉速**：`-cs` 是每路 context，總量＝路數 × `-cs`。8 路 × 65536 會需要 524288 token 的 KV，
  16 GB 卡放不下；先改成 8 路 × 8192（合計 131072）或降低路數。
- **改了核心原始碼卻沒效果**：`.hip` 是 hipify 產生檔，已存在時不會重編。用 `tools\build\build_ext.cmd`
  編譯（會先刪掉舊 `.hip`），細節見 [`BUILD.md`](BUILD.md)。

## 資料夾內容

| 位置 | 用途 |
|---|---|
| `python/` | 內附 Python、PyTorch、ROCm、Triton（A 包） |
| `app/exllamav3/` | ExLlamaV3 1.6.2 原始碼（`app/examples/` 為官方範例與聊天程式） |
| `native/` | 預編譯擴充 `exllamav3_ext.pyd`、`hip_utils.pyd` |
| `app/exllamav3/exllamav3_ext/` | C++／HIP 核心原始碼（自行編譯時使用） |
| `tools/build/` | 重新編譯核心的腳本（`setup_env.cmd`、`build_ext.cmd`） |
| `BUILD.md` | 編譯核心、`.cu`／`.hip` 關係與核心開關說明 |
| `server.py` | OpenAI 相容伺服器 |
| `runtime.py` | 啟動器：隔離環境、設定 DLL 搜尋路徑、檢查 GPU |
| `web/` | 聊天網頁（llama.cpp WebUI） |
| `tests/` | 驗證與測速腳本 |
| `manifest.json` | 所有交付檔案的大小與 SHA-256 |
| `cache/`、`tmp/`、`logs/`、`state/` | 執行時產生（快取、暫存、日誌） |

runtime 以隔離模式執行（`python -I`），不讀系統的 Python、CUDA、ROCm 環境變數，也不會改動系統設定。刪除資料夾即解除安裝。

## 來源與授權

- ExLlamaV3：https://github.com/turboderp-org/exllamav3（MIT）
- PyTorch ROCm（AMD TheRock）：https://github.com/ROCm/TheRock（PyTorch BSD-3-Clause；ROCm 各元件授權見 `licenses/ROCm`）
- Triton Windows：https://github.com/triton-lang/triton-windows（MIT）
- llama.cpp WebUI：https://github.com/ggml-org/llama.cpp（MIT）
- Python：PSF License（`python/LICENSE.txt`）

各套件授權另見 `python/Lib/site-packages/*.dist-info`。模型權重不包含在本 runtime 內，授權依各模型頁面。
