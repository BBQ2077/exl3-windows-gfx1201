# 重新編譯 `native/exllamav3_ext.pyd`

一般使用者不需要這份文件：Releases 的 B 包已含針對 `gfx1201` 編好的核心。只有在改了
`app/exllamav3/exllamav3_ext/` 的 C++／HIP 原始碼，或要換 PyTorch／ROCm 版本時才需要自己編。

## 需求

- Windows 10／11 x64。
- Visual Studio 2022，含「使用 C++ 的桌面開發」工作負載（提供 MSVC 標頭與連結器）。
- [uv](https://docs.astral.sh/uv/)，以及網路（第一次要下載約 4 GB 的編譯環境）。
- 磁碟：`build/` 約需 6～8 GB。編譯期間 CPU 會滿載，約 30～40 分鐘。

## 步驟

```bat
cd <runtime 資料夾>
tools\build\setup_env.cmd
tools\build\build_ext.cmd
```

`setup_env.cmd` 只跑一次，建立 `build\.venv` 並固定版本：

| 套件 | 版本 | 來源 |
|---|---|---|
| Python | 3.12 | uv 管理的 CPython |
| torch | `2.13.0+rocm10.0.0rc4` | `https://rocm.prereleases.amd.com/whl-multi-arch/` |
| rocm（devel） | `10.0.0rc4` | 同上 |

`rocm[devel]` 會把 clang、HIP 標頭與 hipBLAS 解到 `site-packages\_rocm_sdk_devel`；`build_ext.cmd`
用它裡面的 `clang-cl.exe` 當 CXX，並設定 `PYTORCH_ROCM_ARCH=gfx1201`、`DISTUTILS_USE_SDK=1`、
`MAX_JOBS`（預設 CPU 執行緒數）。

`build_ext.py` 做的事：

1. 把 `app/exllamav3` 同步到 `build/src/exllamav3`，只複製位元組有差別的檔案，ninja 因此只重編改到的部分。
2. **刪掉這些檔案的舊 hipify 產物**（`X.hip`、`X_hip.cuh`）。
3. 在 `build/src` 執行 `setup.py build_ext --inplace`。
4. 把新的 `exllamav3_ext.*.pyd` 裝到 `native/exllamav3_ext.pyd`，舊檔留成 `.prev`，
   並更新 `provenance.json` 的 SHA-256。

參數：`--clean`（刪掉 `build/src` 全部重編）、`--no-install`（只編譯，不動 `native\`）。

## `.cu` 改了卻沒效果的原因

PyTorch 的 hipify 遇到已存在的 `X.hip` 會直接跳過（訊息是 `[skipped, already hipified]`），
編譯器讀的是那份舊 `.hip`，不是新的 `.cu`。`build_ext.py` 因此先刪掉對應的 `.hip`／`_hip.cuh` 再編。
手動編譯時若沒做這一步，改動不會進到 `.pyd`。

`.hip`、`_hip.cuh` 是產生檔，不進版控；版控只放 `.cu`／`.cuh` 原始碼。

## 驗證

```bat
verify.cmd                                  :: 檔案完整性與 GPU 可用性
python\python.exe -I -X utf8 -u runtime.py script tests\port\smoke.py
python\python.exe -I -X utf8 -u runtime.py script tests\port\test_gemv32.py
python\python.exe -I -X utf8 -u runtime.py script tests\port\test_gemv64.py
```

`smoke.py` 會印出核心版本、GPU 名稱、擴充函式數量並跑一次矩陣乘法；`test_gemv32/64.py` 比對
多列 GEMV 快路徑與一般路徑的數值。換核心後若 greedy 輸出有人眼可見的差異，先把
`EXL3_GEMV_WMMA=0` 關掉比對，確認是不是快路徑造成的。

## 核心相關環境變數

| 變數 | 預設 | 作用 |
|---|---|---|
| `EXL3_GEMV_WMMA` | `1` | RDNA4 多列 GEMV 快路徑（8～64 列）；`0` 關閉，回到逐列 kernel |
| `EXL3_GEMV_MAX_M` | `64` | 快路徑一次處理的最大列數；設 `32` 時 33 列以上走舊路徑 |
| `EXL3_GEMV_WMMA_U` | 未設 | 快路徑的權重載入變體（測試用） |
| `EXL3_GEMV_WMMA_TN` | `1` | 每個工作組一次算幾塊相鄰輸出（`1`／`2`／`4`），共用同一份輸入片段；`2`／`4` 只用於 1～4 bit 權重，`1` 與舊核心相同 |
| `EXL3_DFLASH_SPEC` | `1` | DFlash2 草稿的拒絕取樣（保留 temperature；`0` 回到只看草稿 token） |
| `EXL3_DFLASH_SPEC_TSCALE` | `0.5` | 上述取樣的溫度縮放 |
| `EXL3_DFLASH2_HOST_CB` | `0` | 草稿選擇表放系統 RAM（省約 240 MB 顯存，`--dflash2-host-cb`） |
| `EXL3_QC_STAGING` | `1` | 量化 KV 的注意力暫存策略；`2` 走 dequantize-then-attend |
| `EXL3_QC_STAGING_PAGES` | `256` | 上述暫存每條序列的頁數上限 |
| `EXL3_QC_STAGING_SLICED` | `0` | 固定大小分段搬運，壓低長 Context 的顯存尖峰（`--qc-staging-sliced`） |
| `EXL3_DEC_GROUP` | `0` | 單一 KV head 的列併成一個 program，減少重複解量化 |
| `EXL3_DEC_SPLIT_MULT` | 未設 | 覆寫 split-decode 的 program 數係數 |
| `EXL3_DEC_WARPS`／`EXL3_DEC_STAGES`／`EXL3_DEC_BLOCK_N` | 未設 | 覆寫 split-decode 的 launch 參數 |
| `EXL3_DEC_Q4W` | `0` | 4-bit KV 的 packed split-decode 路徑 |
| `EXL3_MOE_CPU_SWAP` | `1` | CPU 專家的動態冷熱配置；`0` 固定配置 |
| `EXL3_MOE_CPU_THREADS`／`EXL3_MOE_HOST_CORES` | 未設／`1` | CPU 專家 worker 執行緒數、保留給主執行緒的核心數 |

前面幾個是 `server.py` 的命令列開關（`--gemv-wmma`、`--gemv-max-m`、`--dflash2-host-cb`、
`--qc-staging-sliced`）在載入模型前寫入的環境變數；直接設環境變數也有效。

## 版本對應

`provenance.json` 記錄核心的 SHA-256、對應的 `app/exllamav3` 來源 commit 與編譯環境。
換過核心就把這幾個欄位一起更新，否則 `manifest.json` 與實際檔案會對不起來。
