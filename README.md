# vLLM 本地 LLM 測試平台

以 [vLLM](https://github.com/vllm-project/vllm) 為推論後端，搭配 Flask 網頁介面，提供一個可在本地端使用 GPU 加速執行大型語言模型（LLM）的完整測試環境。

---

## 專案架構

```
vllm_test/
├── .env                          # 環境變數設定（模型、GPU、API 金鑰）
├── compose.yaml                  # Docker Compose 服務編排
├── hf-cache/                     # Hugging Face 模型快取（持久化）
│   └── hub/
│       ├── models--google--gemma-4-E4B-it/
│       ├── models--meta-llama--Llama-3.1-8B-Instruct/
│       └── models--openai--gpt-oss-20b/
└── chat-ui/                      # Flask 聊天介面
    ├── app.py                    # Flask 後端（API 路由）
    ├── Dockerfile                # chat-ui 容器映像
    ├── requirements.txt          # Python 相依套件
    └── templates/
        └── index.html            # 前端聊天介面（繁體中文）
```

---

## 服務架構

```
使用者瀏覽器
    ↓
chat-ui（Port 3000）—— Flask + Gunicorn
    ↓ HTTP / SSE
vLLM（Port 8000）—— OpenAI 相容 API
    ↓
LLM 模型（NVIDIA GPU 加速）
```

---

## 環境需求

- Docker & Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit
- Hugging Face 帳號（需有目標模型的存取權限）

---

## 快速開始

### 1. 設定環境變數

編輯 `.env` 檔案：

```env
HF_TOKEN=your_huggingface_token      # HF 存取金鑰
VLLM_API_KEY=your_vllm_api_key       # vLLM API 金鑰（自訂）
MODEL_NAME=google/gemma-4-E4B-it     # 要載入的模型

GPU_COUNT=1                          # 使用的 GPU 數量
GPU_MEMORY_UTIL=0.90                 # GPU 記憶體使用率（90%）
MAX_MODEL_LEN=2048                   # 最大 Context 長度
MAX_NUM_SEQS=1                       # 最大平行序列數
DTYPE=auto                           # 資料型別
```

### 2. 啟動服務

```bash
docker compose up --build
```

首次執行會自動下載模型，需要一些時間。

### 3. 開啟聊天介面

瀏覽器前往：[http://localhost:3000](http://localhost:3000)

### 4. 停止服務

```bash
docker compose down
```

---

## 切換模型

修改 `.env` 中的 `MODEL_NAME`，然後重新啟動服務：

```env
# 可選模型（需已下載至 hf-cache）
MODEL_NAME=google/gemma-4-E4B-it
# MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
# MODEL_NAME=openai/gpt-oss-20b
```

```bash
docker compose down && docker compose up
```

---

## vLLM 參數詳細說明

所有參數皆透過 `.env` 設定後由 `compose.yaml` 傳入 vLLM 容器。以下依類別說明各參數的意義與調整建議。

### 模型與認證

| 參數（.env） | vLLM 旗標 | 說明 |
|---|---|---|
| `MODEL_NAME` | `--model` | Hugging Face 模型 ID，例如 `google/gemma-4-E4B-it`。需確認 HF Token 有該模型存取權限。 |
| `HF_TOKEN` | 環境變數 | Hugging Face API Token，用於下載需授權的模型（如 Llama、Gemma）。 |
| `VLLM_API_KEY` | `--api-key` | 自訂 API 金鑰，呼叫 vLLM 端點時需在 `Authorization: Bearer` 標頭帶入。設為任意字串即可。 |

---

### GPU 與記憶體

| 參數（.env） | vLLM 旗標 | 預設值 | 說明 |
|---|---|---|---|
| `GPU_COUNT` | `deploy.resources` | `1` | Docker 分配給 vLLM 的 GPU 數量。多 GPU 時需搭配 `--tensor-parallel-size`（見下方）。 |
| `GPU_MEMORY_UTIL` | `--gpu-memory-utilization` | `0.90` | vLLM 可使用的 GPU 顯存比例（0.0–1.0）。設太高可能導致 OOM；若系統還有其他 GPU 程序，建議降至 `0.80`。 |
| `DTYPE` | `--dtype` | `auto` | 模型權重的浮點精度。 |

`DTYPE` 可選值說明：

| 值 | 說明 |
|---|---|
| `auto` | 依模型設定自動選擇（推薦，通常為 `bfloat16`） |
| `float16` | 半精度，速度快、記憶體少，部分舊型 GPU（如 T4）限用此值 |
| `bfloat16` | 訓練常用精度，數值穩定性較 float16 好，需 Ampere（A100）以上 GPU |
| `float32` | 全精度，記憶體用量倍增，一般不建議 |

---

### 推理與序列長度

| 參數（.env） | vLLM 旗標 | 預設值 | 說明 |
|---|---|---|---|
| `MAX_MODEL_LEN` | `--max-model-len` | `2048` | 單一請求最大 token 數（prompt + completion 合計）。增加此值需要更多 KV Cache 記憶體，建議依 GPU 顯存調整。 |
| `MAX_NUM_SEQS` | `--max-num-seqs` | `1` | 同時處理的最大請求數（並行批次）。測試環境設 `1` 即可；若要服務多人可提高，但會增加顯存用量。 |

`MAX_MODEL_LEN` 與顯存的粗估關係（以 8B 模型、bfloat16 為例）：

| MAX_MODEL_LEN | 約需額外 KV Cache 顯存 |
|---|---|
| 2048 | ~1 GB |
| 4096 | ~2 GB |
| 8192 | ~4 GB |
| 16384 | ~8 GB |

---

### 執行模式

| vLLM 旗標 | 說明 |
|---|---|
| `--enforce-eager` | 停用 CUDA Graph（預先編譯的推理圖），改用 PyTorch Eager 模式。啟動較慢但相容性高，適合測試環境或顯存有限時使用。生產環境移除此旗標可獲得較好效能。 |

---

### 進階參數（目前未啟用，可手動加入 compose.yaml）

以下參數目前未寫入 `compose.yaml`，有需要時可在 `command:` 區塊中手動加入。

#### 多 GPU 張量並行

```yaml
- --tensor-parallel-size
- "2"       # 使用 2 張 GPU 分割模型權重
```

當模型大小超過單張 GPU 顯存時使用，需搭配 `GPU_COUNT` 設為相同數值。

#### 模型量化

```yaml
- --quantization
- awq       # 可選：awq / gptq / squeezellm / fp8
```

量化可大幅降低顯存用量（約減半），但可能影響回應品質。需使用對應格式的量化模型。

| 量化格式 | 顯存節省 | 速度影響 | 說明 |
|---|---|---|---|
| `awq` | ~50% | 略快或持平 | 目前最常用，品質損失小 |
| `gptq` | ~50% | 略慢 | 較早期的量化方式 |
| `fp8` | ~50% | 快 | 需 Hopper 架構（H100）以上 GPU |

#### KV Cache 最大批次 token 數

```yaml
- --max-num-batched-tokens
- "4096"
```

控制一個批次中所有序列的 token 總量上限，影響吞吐量與記憶體峰值。

#### 信任遠端程式碼

```yaml
- --trust-remote-code
```

部分模型（如某些 Qwen、InternLM 版本）需要執行自訂 tokenizer 程式碼，加入此旗標才能載入。

#### 自訂模型對外名稱

```yaml
- --served-model-name
- my-model
```

API 回應中的 `model` 欄位會改顯示為此名稱，方便整合外部工具時使用固定識別名稱。

#### Swap Space（CPU 記憶體 KV Cache 溢出）

```yaml
- --swap-space
- "4"        # 單位：GB，允許 KV Cache 溢出至 CPU RAM
```

當 GPU 顯存不足以容納所有 KV Cache 時，允許溢出至系統 RAM，但會增加延遲。

---

### 參數調整速查建議

| 情境 | 建議調整 |
|---|---|
| GPU 顯存不足，模型載入失敗 | 降低 `GPU_MEMORY_UTIL`（如 `0.80`）或 `MAX_MODEL_LEN`（如 `1024`） |
| 想要更長的對話脈絡 | 提高 `MAX_MODEL_LEN`（如 `4096`、`8192`），確認顯存足夠 |
| 多人同時使用 | 提高 `MAX_NUM_SEQS`（如 `4`、`8`） |
| 模型過大放不進單張 GPU | 增加 `GPU_COUNT` 並加入 `--tensor-parallel-size` |
| 想降低顯存用量 | 使用量化模型並加入 `--quantization awq` |
| 啟動太慢但推理效能重要 | 移除 `--enforce-eager` |

---

## 聊天介面功能

| 功能 | 說明 |
|------|------|
| 即時串流 | Server-Sent Events（SSE）逐字顯示回應 |
| 系統提示 | 可自訂 System Prompt |
| 參數調整 | Temperature、Max Tokens 即時調整 |
| 推理顯示 | 支援顯示模型的思考過程（reasoning models） |
| Markdown 渲染 | 回應支援 Markdown 及程式碼高亮 |
| 中止生成 | 可隨時停止模型生成 |
| 連線檢查 | 一鍵檢查與 vLLM 的連線狀態 |
| 快速鍵 | `Ctrl+Enter` / `Cmd+Enter` 送出訊息 |

---

## API 端點

chat-ui 後端提供以下端點：

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | 聊天介面首頁 |
| GET | `/api/health` | 健康狀態檢查 |
| GET | `/api/models` | 列出可用模型 |
| POST | `/api/chat` | 一般聊天（非串流） |
| POST | `/api/chat/stream` | 串流聊天（SSE） |

---

## 已快取模型

| 模型 | 說明 |
|------|------|
| `google/gemma-4-E4B-it` | Google Gemma 4（指令微調版） |
| `meta-llama/Llama-3.1-8B-Instruct` | Meta Llama 3.1 8B（指令微調版） |
| `openai/gpt-oss-20b` | OpenAI GPT OSS 20B |

---

## 常見問題

**Q: 模型載入失敗？**
- 確認 `HF_TOKEN` 有該模型的存取權限
- 確認 GPU 記憶體足夠（可調低 `GPU_MEMORY_UTIL` 或 `MAX_MODEL_LEN`）

**Q: 介面無法連線？**
- 點擊介面上的「檢查連線」按鈕確認 vLLM 是否已啟動
- vLLM 初次載入模型需要數分鐘，請稍候

**Q: 想增加 Context 長度？**
- 修改 `.env` 中的 `MAX_MODEL_LEN`（需 GPU 記憶體支援）
