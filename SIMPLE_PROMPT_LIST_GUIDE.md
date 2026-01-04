# 簡易提示詞列表生成器使用指南

## 📝 概述

這個模板讓您可以輕鬆生成多組圖片提示詞，使用與寫真雜誌相同的 JSON 格式（但去除封面、封底等雜誌資訊），只保留 `pages` 部分。

---

## 🎯 JSON 格式

### 輸出結構

```json
{
  "pages": [
    {
      "page_number": 1,
      "theme": "主題名稱",
      "description": "簡短描述",
      "image_prompt": "詳細的英文圖片生成提示詞"
    },
    {
      "page_number": 2,
      "theme": "主題名稱",
      "description": "簡短描述",
      "image_prompt": "詳細的英文圖片生成提示詞"
    }
  ]
}
```

### 自動提取

LLM 節點會自動從 JSON 中提取所有 `image_prompt`，輸出為提示詞列表：

```
prompts 輸出 = [
  "image_prompt from page 1",
  "image_prompt from page 2",
  "image_prompt from page 3",
  ...
]
```

---

## 🎯 使用方式

### 1. 選擇模板

在 LLM 節點中：
- **Prompt Template**: 選擇 `simple_prompt_list.md`

### 2. 輸入需求

在 **User Prompt** 中輸入：

```
產生 [數量] 組 [主題] 的提示詞，風格為 [風格]，場景為 [場景]
```

### 3. 連接節點

```
[LLM 節點]
├─ prompt_template: simple_prompt_list.md
├─ user_prompt: "產生10組文具的提示詞，風格為拉拉熊，場景為台灣街景"
├─ text (輸出) → 完整 JSON
└─ prompts (輸出) → 自動提取的提示詞列表
    ↓
[圖片生成節點]
```

---

## 📋 完整範例

### 範例 1: 文具主題

**輸入**：
```
產生10組文具的提示詞，風格為拉拉熊，場景為台灣街景
```

**JSON 輸出** (text):
```json
{
  "pages": [
    {
      "page_number": 1,
      "theme": "Rilakkuma Pencil Case",
      "description": "拉拉熊鉛筆盒在台灣街頭",
      "image_prompt": "A cute Rilakkuma-style pencil case with bear ears and brown color scheme, placed on a traditional Taiwanese street food stall counter, colorful street signs and lanterns in background, warm afternoon sunlight, kawaii aesthetic, product photography, high quality, detailed, 4k"
    },
    {
      "page_number": 2,
      "theme": "Rilakkuma Notebook",
      "description": "拉拉熊筆記本在夜市",
      "image_prompt": "Rilakkuma-themed notebook with cute bear pattern cover, sitting on a red plastic stool at Taiwan night market, neon lights and food stalls in background, vibrant evening atmosphere, kawaii illustration style, warm color palette, professional product shot, detailed, high resolution"
    }
    // ... 更多頁面
  ]
}
```

**Prompts 輸出** (自動提取):
```
[
  "A cute Rilakkuma-style pencil case with bear ears and brown color scheme, placed on a traditional Taiwanese street food stall counter, colorful street signs and lanterns in background, warm afternoon sunlight, kawaii aesthetic, product photography, high quality, detailed, 4k",
  "Rilakkuma-themed notebook with cute bear pattern cover, sitting on a red plastic stool at Taiwan night market, neon lights and food stalls in background, vibrant evening atmosphere, kawaii illustration style, warm color palette, professional product shot, detailed, high resolution",
  ...
]
```

---

### 範例 2: 食物主題

**輸入**：
```
Generate 5 food prompts, realistic style, restaurant setting
```

**Prompts 輸出**:
```
[
  "A gourmet burger with melted cheese, fresh lettuce, tomato, and caramelized onions, served on rustic wooden board in upscale restaurant, dramatic side lighting, professional food photography, steam rising, ultra realistic textures, 8k quality, mouth-watering presentation",
  "Perfectly plated sushi arrangement on black slate plate, various nigiri and maki rolls, modern Japanese restaurant interior background, minimalist aesthetic, natural window light, high-end dining atmosphere, sharp focus on details, realistic textures, professional culinary photography, 4k",
  ...
]
```

---

## 🎨 JSON 欄位說明

### page_number
- **類型**: 整數
- **說明**: 頁面序號，從 1 開始
- **範例**: 1, 2, 3, ...

### theme
- **類型**: 字串（英文）
- **說明**: 簡短的主題名稱
- **範例**: "Rilakkuma Pencil Case", "Gourmet Burger"

### description
- **類型**: 字串（中文）
- **說明**: 簡短描述（20字內）
- **範例**: "拉拉熊鉛筆盒在台灣街頭", "高級漢堡餐點"

### image_prompt
- **類型**: 字串（英文）
- **說明**: 詳細的圖片生成提示詞（100-150 tokens）
- **包含**: 主體、風格、場景、構圖、光線、品質標籤
- **範例**: "A cute Rilakkuma-style pencil case with bear ears..."

---

## 📊 完整工作流程

### 使用 GGUF LLM

```
[GGUF LLM]
├─ model: Qwen3-4B-Q5_K_M.gguf
├─ prompt: "產生10組文具的提示詞，風格為拉拉熊，場景為台灣街景"
├─ prompt_template: simple_prompt_list.md
├─ max_tokens: 3072
└─ outputs:
    ├─ text: 完整 JSON
    └─ prompts: ["prompt1", "prompt2", ...] ⭐
        ↓
[圖片生成節點]
├─ prompt: (連接 prompts)
└─ batch_size: 10
```

### 使用 OpenAI Helper

```
[OpenAI Helper]
├─ config_template: openai.json
├─ user_prompt: "Generate 5 food prompts, realistic style, restaurant setting"
├─ prompt_template: simple_prompt_list.md
├─ max_tokens: 2048
└─ outputs:
    ├─ text: 完整 JSON
    └─ prompts: ["prompt1", "prompt2", ...] ⭐
        ↓
[圖片生成節點]
```

---

## ✨ 優勢

### 1. 標準化格式 ⭐
- ✅ 使用與寫真雜誌相同的 JSON 結構
- ✅ 自動解析 `pages` 中的 `image_prompt`
- ✅ 無需額外的解析節點

### 2. 結構化資訊 ⭐
- ✅ 每個提示詞都有主題和描述
- ✅ 頁面編號便於管理
- ✅ 可追蹤每個提示詞的用途

### 3. 批量生成 ⭐
- ✅ 一次生成多組提示詞
- ✅ 自動保持風格一致性
- ✅ 提供多樣化變化

### 4. 自動提取 ⭐
- ✅ LLM 節點自動提取 `image_prompt`
- ✅ 直接輸出提示詞列表
- ✅ 可直接用於圖片生成

---

## 🎯 最佳實踐

### ✅ 好的輸入

```
產生10組文具的提示詞，風格為拉拉熊，場景為台灣街景
```
- 明確的數量
- 具體的主題
- 清晰的風格
- 詳細的場景

### ❌ 不好的輸入

```
給我一些文具
```
- 數量不明確
- 缺少風格
- 缺少場景

---

## ⚠️ 注意事項

### JSON 格式要求

1. **必須包含 `pages` 陣列**
2. **每個 page 必須有**:
   - `page_number` (整數)
   - `theme` (字串)
   - `description` (字串)
   - `image_prompt` (字串)
3. **page_number 必須從 1 開始連續**

### 常見問題

**Q: prompts 輸出是空的？**

A: 檢查：
1. LLM 是否輸出了有效的 JSON
2. JSON 中是否包含 `pages` 陣列
3. 每個 page 是否有 `image_prompt` 欄位

**Q: 提示詞品質不好？**

A: 嘗試：
- 使用更具體的描述
- 添加更多細節要求
- 使用更強大的 LLM 模型（如 GPT-4）

**Q: JSON 格式錯誤？**

A: 確保：
- LLM 輸出純 JSON（無 markdown 代碼塊）
- JSON 語法正確（逗號、括號等）
- 所有字串都用雙引號

---

## 📚 相關文件

- **模板**: `Prompt/simple_prompt_list.md`
- **LLM 節點**: GGUF LLM, OpenAI Helper, OpenRouter LLM
- **JSON 提取**: 自動從 `pages[].image_prompt` 提取
- **雜誌格式參考**: `DesignPrompt/photomagazine_json_output.md`

---

## 🔧 進階用法

### 自訂欄位

如果需要，可以在 `pages` 中添加額外欄位：

```json
{
  "pages": [
    {
      "page_number": 1,
      "theme": "Rilakkuma Pencil Case",
      "description": "拉拉熊鉛筆盒在台灣街頭",
      "image_prompt": "...",
      "tags": ["stationery", "kawaii", "taiwan"],
      "color_scheme": "brown, cream, pastel"
    }
  ]
}
```

### 與雜誌製作器結合

這個格式可以輕鬆擴展為完整的雜誌格式：

```json
{
  "magazine_info": { ... },
  "cover": { ... },
  "pages": [ ... ],  ← 使用這個模板生成
  "story_page": { ... },
  "back_cover": { ... }
}
```

---

**創建時間**: 2026-01-05
**版本**: 2.0
**格式**: 與寫真雜誌相同的 JSON 結構（僅 pages 部分）
