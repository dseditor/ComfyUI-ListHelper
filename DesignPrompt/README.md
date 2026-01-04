# DesignPrompt 模板使用說明

## 📁 資料夾說明

`DesignPrompt/` 資料夾用於存放設計相關的提示詞模板。

---

## 🎯 模板選擇功能

### 自動偵測
PhotoMagazine 提示詞注入器會自動掃描 `DesignPrompt/` 資料夾中的所有 `.md` 檔案，並在節點中提供下拉選單供選擇。

### 使用方式
1. 打開 PhotoMagazine 提示詞注入器節點
2. 在 `template` 參數中選擇想要使用的模板
3. 填寫其他參數（模特兒名稱、風格等）
4. 連接到 LLM 節點

---

## 📝 內建模板

### 1. photomagazine_json_output.md（完整版）
- **用途**：標準寫真雜誌生成
- **特點**：詳細的說明和要求
- **適合**：需要高品質輸出的場景

### 2. photomagazine_simple.md（簡化版）
- **用途**：快速生成寫真雜誌
- **特點**：簡潔的提示詞
- **適合**：快速測試或簡單需求

---

## ✨ 創建自定義模板

### 步驟 1：創建 MD 檔案

在 `DesignPrompt/` 資料夾中創建新的 `.md` 檔案，例如：
```
DesignPrompt/
├─ photomagazine_json_output.md
├─ photomagazine_simple.md
└─ my_custom_template.md  ← 新模板
```

### 步驟 2：編寫模板內容

模板必須支援以下變數：

#### 必要變數
```
{model_name}              # 模特兒名稱
{features}                # 人物特徵
{photo_style}             # 拍攝風格
{custom_scene}            # 場景設定
{content_pages}           # 內容頁數
{features_description}    # 完整的人物特徵說明
```

#### 範例模板結構
```markdown
# 我的自定義模板

## 基本資訊
- 模特兒：{model_name}
- 特徵：{features}
- 風格：{photo_style}
- 場景：{custom_scene}
- 頁數：{content_pages}

## 人物特徵
{features_description}

## 你的自定義內容...

```json
{{
  "magazine_info": {{
    "title": "{model_name}的作品集"
  }},
  ...
}}
```

## 要求
1. 你的自定義要求...
```

### 步驟 3：重啟 ComfyUI

重啟後，新模板會自動出現在下拉選單中。

---

## 🎨 模板範例

### 範例 1：藝術風格模板

**檔案名稱**：`photomagazine_artistic.md`

```markdown
# 藝術寫真集生成

藝術家：{model_name}
風格：{photo_style}
主題：{custom_scene}

創作 {content_pages} 幅藝術作品...
```

### 範例 2：商業攝影模板

**檔案名稱**：`photomagazine_commercial.md`

```markdown
# 商業攝影企劃

客戶：{model_name}
品牌風格：{photo_style}
拍攝場景：{custom_scene}

規劃 {content_pages} 個商業場景...
```

---

## 💡 模板設計建議

### 1. 清晰的結構
- 使用 Markdown 標題組織內容
- 分段說明不同部分

### 2. 詳細的說明
- 解釋 JSON 結構
- 提供範例格式
- 列出具體要求

### 3. 變數使用
- 確保所有必要變數都有使用
- 提供預設值處理（如 `{custom_scene}` 可能為空）

### 4. LLM 友好
- 使用清晰的語言
- 避免歧義
- 提供具體的格式要求

---

## 🔧 進階技巧

### 條件內容

雖然模板本身不支援條件邏輯，但可以在提示詞中提供指引：

```markdown
## 場景說明
{custom_scene}
（如果場景為空，請根據風格自動選擇合適的場景）
```

### 多語言支援

可以創建不同語言版本的模板：

```
DesignPrompt/
├─ photomagazine_zh.md    # 中文版
├─ photomagazine_en.md    # 英文版
└─ photomagazine_ja.md    # 日文版
```

### 專業領域模板

針對不同領域創建專門模板：

```
DesignPrompt/
├─ photomagazine_fashion.md    # 時尚攝影
├─ photomagazine_portrait.md   # 人像攝影
├─ photomagazine_editorial.md  # 編輯攝影
└─ photomagazine_beauty.md     # 美妝攝影
```

---

## 📋 模板檢查清單

創建新模板前，確認：

- [ ] 檔案名稱為 `.md` 結尾
- [ ] 放在 `DesignPrompt/` 資料夾中
- [ ] 包含所有必要變數
- [ ] 使用 `{{` 和 `}}` 表示 JSON 的 `{` 和 `}`
- [ ] 提供清晰的 JSON 結構範例
- [ ] 說明生成要求
- [ ] 測試過變數替換是否正常

---

## 🐛 常見問題

### Q: 新模板沒有出現在下拉選單中？
A: 確認檔案是 `.md` 結尾，並重啟 ComfyUI。

### Q: 模板變數沒有被替換？
A: 檢查變數名稱是否正確，使用 `{model_name}` 而非 `{modelName}`。

### Q: JSON 格式錯誤？
A: 確保使用 `{{` 和 `}}` 來表示 JSON 的花括號。

### Q: 如何刪除模板？
A: 直接刪除 `DesignPrompt/` 資料夾中的 `.md` 檔案，然後重啟 ComfyUI。

---

## 📚 相關資源

- **模板範例**：查看 `photomagazine_json_output.md` 和 `photomagazine_simple.md`
- **節點說明**：參考 `PHOTOMAGAZINE_ARCHITECTURE_V2.md`
- **快速參考**：查看 `PHOTOMAGAZINE_QUICKREF_FINAL.md`

---

**更新時間**：2026-01-04
**版本**：1.0
