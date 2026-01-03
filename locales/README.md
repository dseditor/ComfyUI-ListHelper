# ComfyUI-ListHelper 國際化 (i18n) 語系檔案

本資料夾包含 ComfyUI-ListHelper 節點的多語言翻譯檔案。

## 資料夾結構

```
locales/
├── en/              # 英文
│   ├── main.json
│   └── nodeDefs.json
├── zh/              # 簡體中文
│   ├── main.json
│   └── nodeDefs.json
├── zh-TW/           # 繁體中文
│   ├── main.json
│   └── nodeDefs.json
└── README.md        # 本說明文件
```

## 語系檔案說明

### main.json
定義節點分類的翻譯：
```json
{
  "nodeCategories": {
    "ListHelper": "列表輔助工具"
  }
}
```

### nodeDefs.json
定義各節點的顯示名稱、輸入輸出參數翻譯：
```json
{
  "AudioListGenerator": {
    "display_name": "音訊分割為列表",
    "inputs": {
      "waveform": { "name": "波形" },
      ...
    },
    "outputs": {
      "0": { "name": "循環" },
      ...
    }
  },
  ...
}
```

## 支援的語系

| 語系代碼 | 語言 | 狀態 |
|---------|------|------|
| `en` | English (英文) | ✓ 完成 |
| `zh` | 简体中文 (簡體中文) | ✓ 完成 |
| `zh-TW` | 繁體中文 (繁體中文) | ✓ 完成 |

## 如何更新翻譯

### 方法一：手動編輯
直接編輯對應語系的 JSON 檔案。

### 方法二：使用轉換腳本（簡繁轉換）

本專案提供 OpenCC 轉換腳本，可自動將繁體中文轉換為簡體中文：

```bash
# 從專案根目錄執行
python -m convert_zh_tw_to_zh
```

**注意**：此腳本會自動覆蓋 `locales/zh/` 下的檔案。

## 翻譯工作流程

1. 編輯 `locales/zh-TW/` 下的繁體中文翻譯
2. 手動編輯 `locales/en/` 下的英文翻譯
3. 執行 `convert_zh_tw_to_zh.py` 生成簡體中文翻譯
4. 執行 `validate_json.py` 驗證所有 JSON 格式

## 驗證 JSON 格式

執行以下命令驗證所有語系檔案的 JSON 格式：

```bash
python -m validate_json
```

## ComfyUI 語系支援

ComfyUI 會自動根據使用者的瀏覽器語言設定載入對應的語系檔案。語系檔案的結構遵循 ComfyUI 的標準格式。

## 貢獻翻譯

歡迎為 ComfyUI-ListHelper 貢獻新的語系翻譯！請參考現有語系的檔案結構創建新的翻譯。

## 技術細節

- 簡繁轉換使用 **OpenCC** (Open Chinese Convert)
- 轉換配置：`t2s` (Traditional to Simplified)
- 所有 JSON 檔案使用 UTF-8 編碼
- 遵循 ComfyUI 官方 i18n 標準

## 維護者

ComfyUI-ListHelper 開發團隊

---

**最後更新**: 2026-01-03
