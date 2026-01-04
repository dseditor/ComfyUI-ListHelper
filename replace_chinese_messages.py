"""
批量替換 photo_magazine_generator.py 中的中文訊息為英文
"""

import re

# 中文到英文的映射
translations = {
    # PhotoMagazinePromptGenerator
    "📸 檢測到參考圖片，建議使用 LLM 節點提取人物特徵": "📸 Reference image detected, recommend using LLM node to extract person features",
    "自動注入 {EXTRACT_FROM_IMAGE} 佔位符": "Auto-injecting {EXTRACT_FROM_IMAGE} placeholder",
    "模板：": "Template:",
    "模特兒：": "Model:",
    "風格：": "Style:",
    "場景：": "Scene:",
    "頁數：": "Pages:",
    "特徵：": "Features:",
    
    # PhotoMagazineParser
    "📝 開始解析 LLM 輸出的 JSON...": "📝 Starting to parse LLM JSON output...",
    "✅ 解析完成！提取到": "✅ Parsing complete! Extracted",
    "個圖片提示詞": "image prompts",
    "錯誤：JSON 輸入為空": "Error: JSON input is empty",
    "錯誤：": "Error:",
    "警告：未找到任何 image_prompt": "Warning: No image_prompt found",
    "解析錯誤：": "Parse error:",
    "  ✓ 封面提示詞": "  ✓ Cover prompt",
    "  ✓ 頁面": "  ✓ Page",
    "提示詞": "prompt",
    "  ✓ 故事頁提示詞": "  ✓ Story page prompt",
    
    # PhotoMagazineMaker
    "成功解析JSON，包含": "Successfully parsed JSON, contains",
    "個頁面": "pages",
    "開始轉換圖片，圖片類型:": "Starting image conversion, image type:",
    "轉換圖片": "Converting image",
    "轉換批次圖片": "Converting batch image",
    "轉換單張圖片:": "Converting single image:",
    "成功轉換": "Successfully converted",
    "張圖片": "images",
    "錯誤：沒有有效的圖片可以處理": "Error: No valid images to process",
    "總共需要": "Total required",
    "個頁面，可用圖片": "pages, available images",
    "圖片分配策略:": "Image allocation strategy:",
    "使用圖片": "Using image",
    "作為封面": "as cover",
    "警告：沒有圖片可用作封面": "Warning: No image available for cover",
    "警告：JSON中的cover不是字典格式（類型:": "Warning: cover in JSON is not dict format (type:",
    "），使用預設值": "), using default values",
    "警告：JSON中缺少cover數據，使用預設值": "Warning: Missing cover data in JSON, using default values",
    "✓ 關閉封面排版，使用第一張圖片作為滿版封面": "✓ Cover layout disabled, using first image as full bleed cover",
    "封面繪製完成": "Cover page complete",
    "JSON中有": "JSON contains",
    "個頁面數據，準備繪製所有內頁": "page data, preparing to draw all content pages",
    "作為第": "as page",
    "頁": "page",
    "頁使用備用圖片": "using fallback image",
    "警告：第": "Warning: Page",
    "頁數據格式錯誤，跳過": "data format error, skipping",
    "頁繪製完成": "page complete",
    "第": "Page",
    "警告：JSON中的story_page不是字典格式（類型:": "Warning: story_page in JSON is not dict format (type:",
    "警告：JSON中缺少story_page數據，使用預設值": "Warning: Missing story_page data in JSON, using default values",
    "作為故事頁": "as story page",
    "故事頁繪製完成": "Story page complete",
    "作為尾頁": "as back cover",
    "尾頁繪製完成": "Back cover complete",
    "PDF 生成成功！": "PDF generated successfully!",
    "檔案位置:": "File location:",
    "生成寫真雜誌時發生錯誤：": "Error generating photo magazine:",
    "JSON 解析錯誤:": "JSON parse error:",
    "，接收到的數據:": ", received data:",
    "JSON格式錯誤：期望字典格式，收到": "JSON format error: expected dict, received",
    "JSON格式錯誤：pages必須是列表格式，收到": "JSON format error: pages must be list, received",
    "JSON數據錯誤：pages列表為空，無法生成雜誌": "JSON data error: pages list is empty, cannot generate magazine",
    "圖片分配結果:": "Image allocation result:",
    "封面=": "cover=",
    "內頁=": "pages=",
    "故事頁=": "story=",
    "尾頁=": "footer=",
    "版型B顯示了": "Layout B displayed",
    "行文字，字體大小:": "lines of text, font size:",
    "行高:": "line height:",
    "框高:": "box height:",
    "最大行數:": "max lines:",
    "滿版圖片創建錯誤:": "Full bleed image creation error:",
    "圖片轉換錯誤:": "Image conversion error:",
}

def replace_chinese_messages(file_path):
    """替換文件中的中文訊息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按照從長到短的順序替換，避免部分匹配
    sorted_translations = sorted(translations.items(), key=lambda x: len(x[0]), reverse=True)
    
    for chinese, english in sorted_translations:
        content = content.replace(chinese, english)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Replaced Chinese messages in {file_path}")

if __name__ == "__main__":
    file_path = r"f:\CUI\ComfyUI\custom_nodes\ComfyUI-ListHelper\photo_magazine_generator.py"
    replace_chinese_messages(file_path)
    print("✓ All Chinese messages replaced with English")
