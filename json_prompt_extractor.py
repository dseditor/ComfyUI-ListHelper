"""
共用的 JSON 提示詞提取工具
用於從 LLM 輸出的 JSON 中提取 image_prompt 列表
"""

import json
import re


def extract_prompts_from_json(json_text):
    """
    從 JSON 文本中提取所有 image_prompt
    
    Args:
        json_text: LLM 輸出的 JSON 文本
        
    Returns:
        list: 提示詞列表，如果提取失敗返回空列表
    """
    try:
        if not json_text or not json_text.strip():
            return []
        
        # 清理 markdown 代碼塊標記
        cleaned_text = json_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        # 嘗試解析 JSON
        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # 如果解析失敗，嘗試提取 JSON 部分
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = cleaned_text[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                print("⚠️ 無法從輸出中提取 JSON")
                return []
        
        # 提取所有 image_prompt
        prompts = []
        
        # 1. 封面提示詞
        if isinstance(data, dict) and "cover" in data:
            cover = data["cover"]
            if isinstance(cover, dict) and "image_prompt" in cover:
                prompt = cover["image_prompt"]
                if isinstance(prompt, str) and prompt.strip():
                    prompts.append(prompt.strip())
        
        # 2. 內頁提示詞
        if isinstance(data, dict) and "pages" in data:
            pages = data["pages"]
            if isinstance(pages, list):
                for page in pages:
                    if isinstance(page, dict) and "image_prompt" in page:
                        prompt = page["image_prompt"]
                        if isinstance(prompt, str) and prompt.strip():
                            prompts.append(prompt.strip())
        
        # 3. 故事頁提示詞
        if isinstance(data, dict) and "story_page" in data:
            story = data["story_page"]
            if isinstance(story, dict) and "image_prompt" in story:
                prompt = story["image_prompt"]
                if isinstance(prompt, str) and prompt.strip():
                    prompts.append(prompt.strip())
        
        return prompts
        
    except Exception as e:
        print(f"⚠️ 提取提示詞時發生錯誤: {e}")
        return []


def should_extract_prompts(output_text):
    """
    判斷輸出是否包含 JSON 格式的雜誌數據
    
    Args:
        output_text: LLM 輸出文本
        
    Returns:
        bool: 如果看起來像雜誌 JSON 則返回 True
    """
    if not output_text:
        return False
    
    # 檢查是否包含關鍵字段
    keywords = ["magazine_info", "cover", "pages", "image_prompt"]
    return any(keyword in output_text for keyword in keywords)
