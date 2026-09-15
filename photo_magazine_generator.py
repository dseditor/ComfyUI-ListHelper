import json
import os
from PIL import Image
import torch
import numpy as np
import sys
import subprocess

# ==================== reportlab 延遲載入 ====================
# reportlab 只有 PhotoMagazineMaker 產生 PDF 時才需要。
# 若在模組頂層直接 import，缺少套件時會讓整個 ListHelper 節點套件載入失敗，
# 因此改為延遲載入：真正要出 PDF 時才 import，缺少時才即時安裝。
A4 = None
mm = None
canvas = None
HexColor = None
TTFont = None
registerFont = None
stringWidth = None

_REPORTLAB_READY = False


def _bind_reportlab():
    """把 reportlab 的符號綁定到模組全域變數（供既有程式碼直接使用）"""
    global A4, mm, canvas, HexColor, TTFont, registerFont, stringWidth, _REPORTLAB_READY

    from reportlab.lib.pagesizes import A4 as _A4
    from reportlab.lib.units import mm as _mm
    from reportlab.pdfgen import canvas as _canvas
    from reportlab.lib.colors import HexColor as _HexColor
    from reportlab.pdfbase.ttfonts import TTFont as _TTFont
    from reportlab.pdfbase.pdfmetrics import registerFont as _registerFont, stringWidth as _stringWidth

    A4 = _A4
    mm = _mm
    canvas = _canvas
    HexColor = _HexColor
    TTFont = _TTFont
    registerFont = _registerFont
    stringWidth = _stringWidth
    _REPORTLAB_READY = True


def ensure_reportlab(auto_install=True):
    """
    確保 reportlab 可用。

    Args:
        auto_install: 找不到套件時是否自動安裝

    Raises:
        RuntimeError: 套件不存在且安裝失敗
    """
    if _REPORTLAB_READY:
        return

    try:
        _bind_reportlab()
        return
    except ImportError:
        pass

    if not auto_install:
        raise RuntimeError("需要 reportlab 套件才能產生 PDF，請執行：pip install reportlab")

    print("PhotoMagazine: 未偵測到 reportlab，正在自動安裝（僅需一次）...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    except Exception as e:
        raise RuntimeError(
            f"reportlab 自動安裝失敗：{e}\n請手動執行：{sys.executable} -m pip install reportlab"
        )

    try:
        _bind_reportlab()
    except ImportError as e:
        raise RuntimeError(
            f"reportlab 安裝後仍無法載入：{e}\n請重新啟動 ComfyUI 後再試一次。"
        )

    print("PhotoMagazine: reportlab 安裝完成")


class PhotoMagazinePromptGenerator:
    """
    寫真雜誌prompt生成器
    從 prompts 資料夾讀取模板，注入使用者參數後輸出給 LLM
    """
    def __init__(self):
        pass
    
    def tensor_to_pil(self, tensor):
        """轉換tensor為PIL圖片"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.squeeze()
            if tensor.dim() == 3:
                if tensor.shape[0] in [1, 3, 4]:  # CHW
                    tensor = tensor.permute(1, 2, 0)
                if tensor.shape[2] == 1:  # 灰階
                    tensor = tensor.repeat(1, 1, 3)
                elif tensor.shape[2] == 4:  # RGBA
                    tensor = tensor[:, :, :3]
            
            if tensor.max() <= 1.0:
                tensor = tensor * 255
            
            numpy_image = tensor.cpu().numpy().astype(np.uint8)
            return Image.fromarray(numpy_image)
        return tensor
    
    @classmethod
    def INPUT_TYPES(cls):
        # 獲取 DesignPrompt 資料夾中的模板列表
        design_prompt_dir = os.path.join(os.path.dirname(__file__), "DesignPrompt")
        template_files = ["photomagazine_json_output.md"]  # 預設模板
        
        if os.path.exists(design_prompt_dir):
            # 讀取所有 .md 檔案
            md_files = [f for f in os.listdir(design_prompt_dir) if f.endswith('.md')]
            if md_files:
                template_files = md_files
        
        return {
            "required": {
                "template": (template_files, {"default": template_files[0] if template_files else "photomagazine_json_output.md"}),
                "model_name": ("STRING", {"default": "", "placeholder": "模特兒名稱（例如：小美、Lisa）"}),
                "photo_style": ("STRING", {"default": "自然清新", "placeholder": "拍攝風格（自由輸入）"}),
                "custom_scene": ("STRING", {"default": "", "placeholder": "場景設定（可選）"}),
                "content_pages": ("INT", {"default": 8, "min": 3, "max": 30, "step": 1}),
                "features": ("STRING", {"default": "", "placeholder": "人物特徵（可選，有圖片時自動提取）"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),  # 可選的參考圖片（暫時保留，未來可能移除）
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "DesignPack"
    
    def extract_person_features(self, image_tensor):
        """從圖片中提取人物特徵"""
        try:
            # 轉換 tensor 為 PIL 圖片
            pil_image = self.tensor_to_pil(image_tensor)
            
            # 構建人物特徵提取prompt
            feature_prompt = """請分析這images中的人物，詳細描述以下Features:
1. 國籍/種族特徵
2. 臉型（圓臉、瓜子臉、方臉等）
3. 五官特徵（眼睛、鼻子、嘴巴）
4. 妝容風格
5. 髮型和髮色
6. 其他明顯特徵（眼鏡、飾品等）

請用簡潔的中文描述，約50-80字。"""
            
            # 這裡需要調用 LLM 來分析圖片
            # 由於我們在節點中，可以返回一個提示讓使用者知道需要連接 LLM
            print("📸 Reference image detected, recommend using LLM node to extract person features")
            print("   提示：可以先用 Image to Prompt 節點分析圖片")
            
            # 返回基本的視覺描述（不依賴 LLM）
            return "根據參考圖片的人物特徵"
            
        except Exception as e:
            print(f"圖片特徵提取錯誤: {e}")
            return ""
    
    def generate_prompt(self, template, model_name, photo_style, custom_scene, content_pages, features, reference_image=None):
        """讀取模板並注入參數"""
        try:
            # 如果有參考圖片，自動使用 {EXTRACT_FROM_IMAGE} 佔位符
            if reference_image is not None:
                print("📸 檢測到參考圖片，自動使用 {EXTRACT_FROM_IMAGE} 佔位符")
                features = "{EXTRACT_FROM_IMAGE}"
                print("   提示：請在 LLM 節點中：")
                print("   1. 連接此參考圖片")
                print("   2. 載入 Prompt/extract_person_features.md")
                print("   LLM 會自動提取人物特徵並替換佔位符")
            
            # 讀取模板檔案（從 DesignPrompt 資料夾）
            template_path = os.path.join(os.path.dirname(__file__), "DesignPrompt", template)
            
            if not os.path.exists(template_path):
                return (f"Error:找不到模板檔案 {template_path}",)
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # 準備人物特徵描述
            if features == "{EXTRACT_FROM_IMAGE}":
                features_description = "人物Features:{EXTRACT_FROM_IMAGE}"
            elif features:
                features_description = f"人物Features:{features}"
            else:
                features_description = "人物Features:根據模特兒名稱自行判斷"
            
            # 注入參數
            prompt = template_content.format(
                model_name=model_name,
                features=features if features else "根據模特兒名稱自行判斷",
                photo_style=photo_style,
                custom_scene=custom_scene if custom_scene else "自動判定",
                content_pages=content_pages,
                features_description=features_description
            )
            
            print("📝 prompt生成完成")
            print(f"   Template:{template}")
            print(f"   Model:{model_name}")
            print(f"   Features:{features if features else '自動判定'}")
            if reference_image is not None:
                print(f"   參考圖片：✅ 已提供（將自動提取特徵）")
            print(f"   Style:{photo_style}")
            print(f"   Scene:{custom_scene if custom_scene else '自動判定'}")
            print(f"   Pages:{content_pages}")
            print("💡 請連接到 LLM 節點")
            
            return (prompt,)
            
        except Exception as e:
            import traceback
            error_msg = f"Error:{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (error_msg,)



class PhotoMagazineParser:
    """
    寫真雜誌解析器 - 極簡版
    輸入：LLM 輸出的 JSON 字串（STRING 接口）
    輸出：prompt列表（LIST[STRING]）
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_json_output": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "parse"
    CATEGORY = "DesignPack"
    
    def parse(self, llm_json_output):
        """解析 JSON，提取圖片prompt列表"""
        try:
            if not llm_json_output or not llm_json_output.strip():
                return (["Error: JSON input is empty"],)
            
            print("📝 Starting to parse LLM JSON output...")
            image_prompts = self.extract_prompts(llm_json_output.strip())
            print(f"✅ Parsing complete! Extracted {len(image_prompts)} image prompts")
            return (image_prompts,)
            
        except Exception as e:
            import traceback
            error_msg = f"Error:{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return ([error_msg],)
    
    def extract_prompts(self, json_text):
        """從 JSON 文字中提取所有圖片prompt"""
        try:
            # 清理 JSON 文字（移除 markdown 代碼塊標記）
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            json_text = json_text.strip()
            
            # 嘗試解析 JSON
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                # 如果解析失敗，嘗試提取 JSON 部分
                start_idx = json_text.find('{')
                end_idx = json_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = json_text[start_idx:end_idx]
                    data = json.loads(json_str)
                else:
                    return [f"Error:無法解析 JSON\n內容：{json_text[:200]}..."]
            
            # 提取所有 image_prompt
            prompts = []
            
            # 1. 封面prompt
            if "cover" in data and isinstance(data["cover"], dict):
                prompt = data["cover"].get("image_prompt", "")
                if prompt:
                    prompts.append(str(prompt).strip())
                    print(f"  ✓ Cover prompt")
            
            # 2. 內容pageprompt
            if "pages" in data and isinstance(data["pages"], list):
                for i, page in enumerate(data["pages"]):
                    if isinstance(page, dict):
                        prompt = page.get("image_prompt", "")
                        if prompt:
                            prompts.append(str(prompt).strip())
                            print(f"  ✓ Page {i+1} prompt")
            
            # 3. 故事pageprompt
            if "story_page" in data and isinstance(data["story_page"], dict):
                prompt = data["story_page"].get("image_prompt", "")
                if prompt:
                    prompts.append(str(prompt).strip())
                    print(f"  ✓ Story page prompt")
            
            if not prompts:
                return ["Warning: No image_prompt found"]
            
            return prompts
                    
        except Exception as e:
            import traceback
            return [f"Parse error:{str(e)}\n{traceback.format_exc()}"]

    
    def clean_markdown_content(self, data):
        """清理JSON內容中的markdown標記和符號"""
        import re
        
        def clean_text(text):
            if not isinstance(text, str):
                return text
            
            # 移除markdown標記
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 粗體 **text**
            text = re.sub(r'\*(.*?)\*', r'\1', text)      # 斜體 *text*
            text = re.sub(r'__(.*?)__', r'\1', text)      # 粗體 __text__
            text = re.sub(r'_(.*?)_', r'\1', text)        # 斜體 _text_
            text = re.sub(r'`(.*?)`', r'\1', text)        # 程式碼 `code`
            text = re.sub(r'#+\s*', '', text)             # 標題 # ## ###
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # 連結 [text](url)
            text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text) # 圖片 ![alt](url)
            text = re.sub(r'^\>\s*', '', text, flags=re.MULTILINE)  # 引用 >
            text = re.sub(r'^\s*[-\*\+]\s*', '', text, flags=re.MULTILINE)  # 列表項目
            text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)    # 數字列表
            
            # 清理多餘的空白和換行
            text = re.sub(r'\n\s*\n', '\n', text)  # 多個換行變成單個
            text = text.strip()
            
            return text
        
        def clean_recursive(obj):
            if isinstance(obj, dict):
                return {key: clean_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            elif isinstance(obj, str):
                return clean_text(obj)
            else:
                return obj
        
        return clean_recursive(data)

    def parse_response_and_generate_prompts(self, response_text):
        """解析回應並生成圖片prompt"""
        try:
            # 解析回應並提取JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # 嘗試解析JSON
            try:
                magazine_data = json.loads(response_text)
            except json.JSONDecodeError:
                # 如果JSON解析失敗，嘗試提取JSON部分
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    magazine_data = json.loads(json_str)
                else:
                    return ([f"Error:無法解析為JSON\n回應內容：{response_text[:500]}..."], "")
            
            # 清理markdown標記
            magazine_data = self.clean_markdown_content(magazine_data)
            
            # 提取圖片prompt列表 - 從完整JSON中獲取所有prompts
            image_prompts = []
            
            def extract_prompt(data, key="image_prompt", fallback="portrait photography, professional model"):
                """提取單個prompt的輔助函數"""
                if isinstance(data, dict):
                    prompt = data.get(key, "")
                    if isinstance(prompt, str) and prompt.strip():
                        return prompt.strip()
                    elif isinstance(prompt, dict):
                        if "prompt" in prompt:
                            return str(prompt["prompt"])
                        else:
                            for value in prompt.values():
                                if isinstance(value, str) and value.strip():
                                    return value
                            return fallback
                    else:
                        return str(prompt) if prompt else fallback
                return fallback
            
            # 1. 封面 image_prompt
            cover_data = magazine_data.get("cover", {})
            if cover_data:
                cover_prompt = extract_prompt(cover_data)
                image_prompts.append(cover_prompt)
                print(f"  封面 prompt: {cover_prompt[:50]}...")
            
            # 2. 從 pages 中提取每page的 image_prompt
            pages = magazine_data.get("pages", [])
            for i, page in enumerate(pages):
                page_prompt = extract_prompt(page)
                image_prompts.append(page_prompt)
                print(f"  page面 {i+1} prompt: {page_prompt[:50]}...")
            
            # 3. 故事page image_prompt
            story_data = magazine_data.get("story_page", {})
            if story_data:
                story_prompt = extract_prompt(story_data)
                image_prompts.append(story_prompt)
                print(f"  故事page prompt: {story_prompt[:50]}...")
            
            # 返回完整的 JSON 字串
            json_string = json.dumps(magazine_data, ensure_ascii=False, indent=2)
            
            return (image_prompts, json_string)
                    
        except Exception as e:
            return ([f"Error:{str(e)}"], "")


class PhotoMagazineMaker:
    @classmethod
    def INPUT_TYPES(cls):
        # 獲取字體檔案列表
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        font_files = ["default"]
        if os.path.exists(font_dir):
            font_files.extend([f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.ttc', '.otf'))])
        
        return {
            "required": {
                "images": ("IMAGE",),
                "json_data": ("STRING",),
                "template": (["清新自然", "時尚都市", "復古經典"], {"default": "清新自然"}),
                "layout": (["版型A-經典排版", "版型B-藝術拼貼", "版型C-簡約現代"], {"default": "版型A-經典排版"}),
                "font": (font_files, {"default": font_files[0] if font_files else "default"}),
                "compress_pdf": ("BOOLEAN", {"default": False}),
                "disable_cover_layout": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "關閉封面排版，使用Page一images作為滿版封面（不含文字）"
                }),
                "output_path": ("STRING", {"default": "./ComfyUI/output/MyPDF/photo_magazine.pdf"}),
            }
        }
    
    INPUT_IS_LIST = (True, False, False, False, False, False, False, False)  # 只有images是列表
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "make_photo_magazine"
    CATEGORY = "DesignPack"
    
    def __init__(self):
        self.template_configs = {
            "清新自然": {
                "primary": "#4A90A4",      # 清新藍
                "secondary": "#7FB069",    # 自然綠  
                "accent": "#F7E7CE",       # 米白
                "text": "#2C3E50",         # 深灰
                "background": "#FFFFFF"     # 白色
            },
            "時尚都市": {
                "primary": "#2C3E50",      # 深藍灰
                "secondary": "#E74C3C",    # 時尚紅
                "accent": "#F8F9FA",       # 淺灰
                "text": "#34495E",         # 灰藍
                "background": "#FFFFFF"     # 白色
            },
            "復古經典": {
                "primary": "#8B4513",      # 復古棕
                "secondary": "#DAA520",    # 金黃
                "accent": "#F5F5DC",       # 米色
                "text": "#654321",         # 深棕
                "background": "#FFF8DC"     # 古典白
            }
        }
    
    def tensor_to_pil(self, tensor):
        """轉換tensor為PIL圖片，帶記憶體優化"""
        try:
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.squeeze()
                if tensor.dim() == 3:
                    if tensor.shape[0] in [1, 3, 4]:  # CHW
                        tensor = tensor.permute(1, 2, 0)
                    if tensor.shape[2] == 1:  # 灰階
                        tensor = tensor.repeat(1, 1, 3)
                    elif tensor.shape[2] == 4:  # RGBA
                        tensor = tensor[:, :, :3]
                
                if tensor.max() <= 1.0:
                    tensor = tensor * 255
                
                numpy_image = tensor.cpu().numpy().astype(np.uint8)
                pil_image = Image.fromarray(numpy_image)
                
                # 記憶體優化：限制圖片大小
                max_size = (2048, 2048)
                if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
                    pil_image = pil_image.resize(max_size, Image.Resampling.LANCZOS)
                
                return pil_image
            return tensor
            
        except Exception as e:
            # 備用方案：建立空白圖片
            print(f"Image conversion error: {e}")
            return Image.new('RGB', (800, 600), color='white')
    
    def resize_image_to_fit(self, pil_image, max_width_mm, max_height_mm, dpi=300):
        """調整圖片大小以符合指定的毫米尺寸"""
        try:
            # 轉換毫米到像素
            max_width_px = int(max_width_mm * dpi / 25.4)
            max_height_px = int(max_height_mm * dpi / 25.4)
            
            # 計算縮放比例
            width_ratio = max_width_px / pil_image.width
            height_ratio = max_height_px / pil_image.height
            scale_ratio = min(width_ratio, height_ratio)
            
            # 調整大小
            new_width = int(pil_image.width * scale_ratio)
            new_height = int(pil_image.height * scale_ratio)
            
            resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return resized_image
            
        except Exception as e:
            print(f"圖片調整大小錯誤: {e}")
            return pil_image
    
    def register_font(self, font_name):
        """註冊字體"""
        try:
            if font_name != "default":
                font_path = os.path.join(os.path.dirname(__file__), "fonts", font_name)
                if os.path.exists(font_path):
                    registerFont(TTFont("CustomFont", font_path))
                    return "CustomFont"
            return "Helvetica"
        except:
            return "Helvetica"
    
    def get_fallback_image(self, all_images, used_indices, preferred_index=None):
        """從圖片清單中選擇一張未使用的圖片作為底圖"""
        if not all_images:
            return None
        
        # 如果指定了偏好索引且可用，優先使用
        if preferred_index is not None and preferred_index < len(all_images) and preferred_index not in used_indices:
            return all_images[preferred_index]
        
        # 否則找Page一張未使用的圖片
        for i, img in enumerate(all_images):
            if i not in used_indices:
                return img
        
        # 如果所有圖片都用過了，隨機選一張
        import random
        return random.choice(all_images)
    
    def create_full_bleed_image(self, pil_image, width=None, height=None):
        """創建滿版圖片"""
        # 預設 A4 尺寸需在呼叫時才計算（mm 由 ensure_reportlab 延遲綁定）
        if width is None:
            width = 210 * mm
        if height is None:
            height = 297 * mm
        try:
            # 計算目標比例
            target_ratio = width / height
            image_ratio = pil_image.width / pil_image.height
            
            if image_ratio > target_ratio:
                # 圖片較寬，按高度縮放
                new_height = int(height * 300 / 25.4)  # 轉為像素
                new_width = int(new_height * image_ratio)
            else:
                # 圖片較高，按寬度縮放
                new_width = int(width * 300 / 25.4)  # 轉為像素
                new_height = int(new_width / image_ratio)
            
            # 調整圖片大小
            resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 裁切到目標尺寸
            target_width_px = int(width * 300 / 25.4)
            target_height_px = int(height * 300 / 25.4)
            
            left = (new_width - target_width_px) // 2
            top = (new_height - target_height_px) // 2
            right = left + target_width_px
            bottom = top + target_height_px
            
            cropped_image = resized_image.crop((left, top, right, bottom))
            return cropped_image
            
        except Exception as e:
            print(f"Full bleed image creation error: {e}")
            return pil_image
    
    def wrap_text(self, text, max_width_mm, font_name, font_size, canvas_obj):
        """文字換行處理 - 基於實際字符寬度"""
        lines = []
        # 確保 text 是字符串格式
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        elif not isinstance(text, str):
            text = str(text)
        
        # 按句號和換行符分段
        paragraphs = text.replace('\n', '。').split('。')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            paragraph = paragraph.strip() + '。' if not paragraph.endswith('。') else paragraph.strip()
            words = list(paragraph)  # 中文按字符分割
            current_line = ""
            
            for char in words:
                test_line = current_line + char
                # 計算實際字符寬度
                line_width = stringWidth(test_line, font_name, font_size)
                
                if line_width <= max_width_mm * mm:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = char
            
            if current_line:
                lines.append(current_line)
                
            # 段落間空行
            if paragraph != paragraphs[-1]:
                lines.append("")
        
        return lines
    
    def has_valid_cover_content(self, cover_info):
        """檢查封面是否有有效的文字內容（非空且非None）"""
        if not isinstance(cover_info, dict):
            return False
        
        # 檢查主要文字欄位
        title = cover_info.get("title", "").strip() if cover_info.get("title") else ""
        subtitle = cover_info.get("subtitle", "").strip() if cover_info.get("subtitle") else ""
        description = cover_info.get("description", "").strip() if cover_info.get("description") else ""
        
        # 如果有任何一個有效的文字內容，就認為有內容
        return bool(title or subtitle or description)
    
    def allocate_images_smartly(self, images, num_content_pages):
        """
        智能分配圖片到各pages
        返回一個字典，包含各部分應使用的圖片索引
        """
        total_images = len(images)
        
        # 計算所需圖片數量
        # 封面: 1, 內容page: num_content_pages, 故事page: 1, 尾page: 至少1張（最多4張用於拼貼）
        min_needed = 1 + num_content_pages + 1 + 1  # 最少需求
        
        allocation = {
            "cover": None,
            "pages": [],
            "story": None,
            "footer": []
        }
        
        if total_images == 0:
            print("警告：沒有圖片可分配")
            return allocation
        
        idx = 0
        
        # 1. 封面（優先）
        if idx < total_images:
            allocation["cover"] = idx
            idx += 1
        
        # 2. 內容page
        for i in range(num_content_pages):
            if idx < total_images:
                allocation["pages"].append(idx)
                idx += 1
            else:
                # 圖片不足，重複使用
                allocation["pages"].append(i % total_images)
        
        # 3. 故事page
        if idx < total_images:
            allocation["story"] = idx
            idx += 1
        else:
            # 重複使用Page一張
            allocation["story"] = 0
        
        # 4. 尾page（收集剩餘圖片，最多4張用於四分割）
        remaining_images = total_images - idx
        if remaining_images > 0:
            # 使用剩餘的圖片
            for i in range(min(remaining_images, 4)):
                allocation["footer"].append(idx + i)
        else:
            # 沒有剩餘圖片，使用前面的圖片
            # 優先使用內容page的圖片as back cover拼貼
            if len(allocation["pages"]) >= 4:
                # 如果有足夠的內容page圖片，取最後4張
                allocation["footer"] = allocation["pages"][-4:]
            elif len(allocation["pages"]) > 0:
                # 圖片不足4張，重複使用
                for i in range(4):
                    allocation["footer"].append(allocation["pages"][i % len(allocation["pages"])])
            else:
                # 只有封面圖，重複使用封面
                allocation["footer"] = [0, 0, 0, 0]
        
        return allocation
    
    def draw_cover_page(self, canvas_obj, magazine_data, cover_image, template_config, font_name, layout, compress_enabled=False):
        """繪製封面"""
        try:
            cover_info = magazine_data.get("cover", {})
            # 確保 cover_info 是字典類型
            if not isinstance(cover_info, dict):
                print(f"警告：cover_info 不是字典類型（類型: {type(cover_info)}), using default values")
                cover_info = {}
            print(f"封面數據: {cover_info}")
            
            # 檢查是否有有效的封面內容，如果沒有則僅顯示圖片
            has_content = self.has_valid_cover_content(cover_info)
            print(f"封面是否有有效內容: {has_content}")
            
            if not has_content and cover_image:
                # 純圖片封面：滿版顯示Page一images，不添加任何文字
                print("使用純圖片封面模式")
                full_bleed = self.create_full_bleed_image(cover_image)
                import time
                temp_path = f"temp_cover_simple_{int(time.time() * 1000000)}.jpg"
                quality = self.get_image_quality(compress_enabled)
                full_bleed.save(temp_path, "JPEG", quality=quality)
                
                canvas_obj.drawImage(temp_path, 0, 0, width=210*mm, height=297*mm)
                
                try:
                    os.remove(temp_path)
                except:
                    pass
                return
            
            if layout == "版型A-經典排版":
                # 經典排版：上方滿版圖片 + 下方文字區域
                if cover_image:
                    # 上方滿版圖片（佔據上半部）
                    full_bleed_top = self.create_full_bleed_image(cover_image, width=210*mm, height=200*mm)
                    import time
                    temp_path = f"temp_cover_{int(time.time() * 1000000)}.jpg"
                    quality = self.get_image_quality(compress_enabled)
                    full_bleed_top.save(temp_path, "JPEG", quality=quality)
                    
                    # 滿版圖片（上方）  
                    canvas_obj.drawImage(temp_path, 0, 97*mm, width=210*mm, height=200*mm)
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                else:
                    # 沒有圖片時的上方背景
                    canvas_obj.setFillColor(HexColor(template_config["secondary"]))
                    canvas_obj.rect(0, 97*mm, 210*mm, 200*mm, fill=1, stroke=0)
                
                # 下方文字區域背景
                canvas_obj.setFillColor(HexColor(template_config["background"]))
                canvas_obj.rect(0, 0, 210*mm, 97*mm, fill=1, stroke=0)
                
                # 半透明文字框覆蓋在圖片下方
                canvas_obj.setFillColor(HexColor("#000000"))
                canvas_obj.setFillAlpha(0.7)
                canvas_obj.rect(10*mm, 10*mm, 190*mm, 80*mm, fill=1, stroke=0)
                canvas_obj.setFillAlpha(1)
                
                # 主標題
                canvas_obj.setFont(font_name, 28)
                canvas_obj.setFillColor(HexColor("#FFFFFF"))
                title = cover_info.get("title", "").strip() if cover_info.get("title") else ""
                if not title:
                    title = "寫真集"  # 預設標題
                title_width = stringWidth(title, font_name, 28)
                canvas_obj.drawString((210*mm - title_width) / 2, 65*mm, title)
                
                # 副標題
                canvas_obj.setFont(font_name, 16)
                canvas_obj.setFillColor(HexColor("#E0E0E0"))
                subtitle = cover_info.get("subtitle", "").strip() if cover_info.get("subtitle") else ""
                if subtitle:
                    subtitle_width = stringWidth(subtitle, font_name, 16)
                    canvas_obj.drawString((210*mm - subtitle_width) / 2, 45*mm, subtitle)
                
                # 描述文案（自動換行）
                canvas_obj.setFont(font_name, 12)
                canvas_obj.setFillColor(HexColor("#CCCCCC"))
                description = cover_info.get("description", "").strip() if cover_info.get("description") else ""
                if description:
                    lines = self.wrap_text(description, 170, font_name, 12, canvas_obj)
                    y_pos = 30*mm
                    for line in lines[:3]:  # 最多3行
                        if line.strip():
                            line_width = stringWidth(line, font_name, 12)
                            canvas_obj.drawString((210*mm - line_width) / 2, y_pos, line)
                        y_pos -= 6
            
            elif layout == "版型B-藝術拼貼":
                # 版型B：滿版圖片 + 粗體文字直接壓在底圖上（取消透明框）
                if cover_image:
                    full_bleed = self.create_full_bleed_image(cover_image)
                    import time
                    temp_path = f"temp_cover_b_{int(time.time() * 1000000)}.jpg"
                    full_bleed.save(temp_path, "JPEG", quality=95)
                    canvas_obj.drawImage(temp_path, 0, 0, width=210*mm, height=297*mm)
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # 主標題（左上，無背景框，放大字型）
                title = cover_info.get("title", "").strip() if cover_info.get("title") else ""
                if title:
                    canvas_obj.setFont(font_name, 32)  # 放大標題
                    canvas_obj.setFillColor(HexColor("#FFFFFF"))
                    canvas_obj.setStrokeColor(HexColor("#000000"))
                    canvas_obj.setLineWidth(1.0)
                    canvas_obj.drawString(25*mm, 245*mm, title)
                
                # 副標題（右下，無背景框，放大字型）
                subtitle = cover_info.get("subtitle", "").strip() if cover_info.get("subtitle") else ""
                if subtitle:
                    canvas_obj.setFont(font_name, 18)  # 放大副標題
                    canvas_obj.setFillColor(HexColor("#FFFFFF"))
                    canvas_obj.setStrokeColor(HexColor("#000000"))
                    canvas_obj.setLineWidth(0.8)
                    subtitle_width = stringWidth(subtitle, font_name, 18)
                    canvas_obj.drawString(210*mm - subtitle_width - 15*mm, 35*mm, subtitle)
                
                # 描述文案（中心左側，無背景框，放大字型）
                description = cover_info.get("description", "").strip() if cover_info.get("description") else ""
                if description:
                    canvas_obj.setFont(font_name, 14)  # 放大描述文字
                    canvas_obj.setFillColor(HexColor("#FFFFFF"))
                    canvas_obj.setStrokeColor(HexColor("#000000"))
                    canvas_obj.setLineWidth(0.6)
                    
                    lines = self.wrap_text(description, 110, font_name, 14, canvas_obj)
                    y_position = 150*mm
                    line_height = 16
                    
                    for line in lines[:4]:  # 最多顯示4行
                        if y_position < 100*mm:
                            break
                        if line.strip():
                            canvas_obj.drawString(20*mm, y_position, line)
                        y_position -= line_height
            
            elif layout == "版型C-簡約現代":
                # 現代簡約：非對稱布局，左側裁切圖片 + 右側文字
                # 根據文字內容計算適當的版面寬度
                all_text_content = ""
                if cover_info.get("title"):
                    all_text_content += cover_info.get("title")
                if cover_info.get("subtitle"):
                    all_text_content += cover_info.get("subtitle")
                if cover_info.get("description"):
                    all_text_content += cover_info.get("description")
                
                text_area_width = self.calculate_adaptive_layout_width(all_text_content, 70)
                image_area_width = 210 - text_area_width  # 剩餘空間給圖片
                
                # 背景色
                canvas_obj.setFillColor(HexColor(template_config["accent"]))
                canvas_obj.rect(0, 0, 210*mm, 297*mm, fill=1, stroke=0)
                
                # 左側圖片區域（適應性寬度）- 按比例裁切
                if cover_image:
                    # 目標區域：動態寬度 x 297mm (全高)
                    target_width = image_area_width
                    target_height = 297
                    target_ratio = target_width / target_height
                    
                    # 原圖比例
                    img_ratio = cover_image.width / cover_image.height
                    
                    if img_ratio > target_ratio:
                        # 圖片太寬，需要裁切寬度
                        new_height = cover_image.height
                        new_width = int(new_height * target_ratio)
                        left = (cover_image.width - new_width) // 2
                        cropped_image = cover_image.crop((left, 0, left + new_width, new_height))
                    else:
                        # 圖片太高，需要裁切高度
                        new_width = cover_image.width
                        new_height = int(new_width / target_ratio)
                        top = (cover_image.height - new_height) // 2
                        cropped_image = cover_image.crop((0, top, new_width, top + new_height))
                    
                    # 調整到目標尺寸
                    target_px_w = int(target_width * 300 / 25.4)  # mm轉像素
                    target_px_h = int(target_height * 300 / 25.4)
                    resized_image = cropped_image.resize((target_px_w, target_px_h), Image.Resampling.LANCZOS)
                    
                    import time
                    temp_path = f"temp_cover_{int(time.time() * 1000000)}.jpg"
                    resized_image.save(temp_path, "JPEG", quality=95)
                    
                    canvas_obj.drawImage(temp_path, 0, 0, width=image_area_width*mm, height=297*mm)
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                else:
                    # 沒有圖片時的左側背景
                    canvas_obj.setFillColor(HexColor(template_config["secondary"]))
                    canvas_obj.rect(0, 0, image_area_width*mm, 297*mm, fill=1, stroke=0)
                
                # 右側文字區域（垂直排列）
                canvas_obj.setFillColor(HexColor(template_config["background"]))
                canvas_obj.rect(image_area_width*mm, 0, text_area_width*mm, 297*mm, fill=1, stroke=0)
                
                # 垂直文字排列
                canvas_obj.setFont(font_name, 28)
                canvas_obj.setFillColor(HexColor(template_config["primary"]))
                title = cover_info.get("title", "").strip() if cover_info.get("title") else ""
                if not title:
                    title = "寫真集"  # 預設標題
                
                # 計算文字區域中心位置
                text_center_x = image_area_width + text_area_width / 2
                
                # 從上到下繪製標題字元
                y_pos = 250*mm
                for char in title:
                    char_width = stringWidth(char, font_name, 28)
                    canvas_obj.drawString(text_center_x*mm - char_width/2, y_pos, char)
                    y_pos -= 35
                
                # 副標題（水平）- 調整位置避免與 description 重疊
                canvas_obj.setFont(font_name, 16)
                canvas_obj.setFillColor(HexColor(template_config["secondary"]))
                subtitle = cover_info.get("subtitle", "").strip() if cover_info.get("subtitle") else ""
                if subtitle:
                    # 計算 subtitle 需要的行數
                    available_width = text_area_width - 10
                    subtitle_lines = self.wrap_text(subtitle, available_width, font_name, 16, canvas_obj)
                    subtitle_height = len(subtitle_lines) * 20  # 每行約 20
                    
                    # 從較高位置開始，為 description 留出空間
                    subtitle_y = 120*mm
                    for line in subtitle_lines[:3]:  # 最多 3 行
                        if line.strip():
                            line_width = stringWidth(line, font_name, 16)
                            canvas_obj.drawString(text_center_x*mm - line_width/2, subtitle_y, line)
                        subtitle_y -= 20
                
                # 描述文案（水平，自動換行）- 調整起始位置
                description = cover_info.get("description", "").strip() if cover_info.get("description") else ""
                if description:
                    # 使用適應性字體大小和文字寬度
                    available_width = text_area_width - 10  # 留一些邊距
                    adaptive_font_size = self.calculate_adaptive_font_size(description, available_width, 12, font_name, canvas_obj)
                    canvas_obj.setFont(font_name, adaptive_font_size)
                    canvas_obj.setFillColor(HexColor(template_config["text"]))
                    
                    lines = self.wrap_text(description, available_width, font_name, adaptive_font_size, canvas_obj)
                    
                    # 根據是否有 subtitle 調整起始位置
                    if subtitle:
                        y_pos = subtitle_y - 10  # 在 subtitle 下方，留 10 間距
                    else:
                        y_pos = 100*mm  # 如果沒有 subtitle，從較高位置開始
                    
                    line_height = adaptive_font_size + 2
                    
                    for line in lines[:min(12, len(lines))]:  # 最多12行，適應性顯示
                        if line.strip() and y_pos > 10*mm:  # 確保不超出page面
                            line_width = stringWidth(line, font_name, adaptive_font_size)
                            canvas_obj.drawString(text_center_x*mm - line_width/2, y_pos, line)
                        y_pos -= line_height
            
        except Exception as e:
            print(f"封面繪製錯誤: {e}")
    
    def calculate_adaptive_layout_width(self, text_content, max_width_mm=70):
        """根據文字內容長度計算適當的版面寬度（針對中文優化）"""
        if not text_content:
            return max_width_mm
        
        # 計算中文寬度係數
        width_factor = self.calculate_chinese_text_width_factor(str(text_content))
        
        # 估算文字長度，中文文字需要更寬的版面
        text_length = len(str(text_content))
        
        if text_length > 200:
            # 長文字：基礎寬度 + 中文係數調整
            adjusted_width = int((max_width_mm + 30) * width_factor)
            return min(adjusted_width, 105)
        elif text_length > 100:
            # 中等文字：適度增加寬度
            adjusted_width = int((max_width_mm + 15) * width_factor)
            return min(adjusted_width, 95)
        else:
            # 短文字：根據中文比例微調
            adjusted_width = int(max_width_mm * width_factor)
            return min(adjusted_width, max_width_mm + 10)
    
    def calculate_adaptive_font_size(self, text, max_width_mm, base_font_size, font_name, canvas_obj):
        """根據文字內容和可用空間計算適當字體大小"""
        if not text:
            return base_font_size
        
        # 測試不同字體大小，找到適合的尺寸
        for font_size in range(base_font_size, max(6, base_font_size - 6), -1):
            test_lines = self.wrap_text(text, max_width_mm, font_name, font_size, canvas_obj)
            total_height = len(test_lines) * font_size * 1.2  # 估算總高度
            
            # 如果文字能在合理範圍內顯示，使用此字體大小
            if len(test_lines) <= 15 and total_height <= 200:  # 最多15行，高度不超過200mm
                return font_size
        
        return max(6, base_font_size - 6)  # 最小字體大小為6
    
    def calculate_text_display_width(self, text, font_name, font_size):
        """計算文字實際顯示寬度（考慮中文字符）"""
        if not text:
            return 0
        
        # 使用stringWidth精確計算實際寬度
        total_width = stringWidth(text, font_name, font_size)
        return total_width

    def calculate_chinese_text_width_factor(self, text):
        """計算中文文字的寬度係數，用於更準確的版面計算"""
        if not text:
            return 1.0
        
        # 統計中文字符數量
        chinese_chars = 0
        total_chars = len(text)
        
        for char in text:
            # 檢查是否為中文字符（包括中文標點）
            if '\u4e00' <= char <= '\u9fff' or '\uff00' <= char <= '\uffef':
                chinese_chars += 1
        
        # 中文字符比例越高，需要的寬度係數越大
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        
        # 中文字符通常比英文字符寬1.5-2倍
        width_factor = 1.0 + (chinese_ratio * 0.8)  # 最大增加80%寬度
        return width_factor
    
    def calculate_text_box_dimensions(self, theme, description, font_name, canvas_obj):
        """計算文字框所需的長寬尺寸 - 改進版本，支持動態文字適應"""
        # 設定字體大小
        title_font_size = 14
        desc_font_size = 11
        
        # 設定邊距和間距
        padding = 8*mm
        line_height = desc_font_size + 2  # 根據字體大小動態調整行高
        title_desc_spacing = 3*mm
        
        # 計算主題標題寬度
        theme_width = stringWidth(theme, font_name, title_font_size) if theme else 0
        
        # 動態計算最大文字寬度，根據文字長度調整
        text_length = len(description) if description else 0
        if text_length > 200:
            # 長文字：增加寬度到150mm
            max_line_width = 150*mm
        elif text_length > 100:
            # 中等文字：寬度130mm
            max_line_width = 130*mm
        else:
            # 短文字：保持120mm
            max_line_width = 120*mm
        
        # 確保不超過page面寬度（留出邊距）
        page_max_width = 180*mm  # A4寬度210mm - 左右各15mm邊距
        max_line_width = min(max_line_width, page_max_width)
        
        desc_lines = []
        if description:
            # 使用動態寬度進行文字換行
            desc_lines = self.wrap_text(description, max_line_width/mm, font_name, desc_font_size, canvas_obj)
        
        # 計算描述文字實際最大寬度
        desc_max_width = 0
        if desc_lines:
            for line in desc_lines:
                line_width = stringWidth(line, font_name, desc_font_size)
                if line_width > desc_max_width:
                    desc_max_width = line_width
        
        # 計算所需寬度（取較大者，但至少要有最小寬度）
        min_box_width = 80*mm  # 最小文字框寬度
        content_width = max(theme_width, desc_max_width, min_box_width - padding * 2)
        box_width = content_width + padding * 2
        
        # 確保文字框不超過page面寬度
        if box_width > page_max_width:
            box_width = page_max_width
            # 重新計算內容寬度和換行
            content_width = box_width - padding * 2
            if description:
                desc_lines = self.wrap_text(description, (content_width)/mm, font_name, desc_font_size, canvas_obj)
        
        # 計算所需高度
        title_height = title_font_size + 2 if theme else 0  # 為標題添加小間距
        desc_height = len(desc_lines) * line_height if desc_lines else 0
        spacing = title_desc_spacing if theme and desc_lines else 0
        
        # 最小高度保證
        min_box_height = 30*mm
        box_height = max(padding * 2 + title_height + spacing + desc_height, min_box_height)
        
        return box_width, box_height, desc_lines
    
    def draw_content_page(self, canvas_obj, page_data, page_image, template_config, font_name, layout, compress_enabled=False):
        """繪製內page（滿版圖片 + 統一的文字框）"""
        try:
            print(f"繪製內page，數據: {page_data}")
            
            if page_image:
                print("開始繪製滿版圖片")
                # 滿版圖片
                full_bleed = self.create_full_bleed_image(page_image)
                
                # 儲存圖片為臨時檔案
                import time
                temp_path = f"temp_page_{int(time.time() * 1000000)}.jpg"
                quality = self.get_image_quality(compress_enabled)
                full_bleed.save(temp_path, "JPEG", quality=quality)
                
                # 繪製到PDF
                canvas_obj.drawImage(temp_path, 0, 0, width=210*mm, height=297*mm)
                print("滿版圖片繪製完成")
                
                # 清理臨時檔案
                try:
                    os.remove(temp_path)
                except:
                    pass
            else:
                # 沒有圖片時應該不會發生，因為已經有備用圖片邏輯
                print("警告：內容page備用圖片邏輯失效，使用純色背景")
                canvas_obj.setFillColor(HexColor(template_config["background"]))
                canvas_obj.rect(0, 0, 210*mm, 297*mm, fill=1, stroke=0)
            
            # 獲取page面文字內容
            theme = page_data.get("theme", "").strip() if page_data.get("theme") else ""
            description = page_data.get("description", "").strip() if page_data.get("description") else ""
            
            # 統一的文字框處理（所有版型都相同）
            if theme or description:
                print(f"添加page面文字 - 主題: {theme}, 描述: {description[:20]}...")
                
                # 計算文字框尺寸
                box_width, box_height, desc_lines = self.calculate_text_box_dimensions(
                    theme, description, font_name, canvas_obj
                )
                
                # 設定間距（A4右下角為底部參考）
                margin_right = 15*mm
                margin_bottom = 15*mm
                
                # 計算文字框位置（以A4右下角為參考）
                box_x = 210*mm - box_width - margin_right
                box_y = margin_bottom
                
                # 確保文字框不超出page面邊界
                if box_x < 5*mm:
                    box_x = 5*mm
                    box_width = 210*mm - 10*mm  # 調整寬度適應page面
                
                if box_y + box_height > 297*mm - 5*mm:
                    box_y = 297*mm - box_height - 5*mm
                
                # 繪製半透明背景框
                canvas_obj.setFillColor(HexColor("#000000"))
                canvas_obj.setFillAlpha(0.7)
                canvas_obj.rect(box_x, box_y, box_width, box_height, fill=1, stroke=0)
                canvas_obj.setFillAlpha(1)
                
                # 文字位置計算
                text_x = box_x + 8*mm
                text_y = box_y + box_height - 8*mm
                
                # 繪製主題標題
                if theme:
                    canvas_obj.setFont(font_name, 14)
                    canvas_obj.setFillColor(HexColor("#FFFFFF"))
                    canvas_obj.drawString(text_x, text_y, theme)
                    text_y -= 17*mm  # 標題與描述間距
                
                # 繪製描述文字
                if desc_lines:
                    canvas_obj.setFont(font_name, 11)
                    canvas_obj.setFillColor(HexColor("#E0E0E0"))
                    
                    for line in desc_lines:
                        if text_y < box_y + 5*mm:  # 防止文字超出框外
                            break
                        if line.strip():
                            canvas_obj.drawString(text_x, text_y, line)
                        text_y -= 4*mm  # 行間距
                
                print(f"文字框繪製完成 - 位置: ({box_x/mm:.1f}, {box_y/mm:.1f}), 尺寸: ({box_width/mm:.1f} x {box_height/mm:.1f})")
            
        except Exception as e:
            print(f"內page繪製錯誤: {e}")
            import traceback
            print(f"錯誤詳情: {traceback.format_exc()}")
    
    def draw_story_page(self, canvas_obj, story_data, story_image, template_config, font_name, layout, compress_enabled=False):
        """繪製故事page"""
        try:
            print(f"繪製故事page，數據: {story_data}")
            print(f"story_data 類型: {type(story_data)}")
            
            # 確保 story_data 是字典類型
            if not isinstance(story_data, dict):
                print(f"警告：story_data 不是字典類型（類型: {type(story_data)}), using default values")
                story_data = {"title": "小故事", "content": ""}
            
            title = story_data.get("title", "").strip() if story_data.get("title") else ""
            if not title:
                title = "小故事"  # 預設標題
            content = story_data.get("content", "").strip() if story_data.get("content") else ""
            print(f"故事標題: {title}, 內容長度: {len(content)}")
            
            if layout == "版型A-經典排版":
                # 版型A故事page：與封面完全相同的設計
                print(f"版型A故事page，圖片存在: {story_image is not None}")
                
                if story_image:
                    print(f"處理故事圖片，尺寸: {story_image.size}")
                    # 上方滿版圖片（與封面相同，佔據上半部）
                    try:
                        full_bleed_top = self.create_full_bleed_image(story_image, width=210*mm, height=200*mm)
                        import time
                        temp_path = f"temp_story_a_{int(time.time() * 1000000)}.jpg"
                        full_bleed_top.save(temp_path, "JPEG", quality=95)
                        
                        # 滿版圖片（上方）
                        canvas_obj.drawImage(temp_path, 0, 97*mm, width=210*mm, height=200*mm)
                        print("版型A故事page圖片繪製完成")
                        
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    except Exception as e:
                        print(f"版型A故事圖片處理錯誤: {e}")
                        # 使用純色背景作為備用
                        canvas_obj.setFillColor(HexColor(template_config["secondary"]))
                        canvas_obj.rect(0, 97*mm, 210*mm, 200*mm, fill=1, stroke=0)
                else:
                    print("版型A故事page：沒有圖片，使用純色背景")
                    # 沒有圖片時的上方背景
                    canvas_obj.setFillColor(HexColor(template_config["secondary"]))
                    canvas_obj.rect(0, 97*mm, 210*mm, 200*mm, fill=1, stroke=0)
                
                # 下方文字區域背景（與封面相同）
                canvas_obj.setFillColor(HexColor(template_config["background"]))
                canvas_obj.rect(0, 0, 210*mm, 97*mm, fill=1, stroke=0)
                
                # 進一步擴大半透明文字框空間
                canvas_obj.setFillColor(HexColor("#000000"))
                canvas_obj.setFillAlpha(0.7)
                canvas_obj.rect(3*mm, 3*mm, 204*mm, 91*mm, fill=1, stroke=0)  # 進一步擴大文字框
                canvas_obj.setFillAlpha(1)
                
                # 進一步縮小標題字體
                if title:
                    canvas_obj.setFont(font_name, 14)  # 進一步縮小標題字體
                    canvas_obj.setFillColor(HexColor(template_config["background"]))  # 使用template顏色
                    title_width = stringWidth(title, font_name, 14)
                    canvas_obj.drawString((210*mm - title_width) / 2, 80*mm, title)
                    print(f"版型A故事標題已繪製: {title}")
                
                # 適應性內文字體和版面
                if content:
                    # 動態計算可用寬度，根據文字長度調整
                    text_length = len(content)
                    if text_length > 300:
                        available_width = 200  # 長文字，保持最大寬度
                    elif text_length > 150:
                        available_width = 190  # 中等文字
                    else:
                        available_width = 180  # 短文字，稍微縮小寬度以更好居中
                    
                    available_height = 65  # 可用高度（mm）
                    adaptive_font_size = self.calculate_adaptive_font_size(content, available_width, 12, font_name, canvas_obj)
                    
                    canvas_obj.setFont(font_name, adaptive_font_size)
                    canvas_obj.setFillColor(HexColor(template_config["accent"]))  # 使用template顏色
                    lines = self.wrap_text(content, available_width, font_name, adaptive_font_size, canvas_obj)
                    
                    # 計算最佳行高和起始位置
                    line_height = max(adaptive_font_size + 2, 8)
                    total_text_height = len(lines) * line_height
                    start_y = min(70*mm, 70*mm - (total_text_height - available_height*mm) / 2)
                    
                    y_position = start_y
                    max_lines = int(available_height * mm / line_height)
                    
                    for line in lines[:max_lines]:  # 動態計算最大行數
                        if y_position < 6*mm:
                            break
                        if line.strip():
                            line_width = stringWidth(line, font_name, adaptive_font_size)
                            canvas_obj.drawString((210*mm - line_width) / 2, y_position, line)
                        y_position -= line_height
                    print(f"版型A故事內容已繪製，共{min(len(lines), max_lines)}行，使用字體大小: {adaptive_font_size}")
            
            elif layout == "版型B-藝術拼貼":
                # 版型B：滿版背景圖片 + 右下角文字（背景框，擴大文字範圍）
                if story_image:
                    full_bleed = self.create_full_bleed_image(story_image)
                    import time
                    temp_path = f"temp_story_b_{int(time.time() * 1000000)}.jpg"
                    full_bleed.save(temp_path, "JPEG", quality=95)
                    canvas_obj.drawImage(temp_path, 0, 0, width=210*mm, height=297*mm)
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                else:
                    canvas_obj.setFillColor(HexColor(template_config["secondary"]))
                    canvas_obj.rect(0, 0, 210*mm, 297*mm, fill=1, stroke=0)
                
                # 計算文字內容和適應性文字框範圍
                all_text_content = ""
                if title:
                    all_text_content += title
                if content:
                    all_text_content += content
                
                # 適應性計算文字框尺寸 - 改進版本，針對中文優化
                text_length = len(all_text_content)
                
                # 計算中文寬度係數
                width_factor = self.calculate_chinese_text_width_factor(all_text_content)
                
                # 根據文字長度和中文係數動態計算基礎寬度
                if text_length > 300:
                    base_width = int(160 * width_factor)  # 超長文字
                elif text_length > 180:
                    base_width = int(150 * width_factor)  # 長文字
                elif text_length > 100:
                    base_width = int(140 * width_factor)  # 中等文字
                else:
                    base_width = int(130 * width_factor)  # 短文字
                
                # 確保寬度在合理範圍內
                base_width = min(max(base_width, 120), 175)  # 120-175mm範圍
                
                adaptive_width = self.calculate_adaptive_layout_width(all_text_content, base_width)
                adaptive_font_size = self.calculate_adaptive_font_size(content, adaptive_width, 13, font_name, canvas_obj) if content else 13
                
                content_lines = []
                if content:
                    content_lines = self.wrap_text(content, adaptive_width, font_name, adaptive_font_size, canvas_obj)
                
                total_lines = 1 if title else 0  # 標題行
                total_lines += len(content_lines)
                
                # 創建適應性尺寸的透明背景框（右下角）
                if total_lines > 0:
                    line_height = adaptive_font_size + 2
                    
                    # 動態計算框高度，確保能容納所有文字
                    # 為中文文字預留更多垂直空間
                    title_space = 20*mm if title else 10*mm
                    padding_space = 16*mm  # 上下padding
                    text_area_height = total_lines * line_height + title_space + padding_space
                    
                    # 根據文字長度調整最小高度
                    if text_length > 200:
                        min_height = 90*mm  # 長文字需要更高的框
                    elif text_length > 100:
                        min_height = 75*mm  # 中等文字
                    else:
                        min_height = 60*mm  # 短文字
                    
                    box_height = max(text_area_height, min_height)
                    
                    # 確保不超過page面可用高度
                    max_allowed_height = 290*mm - 10*mm  # page面高度297mm - 上下邊距
                    if box_height > max_allowed_height:
                        box_height = max_allowed_height
                    
                    # 動態計算框寬度，針對中文增加padding
                    padding = int(20 * width_factor)*mm  # 根據中文比例調整padding
                    box_width = min(adaptive_width*mm + padding, 185*mm)  # 最大不超過185mm，給中文更多空間
                    
                    box_x = 210*mm - box_width - 5*mm
                    box_y = 5*mm
                    
                    # 確保文字框不會超出page面頂部
                    if box_y + box_height > 297*mm - 5*mm:
                        box_y = 297*mm - box_height - 5*mm
                    
                    # 繪製透明框，使用template顏色
                    canvas_obj.setFillColor(HexColor(template_config["primary"]))
                    canvas_obj.setFillAlpha(0.8)  # 80%透明度
                    canvas_obj.rect(box_x, box_y, box_width, box_height, fill=1, stroke=0)
                    canvas_obj.setFillAlpha(1)  # 重置透明度
                
                # 標題文字（使用template顏色）
                if title:
                    canvas_obj.setFont(font_name, 16)
                    canvas_obj.setFillColor(HexColor(template_config["background"]))
                    canvas_obj.drawString(box_x + 8*mm, box_y + box_height - 15*mm, title)
                
                # 內容文字（使用template顏色和適應性字體）
                if content_lines:
                    canvas_obj.setFont(font_name, adaptive_font_size)
                    canvas_obj.setFillColor(HexColor(template_config["accent"]))
                    
                    # 動態計算行高，根據字體大小調整
                    line_height = adaptive_font_size + 3  # 動態行高
                    
                    # 計算可用的文字區域高度
                    title_area_height = 20*mm if title else 10*mm  # 標題區域高度
                    padding_bottom = 10*mm  # 底部邊距（增加以確保不溢出）
                    available_text_height = box_height - title_area_height - padding_bottom
                    
                    # 計算最大可顯示行數（直接使用計算值）
                    max_lines = int(available_text_height / line_height)
                    if max_lines < 1:
                        max_lines = 1  # 至少顯示1行
                    
                    # 起始Y位置
                    start_y = box_y + box_height - title_area_height
                    y_position = start_y
                    
                    displayed_lines = 0
                    for line in content_lines:
                        # 檢查是否還有空間顯示下一行
                        next_y = y_position - line_height
                        if next_y < (box_y + padding_bottom):  # 確保不超出底部邊界
                            break
                        if displayed_lines >= max_lines:
                            break
                        if line.strip():
                            canvas_obj.drawString(box_x + 8*mm, y_position, line)
                            displayed_lines += 1
                        y_position -= line_height
                    
                    print(f"Layout B displayed {displayed_lines}/{len(content_lines)} lines of text, font size: {adaptive_font_size}, line height: {line_height}, box height: {box_height}")
            
            elif layout == "版型C-簡約現代":
                # 版型C故事page：與封面相同的設計（左圖右文）但適應性布局
                print(f"版型C故事page，圖片存在: {story_image is not None}")
                
                # 根據文字內容計算適當的版面寬度
                all_text_content = ""
                if title:
                    all_text_content += title
                if content:
                    all_text_content += content
                
                text_area_width = self.calculate_adaptive_layout_width(all_text_content, 70)
                image_area_width = 210 - text_area_width  # 剩餘空間給圖片
                
                # 背景色
                canvas_obj.setFillColor(HexColor(template_config["accent"]))
                canvas_obj.rect(0, 0, 210*mm, 297*mm, fill=1, stroke=0)
                
                # 左側圖片區域（適應性寬度）- 與封面相同的裁切方式
                if story_image:
                    # 目標區域：動態寬度 x 297mm (全高)
                    target_width = image_area_width
                    target_height = 297
                    target_ratio = target_width / target_height
                    
                    # 原圖比例
                    img_ratio = story_image.width / story_image.height
                    
                    if img_ratio > target_ratio:
                        # 圖片太寬，需要裁切寬度
                        new_height = story_image.height
                        new_width = int(new_height * target_ratio)
                        left = (story_image.width - new_width) // 2
                        cropped_image = story_image.crop((left, 0, left + new_width, new_height))
                    else:
                        # 圖片太高，需要裁切高度
                        new_width = story_image.width
                        new_height = int(new_width / target_ratio)
                        top = (story_image.height - new_height) // 2
                        cropped_image = story_image.crop((0, top, new_width, top + new_height))
                    
                    # 調整到目標尺寸
                    target_px_w = int(target_width * 300 / 25.4)  # mm轉像素
                    target_px_h = int(target_height * 300 / 25.4)
                    resized_image = cropped_image.resize((target_px_w, target_px_h), Image.Resampling.LANCZOS)
                    
                    import time
                    temp_path = f"temp_story_c_{int(time.time() * 1000000)}.jpg"
                    resized_image.save(temp_path, "JPEG", quality=95)
                    
                    canvas_obj.drawImage(temp_path, 0, 0, width=image_area_width*mm, height=297*mm)
                    print("版型C故事page圖片繪製完成")
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                else:
                    # 沒有圖片時的左側背景
                    canvas_obj.setFillColor(HexColor(template_config["secondary"]))
                    canvas_obj.rect(0, 0, image_area_width*mm, 297*mm, fill=1, stroke=0)
                
                # 右側文字區域（適應性寬度）
                canvas_obj.setFillColor(HexColor(template_config["background"]))
                canvas_obj.rect(image_area_width*mm, 0, text_area_width*mm, 297*mm, fill=1, stroke=0)
                
                # 取消標題所有文字內容（按要求移除標題）
                
                # 內容描述（水平，自動換行，適應性字體和版面）
                if content:
                    # 使用適應性字體大小和文字寬度
                    available_width = text_area_width - 10  # 留一些邊距
                    adaptive_font_size = self.calculate_adaptive_font_size(content, available_width, 12, font_name, canvas_obj)
                    
                    canvas_obj.setFont(font_name, adaptive_font_size)
                    canvas_obj.setFillColor(HexColor(template_config["text"]))
                    lines = self.wrap_text(content, available_width, font_name, adaptive_font_size, canvas_obj)
                    
                    # 計算最佳起始位置和行高
                    line_height = adaptive_font_size + 2
                    total_height = len(lines) * line_height
                    start_y = min(250*mm, 280*mm - total_height/2)  # 中心對齊
                    text_center_x = image_area_width + text_area_width / 2
                    
                    y_pos = start_y
                    max_lines = int((start_y - 20*mm) / line_height)  # 計算最大可顯示行數
                    
                    for line in lines[:max_lines]:  # 動態最大行數
                        if y_pos < 20*mm:  # 維持適當的底部邊界
                            break
                        if line.strip():
                            line_width = stringWidth(line, font_name, adaptive_font_size)
                            canvas_obj.drawString(text_center_x*mm - line_width/2, y_pos, line)  # 中心對齊
                        y_pos -= line_height
                
                print(f"版型C故事內容已繪製，標題: {title}")
            
        except Exception as e:
            print(f"故事page繪製錯誤: {e}")
    
    def draw_back_cover(self, canvas_obj, back_data, footer_images, template_config, font_name, layout, compress_enabled=False):
        """繪製結尾page精選回顧"""
        try:
            print(f"繪製結尾page，數據: {back_data}")
            print(f"back_data 類型: {type(back_data)}")
            
            # 確保 back_data 是字典類型
            if not isinstance(back_data, dict):
                print(f"警告：back_data 不是字典類型，而是 {type(back_data)}，使用預設值")
                back_data = {"title": "精選回顧", "description": ""}
            
            title = back_data.get("title", "").strip() if back_data.get("title") else ""
            if not title:
                title = "精選回顧"  # 預設標題
            description = back_data.get("description", "").strip() if back_data.get("description") else ""
            print(f"結尾page標題: {title}, 描述: {description}")
            print(f"可用圖片數量: {len(footer_images)}")
            
            if layout == "版型A-經典排版" or layout == "版型B-藝術拼貼":
                # 版型A和B：使用可用圖片創建拼貼效果（統一格式）
                if footer_images and len(footer_images) >= 1:  # 只要有至少一images就創建拼貼
                    # 創建2x2拼接的滿版底圖
                    canvas_width = 210*mm  # A4寬度
                    canvas_height = 297*mm  # A4高度
                    
                    # 每個象限的大小
                    quad_width = canvas_width / 2
                    quad_height = canvas_height / 2
                    
                    positions = [
                        (0, quad_height, quad_width, quad_height),        # 左上
                        (quad_width, quad_height, quad_width, quad_height), # 右上
                        (0, 0, quad_width, quad_height),                  # 左下
                        (quad_width, 0, quad_width, quad_height)          # 右下
                    ]
                    
                    # 用可用圖片創建拼貼，不足四張則重複使用
                    for i in range(4):  # 始終創建4個象限的拼貼
                        x, y, w, h = positions[i]
                        # 如果圖片不足，循環使用可用圖片
                        img_index = i % len(footer_images)
                        original_img = footer_images[img_index]
                        
                        # 等比縮放並裁切以填滿象限
                        target_ratio = w / h
                        img_ratio = original_img.width / original_img.height
                        
                        if img_ratio > target_ratio:
                            # 圖片太寬，以高度為準縮放後裁切寬度
                            new_height = original_img.height
                            new_width = int(new_height * target_ratio)
                            left = (original_img.width - new_width) // 2
                            cropped = original_img.crop((left, 0, left + new_width, new_height))
                        else:
                            # 圖片太高，以寬度為準縮放後裁切高度
                            new_width = original_img.width
                            new_height = int(new_width / target_ratio)
                            top = (original_img.height - new_height) // 2
                            cropped = original_img.crop((0, top, new_width, top + new_height))
                        
                        import time
                        temp_path = f"temp_back_{i}_{int(time.time() * 1000000)}.jpg"
                        cropped.save(temp_path, "JPEG", quality=95)
                        
                        canvas_obj.drawImage(temp_path, x, y, width=w, height=h)
                        
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                else:
                    # 沒有圖片時的純色背景（保持為備用方案）
                    canvas_obj.setFillColor(HexColor(template_config["background"]))
                    canvas_obj.rect(0, 0, 210*mm, 297*mm, fill=1, stroke=0)
                
                # 精選回顧：去除title，改為右下角小白字+透明背景
                if description:
                    # 計算文字行數和所需空間
                    lines = self.wrap_text(description, 100, font_name, 12, canvas_obj)
                    box_height = len(lines) * 14 + 8*mm
                    box_width = 110*mm
                    box_x = 210*mm - box_width - 5*mm
                    box_y = 5*mm
                    
                    # 透明背景框
                    canvas_obj.setFillColor(HexColor("#000000"))
                    canvas_obj.setFillAlpha(0.6)
                    canvas_obj.rect(box_x, box_y, box_width, box_height, fill=1, stroke=0)
                    canvas_obj.setFillAlpha(1)
                    
                    # 小的白色文字
                    canvas_obj.setFont(font_name, 12)
                    canvas_obj.setFillColor(HexColor("#FFFFFF"))
                    y_position = box_y + box_height - 6*mm
                    line_height = 14
                    
                    for line in lines:
                        if y_position < box_y + 3*mm:
                            break
                        if line.strip():
                            canvas_obj.drawString(box_x + 5*mm, y_position, line)
                        y_position -= line_height
            
            elif layout == "版型C-簡約現代":
                # 版型C：一張底圖 + 無背景框文字
                print(f"版型C尾page，圖片數量: {len(footer_images) if footer_images else 0}")
                if footer_images and len(footer_images) >= 1:
                    # 使用Page一images作為滿版底圖
                    first_image = footer_images[0]
                    full_bleed = self.create_full_bleed_image(first_image)
                    import time
                    temp_path = f"temp_back_c_{int(time.time() * 1000000)}.jpg"
                    full_bleed.save(temp_path, "JPEG", quality=95)
                    
                    canvas_obj.drawImage(temp_path, 0, 0, width=210*mm, height=297*mm)
                    print("版型C尾page底圖已繪製")
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                else:
                    print("版型C尾page：沒有圖片，使用純色背景")
                    # 沒有圖片時的純色背景
                    canvas_obj.setFillColor(HexColor(template_config["accent"]))
                    canvas_obj.rect(0, 0, 210*mm, 297*mm, fill=1, stroke=0)
                
                # 右下角文字說明（縮小文字，擴大文字框範圍，套用template顏色）
                if description:
                    # 擴大的透明文字框
                    lines = self.wrap_text(description, 120, font_name, 14, canvas_obj)
                    box_height = len(lines) * 16 + 20*mm  # 擴大框範圍
                    box_width = 140*mm  # 擴大寬度
                    box_x = 210*mm - box_width - 10*mm
                    box_y = 10*mm
                    
                    # 繪製透明框，使用template顏色
                    canvas_obj.setFillColor(HexColor(template_config["secondary"]))
                    canvas_obj.setFillAlpha(0.7)
                    canvas_obj.rect(box_x, box_y, box_width, box_height, fill=1, stroke=0)
                    canvas_obj.setFillAlpha(1)
                    
                    # 縮小文字，套用template顏色
                    canvas_obj.setFont(font_name, 14)  # 縮小字體
                    canvas_obj.setFillColor(HexColor(template_config["background"]))
                    y_position = box_y + box_height - 12*mm
                    line_height = 16
                    
                    for line in lines[:8]:  # 增加顯示行數
                        if y_position < box_y + 8*mm:
                            break
                        if line.strip():
                            canvas_obj.drawString(box_x + 8*mm, y_position, line)
                        y_position -= line_height
            
        except Exception as e:
            print(f"page尾繪製錯誤: {e}")
    
    def crop_image_for_layout(self, image, aspect_ratio, crop_position="center"):
        """裁切圖片為指定比例"""
        if aspect_ratio == "16:9":
            target_ratio = 16 / 9
        elif aspect_ratio == "4:3":
            target_ratio = 4 / 3
        elif aspect_ratio == "1:1":
            target_ratio = 1
        else:
            target_ratio = 16 / 9  # 預設值
        
        # 計算裁切區域
        img_ratio = image.width / image.height
        
        if img_ratio > target_ratio:
            # 圖片太寬，裁切寬度
            new_width = int(image.height * target_ratio)
            left = (image.width - new_width) // 2
            cropped = image.crop((left, 0, left + new_width, image.height))
        else:
            # 圖片太高，裁切高度
            new_height = int(image.width / target_ratio)
            if crop_position == "top":
                # 從上方裁切
                top = 0
            else:
                # 預設從中心裁切
                top = (image.height - new_height) // 2
            cropped = image.crop((0, top, image.width, top + new_height))
        
        return cropped

    def get_image_quality(self, compress_enabled):
        """根據壓縮設定獲取圖片品質"""
        return 60 if compress_enabled else 95
    
    def allocate_images_smartly(self, pil_images, content_pages_count):
        """智能分配圖片到各pages"""
        total_images = len(pil_images)
        total_pages_needed = 1 + content_pages_count + 1 + 1  # 封面 + 內page + 故事page + 尾page
        
        allocation = {
            "cover": None,
            "pages": [],
            "story": None,
            "footer": []
        }
        
        if total_images == 0:
            print("警告：沒有圖片可用")
            return allocation
        
        if total_images >= total_pages_needed:
            # 圖片足夠：按順序分配
            allocation["cover"] = 0
            allocation["pages"] = list(range(1, content_pages_count + 1))
            allocation["story"] = content_pages_count + 1
            remaining_start = content_pages_count + 2
            allocation["footer"] = list(range(remaining_start, min(remaining_start + 5, total_images)))
            
        elif total_images >= content_pages_count + 1:
            # 圖片稍少：優先內容page，盡量避免重複
            allocation["cover"] = 0
            allocation["pages"] = list(range(1, min(content_pages_count + 1, total_images)))
            # 故事page使用下一張可用圖片，如果沒有則使用最後一張
            if content_pages_count + 1 < total_images:
                allocation["story"] = content_pages_count + 1
            else:
                allocation["story"] = total_images - 1  # 使用最後一images
            # 使用所有可用圖片作為footer
            allocation["footer"] = list(range(min(5, total_images)))
            
        else:
            # 圖片很少：大量重複使用，但避免故事page使用封面圖片
            allocation["cover"] = 0
            # 循環Using image填滿內容page
            allocation["pages"] = [i % total_images for i in range(content_pages_count)]
            # 故事page優先使用最後一images，如果只有一張圖則使用同一張
            if total_images > 1:
                allocation["story"] = total_images - 1  # 使用最後一images
            else:
                allocation["story"] = 0  # 只有一images時無選擇
            # footer使用所有圖片
            allocation["footer"] = list(range(total_images))
        
        print(f"Image allocation result: cover={allocation['cover']}, pages={allocation['pages']}, story={allocation['story']}, footer={allocation['footer']}")
        return allocation

    def make_photo_magazine(self, images, json_data, template, layout, font, compress_pdf, disable_cover_layout, output_path):
        """製作寫真雜誌"""
        # 此節點是唯一需要 reportlab 的地方，執行時才載入（必要時自動安裝）
        try:
            ensure_reportlab()
        except RuntimeError as e:
            print(f"PhotoMagazineMaker: {e}")
            return (str(e),)

        try:
            # 根據ComfyUI規範，當INPUT_IS_LIST=True時，所有參數都是列表
            # 從列表中提取實際值
            json_string = json_data[0] if isinstance(json_data, list) and json_data else "{}"
            template_name = template[0] if isinstance(template, list) and template else "清新自然"
            layout_name = layout[0] if isinstance(layout, list) and layout else "版型A-經典排版"
            font_name_input = font[0] if isinstance(font, list) and font else "default"
            compress_enabled = compress_pdf[0] if isinstance(compress_pdf, list) and compress_pdf else False
            disable_cover = disable_cover_layout[0] if isinstance(disable_cover_layout, list) and disable_cover_layout else False
            base_output = output_path[0] if isinstance(output_path, list) and output_path else "./ComfyUI/output/MyPDF/photo_magazine.pdf"
            
            # 為檔案名添加時間戳記，防止覆蓋
            import os.path
            file_dir = os.path.dirname(base_output)
            file_name = os.path.basename(base_output)
            name, ext = os.path.splitext(file_name)
            import time
            timestamp = str(int(time.time()))
            output_file = os.path.join(file_dir, f"{name}_{timestamp}{ext}")
            
            # 解析JSON資料
            try:
                magazine_data = json.loads(json_string)
            except json.JSONDecodeError as e:
                return (f"JSON parse error: {str(e)}, received data: {str(json_string)[:200]}",)
            
            # 驗證JSON結構
            if not isinstance(magazine_data, dict):
                return (f"JSON format error: expected dict, received {type(magazine_data).__name__}",)
            
            # 檢查必要的數據結構
            required_keys = ["magazine_info", "cover", "pages", "story_page", "back_cover"]
            missing_keys = [key for key in required_keys if key not in magazine_data]
            if missing_keys:
                print(f"警告：缺少以下鍵值: {missing_keys}")
            
            # 檢查pages是否為列表且不為空
            pages = magazine_data.get("pages", [])
            if not isinstance(pages, list):
                return (f"JSON format error: pages must be list, received {type(pages).__name__}",)
            
            if len(pages) == 0:
                return ("JSON data error: pages list is empty, cannot generate magazine",)
            
            print(f"Successfully parsed JSON, contains {len(pages)} pages")
            
            template_config = self.template_configs.get(template_name, self.template_configs["清新自然"])
            
            # 註冊字體
            font_name = self.register_font(font_name_input)
            
            # Converting image
            pil_images = []
            print(f"Starting image conversion, image type: {type(images)}")
            
            if isinstance(images, list):
                # 處理圖片列表
                for i, img in enumerate(images):
                    if isinstance(img, torch.Tensor):
                        pil_img = self.tensor_to_pil(img)
                        pil_images.append(pil_img)
                        print(f"Converting image {i}: {pil_img.size}")
            elif isinstance(images, torch.Tensor):
                if len(images.shape) == 4:  # 批次圖片
                    for i in range(images.shape[0]):
                        pil_img = self.tensor_to_pil(images[i])
                        pil_images.append(pil_img)
                        print(f"Converting batch image {i}: {pil_img.size}")
                else:
                    pil_img = self.tensor_to_pil(images)
                    pil_images.append(pil_img)
                    print(f"Converting single image: {pil_img.size}")
            
            if len(pil_images) == 0:
                return ("Error: No valid images to process",)
            
            print(f"Successfully converted {len(pil_images)} images")
            
            # 確保輸出目錄存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 創建PDF
            c = canvas.Canvas(output_file, pagesize=A4)
            
            # 動態計算所需page面數
            pages = magazine_data.get("pages", [])
            total_pages_needed = 1 + len(pages) + 1 + 1  # 封面 + 內page + 故事page + 尾page
            print(f"Total required {total_pages_needed} pages, available images {len(pil_images)} 張")
            
            # 智能圖片分配策略
            image_allocation = self.allocate_images_smartly(pil_images, len(pages))
            print(f"Image allocation strategy: {image_allocation}")
            
            # 計算圖片分配
            image_index = 0
            used_image_indices = set()  # 追蹤已使用的圖片索引
            
            # 封面 - 使用智能分配
            cover_image = None
            if image_allocation["cover"] is not None:
                cover_image = pil_images[image_allocation["cover"]]
                print(f"Using image {image_allocation['cover']} as cover")
            else:
                print("Warning: No image available for cover")
            
            cover_data = magazine_data.get("cover", {})
            # 確保 cover_data 是字典類型
            if not isinstance(cover_data, dict):
                print(f"Warning: cover in JSON is not dict format (type: {type(cover_data)}), using default values")
                cover_data = {"title": "寫真集", "subtitle": "", "description": ""}
            elif not cover_data:
                print("Warning: Missing cover data in JSON, using default values")
                cover_data = {"title": "寫真集", "subtitle": "", "description": ""}
            
            # 檢查是否關閉封面排版
            if disable_cover:
                # 關閉封面排版：使用Page一images作為滿版封面（不含文字）
                print("✓ Cover layout disabled, using first image as full bleed cover")
                if cover_image:
                    # 創建滿版圖片
                    full_bleed = self.create_full_bleed_image(cover_image, 210*mm, 297*mm)
                    # 將 PIL Image 轉換為 ImageReader
                    from reportlab.lib.utils import ImageReader
                    img_reader = ImageReader(full_bleed)
                    c.drawImage(img_reader, 0, 0, width=210*mm, height=297*mm, preserveAspectRatio=False)
                else:
                    # 沒有圖片時使用純色背景
                    c.setFillColor(HexColor(template_config["primary"]))
                    c.rect(0, 0, 210*mm, 297*mm, fill=1, stroke=0)
            else:
                # 正常封面排版
                self.draw_cover_page(c, magazine_data, cover_image, template_config, font_name, layout_name, compress_enabled)
            
            c.showPage()
            print("Cover page complete")
            
            # 內page - 使用智能分配
            print(f"JSON contains {len(pages)} page data, preparing to draw all content pages")
            
            for i, page_data in enumerate(pages):
                # 使用智能分配的圖片索引
                if i < len(image_allocation["pages"]):
                    page_img_idx = image_allocation["pages"][i]
                    page_image = pil_images[page_img_idx]
                    print(f"Using image {page_img_idx} as page {i+1} page")
                else:
                    # 如果分配策略沒有足夠的圖片，使用Page一images
                    page_image = pil_images[0] if len(pil_images) > 0 else None
                    print(f"Page {i+1} using fallback image 0")
                
                # 驗證page面數據
                if not isinstance(page_data, dict):
                    print(f"Warning: Page {i+1} data format error, skipping")
                    continue
                
                self.draw_content_page(c, page_data, page_image, template_config, font_name, layout_name, compress_enabled)
                c.showPage()
                print(f"Page {i+1} page complete")
            
            # 故事page - 使用智能分配
            story_data = magazine_data.get("story_page", {})
            # 確保 story_data 是字典類型
            if not isinstance(story_data, dict):
                print(f"Warning: story_page in JSON is not dict format (type: {type(story_data)}), using default values")
                story_data = {"title": "小故事", "content": ""}
            elif not story_data:
                print("Warning: Missing story_page data in JSON, using default values")
                story_data = {"title": "小故事", "content": ""}
            
            story_image = None
            if image_allocation["story"] is not None:
                story_image = pil_images[image_allocation["story"]]
                print(f"Using image {image_allocation['story']} as story page")
            else:
                print("警告：故事page沒有對應圖片")
            
            self.draw_story_page(c, story_data, story_image, template_config, font_name, layout_name, compress_enabled)
            c.showPage()
            print("Story page complete")
            
            # page尾圖片準備 - 使用智能分配
            footer_images = []
            print(f"準備{layout_name}的page尾圖片")
            
            for idx in image_allocation["footer"]:
                if idx < len(pil_images):
                    footer_images.append(pil_images[idx])
                    print(f"Using image {idx} 作為page尾圖片")
            
            print(f"page尾圖片準備完成，共 {len(footer_images)} 張")
            
            # 尾page（所有版型都需要尾page）
            back_data = magazine_data.get("back_cover", {})
            # 確保 back_data 是字典類型
            if not isinstance(back_data, dict):
                print(f"警告：JSON中的back_cover不是字典格式（類型: {type(back_data)}), using default values")
                back_data = {"title": "精選回顧", "description": ""}
            elif not back_data:
                print("警告：JSON中缺少back_cover數據，使用預設值")
                back_data = {"title": "精選回顧", "description": ""}
            
            print(f"準備繪製{layout_name}尾page")
            self.draw_back_cover(c, back_data, footer_images, template_config, font_name, layout_name, compress_enabled)
            c.showPage()
            print("Back cover complete")
            
            # 儲存PDF
            c.save()
            
            # PDF壓縮處理已透過較低的圖片品質完成
            
            return (f"寫真雜誌生成成功！已儲存至：{output_file}",)
            
        except Exception as e:
            return (f"Error generating photo magazine:{str(e)}",)


# 節點註冊
NODE_CLASS_MAPPINGS = {
    "PhotoMagazinePromptGenerator": PhotoMagazinePromptGenerator,
    "PhotoMagazineParser": PhotoMagazineParser,
    "PhotoMagazineMaker": PhotoMagazineMaker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoMagazinePromptGenerator": "📝 寫真雜誌prompt注入器",
    "PhotoMagazineParser": "🔍 JSON → prompt列表",
    "PhotoMagazineMaker": "📄 寫真雜誌製作器"
}
