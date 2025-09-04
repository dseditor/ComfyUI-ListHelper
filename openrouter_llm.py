import requests
import json
import base64
import os
import io
import re
import torch
import random
from PIL import Image, ImageOps
import numpy as np
import math

class OpenRouterLLM:
    """
    OpenRouter LLM節點，用於處理OpenRouter的LLM類型
    支援圖片和文字輸入輸出，API金鑰管理，動態模型管理
    """
    
    @classmethod
    def _load_models(cls):
        """載入模型列表從models.json檔案"""
        models_file = os.path.join(os.path.dirname(__file__), "models.json")
        try:
            with open(models_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('models', [])
        except Exception as e:
            print(f"⚠️ 載入models.json失敗: {e}")
            # 返回默認模型列表
            return [
                "google/gemini-2.5-flash-image-preview:free",
                "google/gemma-3-27b-it:free",
                "qwen/qwen2.5-vl-32b-instruct:free",
                "mistralai/mistral-small-3.2-24b-instruct:free",
                "meta-llama/llama-4-maverick:free",
                "openai/gpt-oss-20b:free",
                "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
                "moonshotai/kimi-dev-72b:free",
                "qwen/qwen2.5-vl-72b-instruct:free",
                "moonshotai/kimi-k2:free"
            ]
    
    @classmethod
    def _save_models(cls, models_list):
        """保存模型列表到models.json檔案"""
        models_file = os.path.join(os.path.dirname(__file__), "models.json")
        try:
            with open(models_file, 'w', encoding='utf-8') as f:
                json.dump({"models": models_list}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存models.json失敗: {e}")
    
    def _add_custom_model(self, model_name):
        """添加自定義模型到models.json檔案"""
        try:
            current_models = self._load_models()
            
            # 檢查模型是否已存在
            if model_name not in current_models:
                current_models.append(model_name)
                self._save_models(current_models)
                print(f"✅ 已添加新模型: {model_name}")
            else:
                print(f"ℹ️ 模型已存在: {model_name}")
        except Exception as e:
            print(f"❌ 添加自定義模型失敗: {e}")
    
    @classmethod
    def INPUT_TYPES(cls):
        # 從JSON檔案載入模型列表
        text_models = cls._load_models()
        
        # 在列表末尾添加"Add Custom Model..."選項
        text_models_with_add = text_models + ["Add Custom Model..."]
        
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "輸入您的OpenRouter API金鑰"}),
                "user_prompt": ("STRING", {"multiline": True, "default": "Please analyze the provided content."}),
                "text_model": (text_models_with_add, {"default": text_models_with_add[0]}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1, "tooltip": "隨機種子控制(-1為隨機)。注意：圖像生成模型(如Gemini)不支援seed參數"}),
            },
            "optional": {
                "custom_model": ("STRING", {"multiline": False, "default": "", "placeholder": "輸入自定義模型名稱 (選擇Add Custom Model...時使用)"}),
                "system_prompt": ("STRING",),
                "enable_resize": ("BOOLEAN", {"default": False, "tooltip": "啟用圖像尺寸調整功能"}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8, "tooltip": "需要啟用resize才會生效"}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8, "tooltip": "需要啟用resize才會生效"}),
                "resize_method": (["bicubic", "lanczos", "bilinear", "nearest"], {"default": "lanczos", "tooltip": "需要啟用resize才會生效"}),
                "image_input_1": ("IMAGE",),
                "image_input_2": ("IMAGE",),
                "image_input_3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_output", "text_output")
    FUNCTION = "process_llm"
    CATEGORY = "ListHelper"
    
    def __init__(self):
        self.config_file = "config.json"
    
    def _is_image_generation_model(self, model_name):
        """檢查是否為圖像生成模型"""
        image_generation_models = [
            "gemini-2.5-flash-image-preview",
            "gemini-image",  # 可能的其他變體
        ]
        return any(img_model in model_name.lower() for img_model in image_generation_models)
    
    def _get_default_tensor_size(self, enable_resize, target_width, target_height):
        """獲取默認tensor尺寸"""
        if enable_resize:
            return (1, target_height, target_width, 3)
        else:
            return (1, 512, 512, 3)
    
    def _create_blank_image_tensor(self, width, height):
        """創建指定尺寸的空白圖像tensor（白色背景）"""
        # 創建白色圖像 (1.0 = 白色 in 0-1 range)
        blank_tensor = torch.ones(1, height, width, 3, dtype=torch.float32)
        return blank_tensor
        
    def _load_config(self):
        """載入配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"載入配置文件失敗: {e}")
        return {}
    
    def _save_config(self, config):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置文件失敗: {e}")
    
    def _get_api_key(self, api_key_input):
        """獲取API金鑰"""
        config = self._load_config()
        
        # 如果輸入了新的API金鑰，則更新配置
        if api_key_input.strip():
            config['openrouter_api_key'] = api_key_input.strip()
            self._save_config(config)
            return api_key_input.strip()
        
        # 否則使用配置文件中的金鑰
        return config.get('openrouter_api_key', '')
    
    def _tensor_to_base64(self, tensor, enable_resize=False, target_width=None, target_height=None):
        """將ComfyUI圖像tensor轉換為base64編碼，並可選擇性調整尺寸"""
        # tensor shape: [H, W, C] (0-1 range)
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 移除批次維度
        
        # 轉換為numpy並調整範圍到0-255
        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # 轉換為PIL圖像
        pil_image = Image.fromarray(numpy_image)
        
        # 只有在啟用resize且指定了目標尺寸時，才進行padding處理
        if enable_resize and target_width is not None and target_height is not None:
            pil_image = self._pad_and_resize_image(pil_image, target_width, target_height)
        
        # 轉換為base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    
    def _base64_to_tensor(self, base64_str, enable_resize=False, target_width=None, target_height=None, resize_method="lanczos"):
        """將base64編碼轉換為ComfyUI圖像tensor，並可選擇性調整尺寸"""
        try:
            # 移除各種可能的前綴
            prefixes = [
                'data:image/png;base64,',
                'data:image/jpeg;base64,',
                'data:image/jpg;base64,',
            ]
            
            original_str = base64_str
            for prefix in prefixes:
                if base64_str.startswith(prefix):
                    base64_str = base64_str[len(prefix):]
                    break
            
            # 清理base64字符串（移除可能的換行符和空格）
            base64_str = re.sub(r'[\n\r\s]', '', base64_str)
            
            # 解碼base64
            image_bytes = base64.b64decode(base64_str)
            
            # 轉換為PIL圖像
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # 確保是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 只有在啟用resize且指定了目標尺寸時，才進行縮放處理
            if enable_resize and target_width is not None and target_height is not None:
                pil_image = self._resize_image(pil_image, target_width, target_height, resize_method)
            
            # 轉換為numpy數組
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            
            # 轉換為tensor並添加批次維度 [1, H, W, C]
            tensor = torch.from_numpy(numpy_image).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            print(f"❌ base64轉tensor失敗: {e}")
            # 返回一個較大的默認圖像而不是1x1
            return torch.zeros(1, 512, 512, 3)
    
    def _url_to_tensor(self, image_url, enable_resize=False, target_width=None, target_height=None, resize_method="lanczos"):
        """從URL下載圖像並轉換為ComfyUI圖像tensor，並可選擇性調整尺寸"""
        try:
            # 下載圖像
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 轉換為PIL圖像
            pil_image = Image.open(io.BytesIO(response.content))
            
            # 確保是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 只有在啟用resize且指定了目標尺寸時，才進行縮放處理
            if enable_resize and target_width is not None and target_height is not None:
                pil_image = self._resize_image(pil_image, target_width, target_height, resize_method)
            
            # 轉換為numpy數組
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            
            # 轉換為tensor並添加批次維度 [1, H, W, C]
            tensor = torch.from_numpy(numpy_image).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            print(f"❌ 從URL載入圖像失敗: {e}")
            # 返回一個較大的默認圖像而不是1x1
            return torch.zeros(1, 512, 512, 3)
    
    def _call_openrouter_api(self, api_key, model, messages, seed=None):
        """調用OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "",
            "X-Title": "ComfyUI"
        }
        
        data = {
            "model": model,
            "messages": messages
        }
        
        # 檢查是否為圖像生成模型
        is_image_generation_model = self._is_image_generation_model(model)
        
        # 對於圖像生成模型，添加特殊參數
        if is_image_generation_model:
            # 確保包含image輸出模式
            data["modalities"] = ["image", "text"]
            # 不設置max_tokens，讓模型有足夠空間生成圖像
            
        # 只對非圖像生成模型設置seed參數
        if seed is not None and not is_image_generation_model:
            data["seed"] = seed
        
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            elif response.status_code == 429:
                # Rate limit錯誤
                try:
                    error_data = response.json()
                    return {"error": {"code": 429, "message": error_data.get("error", {}).get("message", "Rate limit exceeded")}}
                except:
                    return {"error": {"code": 429, "message": "Rate limit exceeded"}}
            else:
                try:
                    error_data = response.json()
                    return {"error": error_data.get("error", {"message": f"HTTP {response.status_code}"})}
                except:
                    return {"error": {"message": f"HTTP {response.status_code}: {response.text[:200]}"}}
                
        except Exception as e:
            print(f"❌ API調用異常: {e}")
            return None
    
    def _pad_and_resize_image(self, pil_image, target_width, target_height):
        """使用比例縮放+padding的方式處理圖像到指定尺寸"""
        try:
            original_width, original_height = pil_image.size
            target_ratio = target_width / target_height
            original_ratio = original_width / original_height
            
            # 先根據比例關係決定如何縮放
            if original_ratio > target_ratio:
                # 原圖較寬，以寬度為準縮放
                scale_ratio = target_width / original_width
                new_width = target_width
                new_height = int(original_height * scale_ratio)
            else:
                # 原圖較高，以高度為準縮放
                scale_ratio = target_height / original_height
                new_width = int(original_width * scale_ratio)
                new_height = target_height
            
            # 縮放圖像
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # 計算需要的padding
            pad_width = target_width - new_width
            pad_height = target_height - new_height
            
            # 應用padding（居中）
            if pad_width > 0 or pad_height > 0:
                left_pad = pad_width // 2
                right_pad = pad_width - left_pad
                top_pad = pad_height // 2
                bottom_pad = pad_height - top_pad
                
                # 使用黑色填充padding區域
                pil_image = ImageOps.expand(pil_image, (left_pad, top_pad, right_pad, bottom_pad), fill=(0, 0, 0))
            
            # 檢查是否需要進一步縮放（如果任一邊超過1536）
            current_width, current_height = pil_image.size
            if current_width > 1536 or current_height > 1536:
                # 保持比例縮放到1536以內
                final_scale_ratio = min(1536 / current_width, 1536 / current_height)
                final_width = int(current_width * final_scale_ratio)
                final_height = int(current_height * final_scale_ratio)
                pil_image = pil_image.resize((final_width, final_height), Image.LANCZOS)
            
            return pil_image
            
        except Exception as e:
            print(f"❌ 圖像padding處理失敗: {e}")
            return pil_image
    
    def _resize_image(self, pil_image, target_width, target_height, resize_method="lanczos"):
        """使用指定方法縮放圖像到目標尺寸"""
        try:
            # 選擇縮放方法
            method_map = {
                "bicubic": Image.BICUBIC,
                "lanczos": Image.LANCZOS,
                "bilinear": Image.BILINEAR,
                "nearest": Image.NEAREST
            }
            
            resize_filter = method_map.get(resize_method, Image.LANCZOS)
            return pil_image.resize((target_width, target_height), resize_filter)
            
        except Exception as e:
            print(f"❌ 圖像縮放失敗: {e}")
            return pil_image

    def process_llm(self, api_key, user_prompt, text_model, seed=-1, custom_model="", system_prompt=None, 
                   enable_resize=False, target_width=512, target_height=512, resize_method="lanczos",
                   image_input_1=None, image_input_2=None, image_input_3=None):
        """處理LLM請求"""
        
        # 獲取API金鑰
        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            # 使用默認尺寸512x512當enable_resize關閉時
            default_height = target_height if enable_resize else 512
            default_width = target_width if enable_resize else 512
            return (torch.zeros(*self._get_default_tensor_size(enable_resize, target_width, target_height)), "❌ 錯誤: 請提供OpenRouter API金鑰")
        
        # 處理種子設定 - Control After Generate支援
        if seed == -1:
            # 隨機種子
            actual_seed = random.randint(0, 2147483647)
        else:
            # 固定種子
            actual_seed = seed
        
        # 處理模型選擇
        if text_model == "Add Custom Model...":
            if not custom_model or not custom_model.strip():
                return (torch.zeros(1, target_height, target_width, 3), "❌ 請在custom_model欄位輸入自定義模型名稱")
            
            # 使用自定義模型
            selected_model = custom_model.strip()
            
            # 將新模型添加到models.json中
            self._add_custom_model(selected_model)
        else:
            # 使用選擇的預設模型
            selected_model = text_model
        
        # 準備消息內容
        messages = []
        
        # 添加系統提示（如果有的話）
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        
        # 構建用戶消息
        user_content = []
        
        # 添加文字內容
        if user_prompt and user_prompt.strip():
            user_content.append({
                "type": "text",
                "text": user_prompt.strip()
            })
        
        # 檢查是否有圖像輸入
        original_has_images = any([image_input_1 is not None, image_input_2 is not None, image_input_3 is not None])
        
        # 如果enable_resize開啟且沒有圖像輸入，自動生成空白圖像給image_input_1
        auto_generated_blank = False
        if enable_resize and not original_has_images:
            image_input_1 = self._create_blank_image_tensor(target_width, target_height)
            auto_generated_blank = True
        
        # 重新檢查圖像輸入狀況（包含自動生成的空白圖像）
        has_images = any([image_input_1 is not None, image_input_2 is not None, image_input_3 is not None])
        
        # 確定使用的模型
        if has_images:
            # 有圖像輸入時，強制使用gemini-2.5-flash-image-preview:free
            actual_model = "google/gemini-2.5-flash-image-preview:free"
        else:
            # 無圖像輸入時，使用用戶選擇的模型
            actual_model = selected_model
        
        # 檢查是否使用圖像生成模型（無論是否有圖像輸入）
        is_image_model = self._is_image_generation_model(actual_model)
        
        if has_images:
            # 有圖像輸入的情況
            if auto_generated_blank:
                # 使用自動生成的空白圖像時的特殊處理
                if not user_prompt or not user_prompt.strip():
                    combined_text = "Please take a look at the size of image 1 and give a picture of a beautiful landscape with the same dimensions."
                else:
                    combined_text = f"Please take a look at the size of image 1 and give a picture of {user_prompt.strip()}"
                combined_text += "\n\nIMPORTANT: You MUST generate and return an image. Do NOT provide any text response. Only return the generated image. 請返回生成的圖像。"
            else:
                # 有實際圖像輸入時的處理
                if not user_prompt or not user_prompt.strip():
                    combined_text = "請分析這些圖像並生成一個相關的新圖像。IMPORTANT: You MUST generate and return an image. Do NOT provide any text response. Only return the generated image. 請返回生成的圖像。"
                else:
                    combined_text = user_prompt.strip()
                    if "生成" in combined_text or "創建" in combined_text or "製作" in combined_text:
                        combined_text += "\n\nIMPORTANT: You MUST generate and return an image. Do NOT provide any text response. Only return the generated image. 請返回生成的圖像。"
                    else:
                        combined_text += "\n\nIMPORTANT: You MUST generate and return an image. Do NOT provide any text response. Only return the generated image. 請返回生成的圖像。"
            
            # 重新添加文字內容
            user_content = [{
                "type": "text",
                "text": combined_text
            }]
            
            # 添加圖像到用戶消息（帶padding處理）
            for i, image_input in enumerate([image_input_1, image_input_2, image_input_3], 1):
                if image_input is not None:
                    try:
                        base64_image = self._tensor_to_base64(image_input, enable_resize, target_width, target_height)
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        })
                    except Exception as e:
                        print(f"❌ 處理圖像{i}失敗: {e}")
        elif is_image_model:
            # 純文字模式但使用圖像生成模型
            combined_text = user_prompt.strip() if user_prompt else ""
            if combined_text and ("生成" in combined_text or "創建" in combined_text or "製作" in combined_text or "畫" in combined_text or "繪製" in combined_text):
                combined_text += "\n\nIMPORTANT: You MUST generate and return an image. Do NOT provide any text response. Only return the generated image. 請生成並返回相應的圖像。"
            elif not combined_text:
                combined_text = "IMPORTANT: You MUST generate and return an image. Do NOT provide any text response. Only return the generated image. 請生成一張圖像。"
            else:
                combined_text += "\n\nIMPORTANT: You MUST generate and return an image. Do NOT provide any text response. Only return the generated image."
            
            # 添加文字內容
            user_content.append({
                "type": "text",
                "text": combined_text
            })
        
        # 添加用戶消息
        if len(user_content) > 1:
            # 多媒體內容（包含圖像）
            messages.append({
                "role": "user", 
                "content": user_content
            })
        elif user_content:
            # 純文字內容
            messages.append({
                "role": "user",
                "content": user_content[0]["text"]
            })
        else:
            # 空內容
            messages.append({
                "role": "user",
                "content": "請分析提供的內容。"
            })
        
        
        # 調用API
        response = self._call_openrouter_api(actual_api_key, actual_model, messages, actual_seed)
        
        if not response:
            return (torch.zeros(1, target_height, target_width, 3), "❌ API調用失敗")
        
        # 檢查rate limit錯誤
        if 'error' in response:
            error_info = response['error']
            if isinstance(error_info, dict) and 'code' in error_info:
                if error_info['code'] == 429 or 'rate_limit' in str(error_info).lower():
                    error_msg = f"⚠️ Rate Limit達到: {error_info.get('message', '請稍後再試')}"
                    return (torch.zeros(1, target_height, target_width, 3), error_msg)
                else:
                    error_msg = f"❌ API錯誤: {error_info.get('message', str(error_info))}"
                    return (torch.zeros(1, target_height, target_width, 3), error_msg)
            else:
                error_msg = f"❌ API錯誤: {str(error_info)}"
                return (torch.zeros(1, target_height, target_width, 3), error_msg)
        
        if 'choices' not in response or not response['choices']:
            return (torch.zeros(1, target_height, target_width, 3), "❌ API回應格式異常")
        
        # 獲取回應訊息
        message = response['choices'][0]['message']
        response_content = message.get('content', '')
        
        # 檢查回應是否包含圖像數據
        output_image = torch.zeros(1, target_height, target_width, 3)  # 預設指定尺寸的空圖像
        output_text = response_content
        found_image = False
        
        # 首先檢查message.images字段
        if 'images' in message and message['images']:
            image_data = message['images'][0]
            
            try:
                if isinstance(image_data, str):
                    if image_data.startswith('data:image/'):
                        output_image = self._base64_to_tensor(image_data, enable_resize, target_width, target_height, resize_method)
                        found_image = True
                    elif image_data.startswith('http'):
                        output_image = self._url_to_tensor(image_data, enable_resize, target_width, target_height, resize_method)
                        found_image = True
                    else:
                        # 可能是純 base64，添加 data URL 前綴
                        full_data_url = f"data:image/png;base64,{image_data}"
                        output_image = self._base64_to_tensor(full_data_url, enable_resize, target_width, target_height, resize_method)
                        found_image = True
                        
                elif isinstance(image_data, dict):
                    # 處理嵌套的 image_url 字典格式
                    if 'type' in image_data and image_data['type'] == 'image_url' and 'image_url' in image_data:
                        nested_image_url = image_data['image_url']
                        if isinstance(nested_image_url, dict) and 'url' in nested_image_url:
                            image_url_value = nested_image_url['url']
                            if str(image_url_value).startswith('data:image/'):
                                output_image = self._base64_to_tensor(str(image_url_value), enable_resize, target_width, target_height, resize_method)
                            else:
                                output_image = self._url_to_tensor(str(image_url_value), enable_resize, target_width, target_height, resize_method)
                            found_image = True
                    # 處理直接包含 url 的格式
                    elif 'url' in image_data:
                        url_value = str(image_data['url'])
                        if url_value.startswith('data:image/'):
                            output_image = self._base64_to_tensor(url_value, enable_resize, target_width, target_height, resize_method)
                        else:
                            output_image = self._url_to_tensor(url_value, enable_resize, target_width, target_height, resize_method)
                        found_image = True
                    # 處理直接包含 data 的格式
                    elif 'data' in image_data:
                        output_image = self._base64_to_tensor(str(image_data['data']), enable_resize, target_width, target_height, resize_method)
                        found_image = True
                    
            except Exception as e:
                print(f"❌ 處理圖像失敗: {e}")
                found_image = False
        
        # 如果在message.images中沒有找到圖像，檢查content中的嵌入圖像
        if not found_image and is_image_model:
            # 檢查URL和base64格式的圖像
            url_patterns = [
                r'https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|webp)',
                r'!\[.*?\]\((https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|webp))\)',
                r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',
            ]
            
            base64_patterns = [
                r'data:image/png;base64,([A-Za-z0-9+/=\n\r]+)',
                r'data:image/jpeg;base64,([A-Za-z0-9+/=\n\r]+)',
                r'data:image/jpg;base64,([A-Za-z0-9+/=\n\r]+)',
            ]
            
            # 檢查URL格式
            for pattern in url_patterns:
                matches = re.findall(pattern, response_content, re.DOTALL)
                if matches:
                    try:
                        output_image = self._url_to_tensor(matches[0], enable_resize, target_width, target_height, resize_method)
                        found_image = True
                        output_text = re.sub(pattern, "[Generated Image]", response_content, flags=re.DOTALL)
                        break
                    except Exception as e:
                        continue
            
            # 檢查base64格式
            if not found_image:
                for pattern in base64_patterns:
                    matches = re.findall(pattern, response_content, re.DOTALL)
                    if matches:
                        try:
                            clean_base64 = re.sub(r'[\n\r\s]', '', matches[0])
                            base64_image = f"data:image/png;base64,{clean_base64}"
                            output_image = self._base64_to_tensor(base64_image, enable_resize, target_width, target_height, resize_method)
                            found_image = True
                            output_text = re.sub(pattern, "[Generated Image]", response_content, flags=re.DOTALL)
                            break
                        except Exception as e:
                            continue
        
        return (output_image, output_text)