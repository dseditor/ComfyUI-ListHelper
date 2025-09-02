import requests
import json
import base64
import os
import io
import re
import torch
from PIL import Image
import numpy as np

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
            },
            "optional": {
                "custom_model": ("STRING", {"multiline": False, "default": "", "placeholder": "輸入自定義模型名稱 (選擇Add Custom Model...時使用)"}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
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
    
    def _tensor_to_base64(self, tensor):
        """將ComfyUI圖像tensor轉換為base64編碼"""
        # tensor shape: [H, W, C] (0-1 range)
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 移除批次維度
        
        # 轉換為numpy並調整範圍到0-255
        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # 轉換為PIL圖像
        pil_image = Image.fromarray(numpy_image)
        
        # 轉換為base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    
    def _base64_to_tensor(self, base64_str):
        """將base64編碼轉換為ComfyUI圖像tensor"""
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
            
            # 轉換為numpy數組
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            
            # 轉換為tensor並添加批次維度 [1, H, W, C]
            tensor = torch.from_numpy(numpy_image).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            print(f"❌ base64轉tensor失敗: {e}")
            return torch.zeros(1, 1, 1, 3)
    
    def _url_to_tensor(self, image_url):
        """從URL下載圖像並轉換為ComfyUI圖像tensor"""
        try:
            # 下載圖像
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 轉換為PIL圖像
            pil_image = Image.open(io.BytesIO(response.content))
            
            # 確保是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 轉換為numpy數組
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            
            # 轉換為tensor並添加批次維度 [1, H, W, C]
            tensor = torch.from_numpy(numpy_image).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            print(f"❌ 從URL載入圖像失敗: {e}")
            return torch.zeros(1, 1, 1, 3)
    
    def _call_openrouter_api(self, api_key, model, messages):
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
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
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
    
    def process_llm(self, api_key, user_prompt, text_model, custom_model="", system_prompt="",
                   image_input_1=None, image_input_2=None, image_input_3=None):
        """處理LLM請求"""
        
        # 獲取API金鑰
        actual_api_key = self._get_api_key(api_key)
        if not actual_api_key:
            return (torch.zeros(1, 1, 1, 3), "❌ 錯誤: 請提供OpenRouter API金鑰")
        
        # 處理模型選擇
        if text_model == "Add Custom Model...":
            if not custom_model or not custom_model.strip():
                return (torch.zeros(1, 1, 1, 3), "❌ 請在custom_model欄位輸入自定義模型名稱")
            
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
        has_images = any([image_input_1 is not None, image_input_2 is not None, image_input_3 is not None])
        
        # 確定使用的模型
        if has_images:
            # 有圖像輸入時，強制使用gemini-2.5-flash-image-preview:free
            actual_model = "google/gemini-2.5-flash-image-preview:free"
        else:
            # 無圖像輸入時，使用用戶選擇的模型
            actual_model = selected_model
        
        # 檢查是否使用圖像生成模型（無論是否有圖像輸入）
        is_image_model = actual_model == "google/gemini-2.5-flash-image-preview:free"
        
        if has_images:
            # 有圖像輸入的情況
            # 為圖像生成添加特殊提示
            if not user_prompt or not user_prompt.strip():
                combined_text = "請分析這些圖像並生成一個相關的新圖像。請返回生成的圖像。"
            else:
                combined_text = user_prompt.strip()
                if "生成" in combined_text or "創建" in combined_text or "製作" in combined_text:
                    combined_text += "\n\n請返回生成的圖像。"
            
            # 重新添加文字內容
            user_content = [{
                "type": "text",
                "text": combined_text
            }]
            
            # 添加圖像到用戶消息
            for i, image_input in enumerate([image_input_1, image_input_2, image_input_3], 1):
                if image_input is not None:
                    try:
                        base64_image = self._tensor_to_base64(image_input)
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
                combined_text += "\n\n請生成並返回相應的圖像。"
            elif not combined_text:
                combined_text = "請生成一張圖像。"
            
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
        response = self._call_openrouter_api(actual_api_key, actual_model, messages)
        
        if not response:
            return (torch.zeros(1, 1, 1, 3), "❌ API調用失敗")
        
        # 檢查rate limit錯誤
        if 'error' in response:
            error_info = response['error']
            if isinstance(error_info, dict) and 'code' in error_info:
                if error_info['code'] == 429 or 'rate_limit' in str(error_info).lower():
                    error_msg = f"⚠️ Rate Limit達到: {error_info.get('message', '請稍後再試')}"
                    print(error_msg)
                    return (torch.zeros(1, 1, 1, 3), error_msg)
                else:
                    error_msg = f"❌ API錯誤: {error_info.get('message', str(error_info))}"
                    return (torch.zeros(1, 1, 1, 3), error_msg)
            else:
                error_msg = f"❌ API錯誤: {str(error_info)}"
                return (torch.zeros(1, 1, 1, 3), error_msg)
        
        if 'choices' not in response or not response['choices']:
            return (torch.zeros(1, 1, 1, 3), "❌ API回應格式異常")
        
        # 獲取回應訊息
        message = response['choices'][0]['message']
        response_content = message.get('content', '')
        
        # 檢查回應是否包含圖像數據
        output_image = torch.zeros(1, 1, 1, 3)  # 預設空圖像
        output_text = response_content
        found_image = False
        
        # 首先檢查message.images字段
        if 'images' in message and message['images']:
            image_data = message['images'][0]
            
            try:
                if isinstance(image_data, str):
                    if image_data.startswith('data:image/'):
                        output_image = self._base64_to_tensor(image_data)
                        found_image = True
                    elif image_data.startswith('http'):
                        output_image = self._url_to_tensor(image_data)
                        found_image = True
                    else:
                        # 可能是純 base64，添加 data URL 前綴
                        full_data_url = f"data:image/png;base64,{image_data}"
                        output_image = self._base64_to_tensor(full_data_url)
                        found_image = True
                        
                elif isinstance(image_data, dict):
                    # 處理嵌套的 image_url 字典格式
                    if 'type' in image_data and image_data['type'] == 'image_url' and 'image_url' in image_data:
                        nested_image_url = image_data['image_url']
                        if isinstance(nested_image_url, dict) and 'url' in nested_image_url:
                            image_url_value = nested_image_url['url']
                            if str(image_url_value).startswith('data:image/'):
                                output_image = self._base64_to_tensor(str(image_url_value))
                            else:
                                output_image = self._url_to_tensor(str(image_url_value))
                            found_image = True
                    # 處理直接包含 url 的格式
                    elif 'url' in image_data:
                        url_value = str(image_data['url'])
                        if url_value.startswith('data:image/'):
                            output_image = self._base64_to_tensor(url_value)
                        else:
                            output_image = self._url_to_tensor(url_value)
                        found_image = True
                    # 處理直接包含 data 的格式
                    elif 'data' in image_data:
                        output_image = self._base64_to_tensor(str(image_data['data']))
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
                        output_image = self._url_to_tensor(matches[0])
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
                            output_image = self._base64_to_tensor(base64_image)
                            found_image = True
                            output_text = re.sub(pattern, "[Generated Image]", response_content, flags=re.DOTALL)
                            break
                        except Exception as e:
                            continue
        
        return (output_image, output_text)