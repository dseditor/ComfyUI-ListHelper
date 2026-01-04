import requests
import json
import base64
import os
import io
import torch
from PIL import Image
import numpy as np
import tempfile

# 檢查 torchaudio 是否可用
TORCHAUDIO_AVAILABLE = False
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    print("⚠️ torchaudio 未安裝，音訊功能將無法使用")

# Import JSON prompt extractor for automatic prompt list extraction
from .json_prompt_extractor import extract_prompts_from_json

class OpenAIHelper:
    """
    OpenAI Helper節點，用於呼叫OpenAI相容的API
    支援圖片、音訊輸入，配置管理，模型列表獲取，多配置範本
    """

    @classmethod
    def _get_config_files(cls):
        """獲取 modeldata 資料夾中的所有配置檔案"""
        modeldata_dir = os.path.join(os.path.dirname(__file__), "modeldata")
        
        # 確保資料夾存在
        if not os.path.exists(modeldata_dir):
            os.makedirs(modeldata_dir)
            # 創建預設配置
            default_config = {
                "endpoint": "",
                "api_key": "",
                "model_name": ""
            }
            default_path = os.path.join(modeldata_dir, "default.json")
            with open(default_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        # 獲取所有 .json 檔案
        config_files = []
        try:
            for file in os.listdir(modeldata_dir):
                if file.lower().endswith('.json'):
                    config_files.append(file)
        except Exception as e:
            print(f"⚠️ 讀取 modeldata 資料夾失敗: {e}")
        
        if not config_files:
            return ["default.json"]
        
        return sorted(config_files)

    @classmethod
    def _load_config(cls, config_file="default.json"):
        """載入指定的配置檔案"""
        modeldata_dir = os.path.join(os.path.dirname(__file__), "modeldata")
        config_path = os.path.join(modeldata_dir, config_file)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 載入配置檔案 {config_file} 失敗: {e}")
            return {
                "endpoint": "",
                "api_key": "",
                "model_name": ""
            }

    @classmethod
    def _save_config(cls, config_file, endpoint, api_key, model_name):
        """保存配置到指定的配置檔案"""
        modeldata_dir = os.path.join(os.path.dirname(__file__), "modeldata")
        config_path = os.path.join(modeldata_dir, config_file)
        
        try:
            config = {
                "endpoint": endpoint,
                "api_key": api_key,
                "model_name": model_name
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存配置檔案 {config_file} 失敗: {e}")

    @classmethod
    def _get_prompt_templates(cls):
        """Get all .md template files from Prompt folder"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_dir = os.path.join(current_dir, "Prompt")

        templates = []

        if os.path.exists(prompt_dir):
            for file in os.listdir(prompt_dir):
                if file.lower().endswith('.md'):
                    templates.append(file)

        if not templates:
            return ["No Template"]

        return sorted(templates)

    @classmethod
    def _load_template_content(cls, template_name):
        """Load template content"""
        if template_name == "No Template" or template_name == "Custom":
            return ""

        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "Prompt", template_name)

        if os.path.exists(template_path):
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""

        return ""

    @classmethod
    def INPUT_TYPES(cls):
        # 獲取配置檔案列表
        config_files = cls._get_config_files()
        
        # 載入預設配置（第一個配置檔案）
        default_config_file = config_files[0] if config_files else "default.json"
        config = cls._load_config(default_config_file)

        # 獲取範本列表
        templates = cls._get_prompt_templates()
        template_options = ["Custom"] + templates

        return {
            "required": {
                "config_template": (config_files, {
                    "default": default_config_file,
                    "tooltip": "選擇 API 配置範本（從 modeldata 資料夾）"
                }),
                "endpoint": ("STRING", {
                    "multiline": False,
                    "default": config.get("endpoint", ""),
                    "placeholder": "輸入OpenAI API端點，例如: https://api.openai.com/v1/chat/completions"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": config.get("api_key", ""),
                    "placeholder": "輸入您的API金鑰"
                }),
                "model_name": ("STRING", {
                    "multiline": False,
                    "default": config.get("model_name", ""),
                    "placeholder": "輸入模型名稱，例如: gpt-4o"
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "請分析提供的內容。"
                }),
                "prompt_template": (template_options, {
                    "default": template_options[0] if template_options else "Custom"
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 1,
                    "max": 128000,
                    "step": 1
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "請以繁體中文輸出使用者內容，不須包括引導或後綴，如「這就是你要的結果」、「以下是你要的結果」、「你要不要我幫你」、「你說的對」等等，只需要輸出使用者要的結論raw_text。請勿使用Markdown語法（如**粗體**），直接輸出純文字即可。"
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "audio": ("AUDIO",),
                "file_path": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "model_name_list", "prompts")
    OUTPUT_IS_LIST = (False, False, True)  # prompts 是列表
    FUNCTION = "process_openai"
    CATEGORY = "ListHelper/LLM"

    def __init__(self):
        pass

    def _process_audio(self, audio):
        """處理音訊並轉換為base64編碼"""
        if not TORCHAUDIO_AVAILABLE:
            print("❌ torchaudio未安裝，無法處理音訊")
            return None

        try:
            temp_file = None

            # 檢查不同的音訊輸入格式
            if isinstance(audio, dict):
                if "path" in audio:
                    # 直接路徑格式
                    audio_path = audio["path"]
                    print(f"處理來自路徑的音訊: {audio_path}")

                    if not os.path.exists(audio_path):
                        print(f"❌ 音訊文件不存在: {audio_path}")
                        return None

                    # 讀取音訊文件並轉換為base64
                    with open(audio_path, 'rb') as f:
                        audio_bytes = f.read()
                        return base64.b64encode(audio_bytes).decode('utf-8')

                elif "waveform" in audio and "sample_rate" in audio:
                    # ComfyUI音訊節點格式
                    print(f"處理來自waveform tensor的音訊")
                    waveform = audio["waveform"]
                    sample_rate = audio["sample_rate"]

                    # 創建臨時WAV文件
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_path = temp_file.name
                    temp_file.close()

                    # 確保waveform格式正確 [channels, samples]
                    if waveform.dim() == 3:
                        waveform = waveform.squeeze(0)  # 移除批次維度

                    # 保存為WAV文件
                    torchaudio.save(temp_path, waveform, sample_rate)

                    # 讀取並轉換為base64
                    with open(temp_path, 'rb') as f:
                        audio_bytes = f.read()
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                    # 清理臨時文件
                    os.unlink(temp_path)
                    return audio_b64

                else:
                    # 未知字典格式
                    print(f"❌ 未知的音訊字典格式: {list(audio.keys())}")
                    return None

            elif isinstance(audio, str) and os.path.exists(audio):
                # 直接文件路徑
                print(f"處理來自直接路徑的音訊: {audio}")
                with open(audio, 'rb') as f:
                    audio_bytes = f.read()
                    return base64.b64encode(audio_bytes).decode('utf-8')

            else:
                # 嘗試作為tensor處理
                print(f"嘗試將音訊作為tensor處理")
                if hasattr(audio, 'shape'):
                    # 創建臨時WAV文件
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_path = temp_file.name
                    temp_file.close()

                    # 假設sample_rate為44100（可以調整）
                    sample_rate = 44100
                    torchaudio.save(temp_path, audio, sample_rate)

                    # 讀取並轉換為base64
                    with open(temp_path, 'rb') as f:
                        audio_bytes = f.read()
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                    # 清理臨時文件
                    os.unlink(temp_path)
                    return audio_b64
                else:
                    print(f"❌ 無法識別的音訊格式")
                    return None

        except Exception as e:
            print(f"❌ 處理音訊時出錯: {e}")
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            return None

    def _tensor_to_base64(self, tensor):
        """將ComfyUI圖像tensor轉換為base64編碼"""
        # tensor shape: [B, H, W, C] (0-1 range)
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

    def _get_model_list(self, endpoint, api_key):
        """獲取可用模型列表"""
        try:
            # 將 chat/completions 端點改為 models 端點
            base_url = endpoint.rsplit('/chat/completions', 1)[0]
            models_endpoint = f"{base_url}/models"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            response = requests.get(
                models_endpoint,
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    # 提取模型ID
                    models = [model.get('id', '') for model in data['data']]
                    return ', '.join(models)
                else:
                    return "無法獲取模型列表"
            else:
                return f"獲取模型列表失敗: HTTP {response.status_code}"

        except Exception as e:
            print(f"❌ 獲取模型列表異常: {e}")
            return f"獲取模型列表異常: {str(e)}"

    def _call_openai_api(self, endpoint, api_key, model, messages, max_tokens, audio_b64=None):
        """呼叫OpenAI API"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result
            else:
                try:
                    error_data = response.json()
                    return {"error": error_data.get("error", {"message": f"HTTP {response.status_code}"})}
                except:
                    return {"error": {"message": f"HTTP {response.status_code}: {response.text[:200]}"}}

        except Exception as e:
            print(f"❌ API呼叫異常: {e}")
            return {"error": {"message": str(e)}}

    def process_openai(self, config_template, endpoint, api_key, model_name, user_prompt, prompt_template, max_tokens,
                       system_prompt=None, image1=None, image2=None, image3=None,
                       audio=None, file_path=None):
        """處理OpenAI請求"""

        # 驗證必填參數
        if not endpoint or not endpoint.strip():
            return ("❌ 錯誤: 請提供API端點", "", [])

        if not api_key or not api_key.strip():
            return ("❌ 錯誤: 請提供API金鑰", "", [])

        if not model_name or not model_name.strip():
            return ("❌ 錯誤: 請提供模型名稱", "", [])

        # 保存配置到選擇的配置範本
        self._save_config(config_template, endpoint.strip(), api_key.strip(), model_name.strip())

        # 獲取模型列表
        model_list = self._get_model_list(endpoint.strip(), api_key.strip())

        # 加載並應用範本
        template_content = ""
        if prompt_template != "Custom":
            template_content = self._load_template_content(prompt_template)

        # 如果範本內容存在，使用範本內容替換 system_prompt
        if template_content:
            system_prompt = template_content

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
        images_to_process = []
        for img_input in [image1, image2, image3]:
            if img_input is not None:
                images_to_process.append(img_input)

        # 添加圖像到用戶消息
        for image_tensor in images_to_process:
            try:
                base64_image = self._tensor_to_base64(image_tensor)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                })
            except Exception as e:
                print(f"❌ 處理圖像失敗: {e}")

        # 處理音訊輸入
        audio_b64 = None
        if audio is not None:
            if not TORCHAUDIO_AVAILABLE:
                return ("❌ 錯誤: torchaudio未安裝，無法處理音訊", model_list, [])

            print(f"處理音訊輸入")
            try:
                audio_b64 = self._process_audio(audio)
                if audio_b64:
                    # 添加音訊到用戶消息（使用inline_data格式）
                    user_content.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav"
                        }
                    })
                    print(f"✓ 音訊處理成功")
                else:
                    return ("❌ 錯誤: 音訊處理失敗", model_list, [])
            except Exception as e:
                print(f"❌ 處理音訊時出錯: {str(e)}")
                return (f"❌ 錯誤: 處理音訊時出錯: {str(e)}", model_list, [])

        # 添加用戶消息
        if len(user_content) > 1:
            # 多媒體內容（包含圖像或音訊）
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

        # 呼叫API
        response = self._call_openai_api(
            endpoint.strip(),
            api_key.strip(),
            model_name.strip(),
            messages,
            max_tokens,
            audio_b64
        )

        # 處理響應
        if not response:
            return ("❌ API呼叫失敗", model_list, [])

        # 檢查錯誤
        if 'error' in response:
            error_info = response['error']
            if isinstance(error_info, dict):
                error_msg = f"❌ API錯誤: {error_info.get('message', str(error_info))}"
            else:
                error_msg = f"❌ API錯誤: {str(error_info)}"
            return (error_msg, model_list, [])

        if 'choices' not in response or not response['choices']:
            return ("❌ API回應格式異常", model_list, [])

        # 獲取回應訊息
        message = response['choices'][0]['message']
        response_content = message.get('content', '')

        # 自動提取提示詞列表（如果輸出包含 JSON 格式的雜誌數據）
        try:
            prompts = extract_prompts_from_json(response_content)
            if prompts:
                print(f"✅ 自動提取到 {len(prompts)} 個圖片提示詞")
        except Exception as e:
            print(f"ℹ️ 提示詞提取失敗（這是正常的，如果輸出不是雜誌 JSON）: {e}")
            prompts = []

        return (response_content, model_list, prompts)
