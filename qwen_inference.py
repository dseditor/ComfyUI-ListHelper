import os
import torch
import folder_paths
import requests
import time
import re
import gc
from typing import Optional, Tuple, Dict

class QwenGPUInference:
    """
    Qwen3-4B GPU 推理節點（優化載入速度版本 v3 - 支援記憶體管理）
    自動下載所需配置檔案並使用 GPU 進行推理
    包含 GPU 記憶體檢查與清理功能，避免與 ComfyUI 的 CLIP 模型衝突
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        self.config_dir = None

    @classmethod
    def _get_safetensors_files(cls):
        """從 text_encoders 資料夾中獲取所有 safetensors 檔案"""
        safetensors_files = []

        try:
            text_encoder_paths = folder_paths.get_folder_paths("text_encoders")
            for path in text_encoder_paths:
                if os.path.exists(path):
                    for file in os.listdir(path):
                        if file.lower().endswith('.safetensors'):
                            full_path = os.path.join(path, file)
                            if full_path not in safetensors_files:
                                safetensors_files.append(full_path)
        except:
            pass

        if not safetensors_files:
            return ["No safetensors files found"]

        return sorted(safetensors_files)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "一個女孩在咖啡廳"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": """你是一位專業的攝影提示詞優化專家。你的任務是將簡單的場景描述轉換為詳細、專業的攝影提示詞。

請根據用戶輸入的簡單描述，生成包含以下元素的完整提示詞：

1. **主體描述**：詳細描述主要拍攝對象（人物、物體、場景）
2. **環境細節**：周圍環境、背景元素、場景氛圍
3. **光影效果**：光線類型（自然光/人造光）、光線方向、光影對比、色溫
4. **相機設定**：視角、景深、焦距效果
5. **構圖元素**：畫面佈局、前景/中景/背景關係
6. **色彩氛圍**：主色調、色彩搭配、飽和度
7. **質感細節**：材質、紋理、細節表現
8. **情緒氛圍**：整體氛圍、情感表達

輸出格式：
- 使用英文輸出專業攝影術語
- 用逗號分隔各個元素
- 確保描述具體、可視覺化
- 長度控制在 150-300 個英文單詞

範例：
輸入：一個女孩在咖啡廳
輸出：A young woman sitting by the window in a cozy coffee shop, warm afternoon sunlight streaming through large glass windows creating soft shadows, wearing casual outfit, holding a cup of coffee, wooden table with laptop and notebook, blurred background with other customers, shallow depth of field, bokeh effect, warm color temperature, golden hour lighting, natural skin tones, professional photography, shot with 50mm lens, f/1.8 aperture, Instagram aesthetic, lifestyle photography, candid moment, peaceful atmosphere"""
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "do_sample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否使用採樣"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "inference"
    CATEGORY = "ListHelper"

    def _find_qwen_model(self) -> Optional[str]:
        """自動尋找 qwen_3_4b.safetensors 模型"""
        safetensors_files = self._get_safetensors_files()

        # 優先尋找 qwen_3_4b.safetensors
        for path in safetensors_files:
            if path != "No safetensors files found":
                basename = os.path.basename(path).lower()
                if "qwen" in basename and "3" in basename and "4b" in basename:
                    return path

        # 如果找不到特定模型，返回第一個 safetensors 檔案
        if safetensors_files and safetensors_files[0] != "No safetensors files found":
            return safetensors_files[0]

        return None

    def _remove_thinking_tags(self, text: str) -> str:
        """移除 <think>...</think> 標籤及其內容"""
        # 使用正則表達式移除所有 <think>...</think> 區塊
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # 移除多餘的空白行
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        return cleaned_text.strip()

    def _check_gpu_memory(self, required_gb: float = 8.0) -> Tuple[bool, str]:
        """
        檢查 GPU 記憶體是否足夠

        Args:
            required_gb: 需要的 GPU 記憶體大小（GB）

        Returns:
            (是否足夠, 詳細訊息)
        """
        if not torch.cuda.is_available():
            return True, "使用 CPU 模式，無需檢查 GPU 記憶體"

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
            free_memory = total_memory - reserved_memory

            info = f"""
GPU 記憶體狀態:
  總記憶體: {total_memory:.2f} GB
  已分配: {allocated_memory:.2f} GB
  已保留: {reserved_memory:.2f} GB
  可用: {free_memory:.2f} GB
  需要: {required_gb:.2f} GB
"""

            if free_memory < required_gb:
                return False, info + f"\n⚠️ 記憶體不足！缺少 {required_gb - free_memory:.2f} GB"

            return True, info + "\n✓ 記憶體充足"

        except Exception as e:
            return True, f"無法檢查 GPU 記憶體: {e}"

    def _free_gpu_memory(self) -> None:
        """
        釋放 GPU 記憶體
        清理 PyTorch 快取和執行垃圾回收
        """
        try:
            if torch.cuda.is_available():
                # 記錄清理前的記憶體
                before_allocated = torch.cuda.memory_allocated(0) / 1024**3
                before_reserved = torch.cuda.memory_reserved(0) / 1024**3

                # 清理 CUDA 快取
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # 強制垃圾回收
                gc.collect()

                # 再次清理
                torch.cuda.empty_cache()

                # 記錄清理後的記憶體
                after_allocated = torch.cuda.memory_allocated(0) / 1024**3
                after_reserved = torch.cuda.memory_reserved(0) / 1024**3

                freed_allocated = before_allocated - after_allocated
                freed_reserved = before_reserved - after_reserved

                print(f"\n✓ GPU 記憶體已清理:")
                print(f"  釋放已分配記憶體: {freed_allocated:.2f} GB")
                print(f"  釋放已保留記憶體: {freed_reserved:.2f} GB")
                print(f"  當前已分配: {after_allocated:.2f} GB")
                print(f"  當前已保留: {after_reserved:.2f} GB")
            else:
                gc.collect()
                print("✓ 執行垃圾回收（CPU 模式）")

        except Exception as e:
            print(f"⚠️ 清理記憶體時發生錯誤: {e}")
            # 即使發生錯誤，仍嘗試垃圾回收
            gc.collect()

    def _download_config_files(self, model_path: str, repo_id: str) -> bool:
        """自動下載 HuggingFace 配置檔案"""
        try:
            model_dir = os.path.dirname(model_path)
            model_basename = os.path.splitext(os.path.basename(model_path))[0]
            self.config_dir = os.path.join(model_dir, f"{model_basename}_config")

            config_files = [
                "config.json",
                "generation_config.json",
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json"
            ]

            all_exist = all(os.path.exists(os.path.join(self.config_dir, f)) for f in config_files)

            if all_exist:
                print(f"✓ 配置檔案已存在: {self.config_dir}")
                return True

            os.makedirs(self.config_dir, exist_ok=True)
            print(f"下載配置檔案到: {self.config_dir}")

            base_url = f"https://huggingface.co/{repo_id}/resolve/main/"

            for filename in config_files:
                filepath = os.path.join(self.config_dir, filename)

                if os.path.exists(filepath):
                    print(f"  ✓ {filename} 已存在")
                    continue

                url = base_url + filename
                print(f"  下載 {filename}...")

                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"  ✓ {filename} 下載完成")

                except Exception as e:
                    print(f"  ✗ {filename} 下載失敗: {str(e)}")
                    continue

            return True

        except Exception as e:
            print(f"❌ 下載配置檔案失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_model(self, model_path: str, repo_id: str) -> bool:
        """載入模型和 tokenizer（優化版 v3 - 包含記憶體管理）"""
        try:
            if self.model is not None and self.current_model_path == model_path:
                print(f"✓ 模型已載入: {os.path.basename(model_path)}")
                return True

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
                from safetensors.torch import load_file
            except ImportError as e:
                print(f"❌ 缺少必要的套件: {e}")
                print("請執行: pip install transformers safetensors")
                return False

            if not self._download_config_files(model_path, repo_id):
                return False

            print(f"\n載入模型: {os.path.basename(model_path)}")
            print("=" * 80)

            # 步驟 1: 檢查 GPU 記憶體
            print("\n步驟 1/3: 檢查 GPU 記憶體...")
            is_enough, memory_info = self._check_gpu_memory(required_gb=7.5)
            print(memory_info)

            # 步驟 2: 如果記憶體不足，嘗試清理
            if not is_enough:
                print("\n步驟 2/3: 記憶體不足，執行清理...")
                self._free_gpu_memory()

                # 再次檢查
                is_enough, memory_info = self._check_gpu_memory(required_gb=7.5)
                print("\n清理後的記憶體狀態:")
                print(memory_info)

                if not is_enough:
                    print("\n⚠️ GPU 記憶體仍然不足，將使用 CPU Offload 策略")
                    print("  - 部分模型層會放在 CPU，推理速度會較慢")
            else:
                print("\n步驟 2/3: 記憶體充足，跳過清理")

            print("\n步驟 3/3: 載入模型...")
            overall_start = time.time()

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # 1. 載入 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config_dir,
                trust_remote_code=True
            )

            # 2. 載入配置
            config = AutoConfig.from_pretrained(self.config_dir, trust_remote_code=True)

            # 3. 根據記憶體情況選擇載入策略
            import shutil
            temp_model_dir = os.path.join(self.config_dir, "temp_model")
            os.makedirs(temp_model_dir, exist_ok=True)

            # 決定載入策略
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 - torch.cuda.memory_reserved(0) / 1024**3

                if free_memory >= 7.5:
                    # 充足記憶體：完全載入到 GPU
                    print(f"⚡ 策略 1: 完全 GPU 載入（可用記憶體: {free_memory:.2f} GB）")
                    max_memory_config = None
                    offload_folder = None
                else:
                    # 記憶體不足：使用 CPU Offload
                    available_gpu = max(3.0, free_memory - 1.0)  # 至少保留 1GB 給其他操作
                    print(f"⚡ 策略 2: CPU Offload（可用 GPU: {free_memory:.2f} GB，分配: {available_gpu:.2f} GB）")
                    print(f"  - 部分模型層將放在 CPU，推理速度會較慢")
                    max_memory_config = {
                        0: f"{available_gpu:.1f}GB",
                        "cpu": "16GB"
                    }
                    # 創建 offload 資料夾
                    offload_folder = os.path.join(self.config_dir, "offload")
                    os.makedirs(offload_folder, exist_ok=True)
            else:
                print("⚡ 策略 3: CPU 模式")
                max_memory_config = None
                offload_folder = None

            load_start = time.time()

            try:
                # 複製配置檔案
                for file in ["config.json", "generation_config.json"]:
                    src = os.path.join(self.config_dir, file)
                    if os.path.exists(src):
                        shutil.copy(src, temp_model_dir)

                # 創建符號連結或複製 safetensors
                safetensors_target = os.path.join(temp_model_dir, "model.safetensors")
                if os.path.exists(safetensors_target):
                    os.remove(safetensors_target)

                # Windows 使用硬連結而不是符號連結
                try:
                    os.link(model_path, safetensors_target)
                except:
                    shutil.copy(model_path, safetensors_target)

                # 使用適當的載入策略
                load_kwargs = {
                    "pretrained_model_name_or_path": temp_model_dir,
                    "trust_remote_code": True,
                    "device_map": "auto",
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": True
                }

                if max_memory_config is not None:
                    load_kwargs["max_memory"] = max_memory_config

                if offload_folder is not None:
                    load_kwargs["offload_folder"] = offload_folder

                self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

                print(f"  ✓ 模型載入完成（耗時: {time.time() - load_start:.2f} 秒）")

            finally:
                # 清理臨時目錄
                try:
                    if os.path.exists(temp_model_dir):
                        shutil.rmtree(temp_model_dir)
                except:
                    pass

            missing_keys = []
            unexpected_keys = []

            if missing_keys:
                print(f"  警告: 缺少的鍵值: {len(missing_keys)} 個")
            if unexpected_keys:
                print(f"  警告: 未預期的鍵值: {len(unexpected_keys)} 個")

            self.current_model_path = model_path
            total_time = time.time() - overall_start

            print(f"\n驗證模型狀態...")
            if torch.cuda.is_available():
                current_allocated = torch.cuda.memory_allocated(0) / 1024**3
                current_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"  ✓ 模型已載入")
                print(f"  - GPU 記憶體已分配: {current_allocated:.2f} GB")
                print(f"  - GPU 記憶體已保留: {current_reserved:.2f} GB")

                # 檢查模型設備分佈
                device_map = {}
                for name, param in self.model.named_parameters():
                    device_str = str(param.device)
                    device_map[device_str] = device_map.get(device_str, 0) + 1

                print(f"  - 模型設備分佈:")
                for device_name, count in device_map.items():
                    print(f"    * {device_name}: {count} 個參數")
            else:
                print(f"  ✓ 模型在 CPU 上")

            print("\n" + "=" * 80)
            print(f"✓ 模型載入成功（總耗時: {total_time:.2f} 秒）")
            print("=" * 80)

            # 給出優化建議
            if total_time > 60:
                print(f"\n💡 載入優化建議:")
                print(f"  - 當前載入時間: {total_time:.1f} 秒")
                print(f"  - 主要瓶頸: 移動模型到 GPU")
                print(f"  - 這是正常的，無法進一步優化（硬體限制）")
                print(f"  - 模型會保留在記憶體中，下次使用會即時載入")

            print()
            return True

        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.tokenizer = None
            self.current_model_path = None
            return False

    def inference(
        self,
        user_prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Tuple[str]:
        """執行推理"""

        # 自動尋找 Qwen 模型
        model_path = self._find_qwen_model()
        if model_path is None:
            return ("❌ 錯誤: 未找到 Qwen 模型檔案\n請將 qwen_3_4b.safetensors 檔案放在 ComfyUI 的 models/text_encoders 資料夾中",)

        if not os.path.exists(model_path):
            return (f"❌ 錯誤: 模型檔案不存在: {model_path}",)

        # 使用固定的 repo_id
        repo_id = "Qwen/Qwen3-4B"
        if not self._load_model(model_path, repo_id):
            return ("❌ 錯誤: 模型載入失敗",)

        try:
            messages = []
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt")

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            inference_start = time.time()
            print(f"開始推理...")

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 移除 assistant 標記
            if "assistant" in response:
                for separator in ["<|im_start|>assistant\n", "assistant\n", "Assistant:", "assistant:"]:
                    if separator in response:
                        response = response.split(separator)[-1].strip()
                        break

            # 移除 <think> 標籤
            response = self._remove_thinking_tags(response)

            inference_time = time.time() - inference_start

            tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
            tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0

            print(f"✓ 推理完成（耗時: {inference_time:.2f} 秒）")
            print(f"  生成 tokens: {tokens_generated}")
            print(f"  速度: {tokens_per_sec:.1f} tokens/秒")

            if torch.cuda.is_available():
                print(f"  GPU 記憶體使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                print(f"  GPU 記憶體峰值: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

            return (response,)

        except Exception as e:
            import traceback
            error_msg = f"❌ 推理失敗: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (error_msg,)
