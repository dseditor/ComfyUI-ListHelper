import os
import torch
import folder_paths
import requests
import time
import re
import gc
import random
from typing import Optional, Tuple, Dict

class QwenGPUInference:
    """
    Qwen3-4B GPU Inference Node with intelligent memory management
    Auto-downloads required config files and performs GPU inference
    Includes GPU memory checking and cleanup to avoid conflicts with ComfyUI's CLIP models
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        self.config_dir = None
        self.current_quantization = None

    @classmethod
    def _get_safetensors_files(cls):
        """Get all safetensors files from text_encoders and clip folders"""
        safetensors_files = []

        # Search in text_encoders folder
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

        # Search in clip folder
        try:
            clip_paths = folder_paths.get_folder_paths("clip")
            for path in clip_paths:
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
        templates = cls._get_prompt_templates()
        # Add "Custom" option at the beginning of the list
        template_options = ["Custom"] + templates

        return {
            "required": {
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A girl in a coffee shop"
                }),
                "prompt_template": (template_options, {
                    "default": template_options[0] if template_options else "Custom"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "WARNING: Set to False to unload model after generation. Required for low VRAM workflows."
                }),
                "use_flash_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable FlashAttention-2 (may not improve speed for small batch inference)"
                }),
                "use_quantization": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable INT8 quantization for lower memory usage (slower inference)"
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

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("text", "used_seed",)
    FUNCTION = "inference"
    CATEGORY = "ListHelper"

    def _find_qwen_model(self) -> Optional[str]:
        """Auto-find qwen_3_4b.safetensors model"""
        safetensors_files = self._get_safetensors_files()

        # Prioritize finding qwen_3_4b.safetensors
        for path in safetensors_files:
            if path != "No safetensors files found":
                basename = os.path.basename(path).lower()
                if "qwen" in basename and "3" in basename and "4b" in basename:
                    return path

        # If specific model not found, return first safetensors file
        if safetensors_files and safetensors_files[0] != "No safetensors files found":
            return safetensors_files[0]

        return None

    def _remove_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags and their content"""
        # Use regex to remove all <think>...</think> blocks
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove extra blank lines
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        return cleaned_text.strip()

    def _check_gpu_memory(self, required_gb: float = 8.0) -> Tuple[bool, str]:
        """
        Check if GPU memory is sufficient

        Args:
            required_gb: Required GPU memory size (GB)

        Returns:
            (is_sufficient, detailed_message)
        """
        if not torch.cuda.is_available():
            return True, "GPU Memory: Using CPU mode"

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
            free_memory = total_memory - reserved_memory

            info = f"GPU Memory: Total {total_memory:.2f}GB | Free {free_memory:.2f}GB | Required {required_gb:.2f}GB"

            if free_memory < required_gb:
                return False, info + f" | Insufficient: need {required_gb - free_memory:.2f}GB more"

            return True, info + " | Sufficient"

        except Exception as e:
            return True, f"GPU Memory: Cannot check - {e}"

    def _free_gpu_memory(self) -> None:
        """
        Free GPU memory
        Clear PyTorch cache and run garbage collection
        """
        try:
            if torch.cuda.is_available():
                # Record memory before cleanup
                before_reserved = torch.cuda.memory_reserved(0) / 1024**3

                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Force garbage collection
                gc.collect()

                # Clear again
                torch.cuda.empty_cache()

                # Record memory after cleanup
                after_reserved = torch.cuda.memory_reserved(0) / 1024**3

                freed_reserved = before_reserved - after_reserved

                print(f"GPU Memory: Freed {freed_reserved:.2f}GB | Current reserved {after_reserved:.2f}GB")
            else:
                gc.collect()
                print("Memory cleanup: CPU mode")

        except Exception as e:
            print(f"Memory cleanup error: {e}")
            # Attempt garbage collection even if error occurs
            gc.collect()

    def _download_config_files(self, model_path: str, repo_id: str) -> bool:
        """Auto-download HuggingFace config files"""
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
                print(f"Config files exist: {self.config_dir}")
                return True

            os.makedirs(self.config_dir, exist_ok=True)
            print(f"Downloading config files to: {self.config_dir}")

            base_url = f"https://huggingface.co/{repo_id}/resolve/main/"

            for filename in config_files:
                filepath = os.path.join(self.config_dir, filename)

                if os.path.exists(filepath):
                    print(f"  {filename} exists")
                    continue

                url = base_url + filename
                print(f"  Downloading {filename}...")

                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"  {filename} downloaded")

                except Exception as e:
                    print(f"  Failed to download {filename}: {str(e)}")
                    continue

            return True

        except Exception as e:
            print(f"Failed to download config files: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_model(self, model_path: str, repo_id: str, use_quantization: bool = False, use_flash_attention: bool = False) -> bool:
        """Load model and tokenizer (optimized v4 - with memory management, quantization, and FlashAttention support)"""
        try:
            # Check if model is already loaded with same settings
            model_config_key = (model_path, use_quantization, use_flash_attention)
            current_config_key = (self.current_model_path, self.current_quantization, getattr(self, 'current_flash_attention', None))

            if self.model is not None and model_config_key == current_config_key:
                print(f"Model already loaded: {os.path.basename(model_path)} (Quantized: {use_quantization}, FlashAttn: {use_flash_attention})")
                return True

            # If any setting changed, need to reload
            if self.model is not None and model_config_key != current_config_key:
                print(f"Model settings changed, reloading...")
                self.model = None
                self.tokenizer = None
                self._free_gpu_memory()

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
                from safetensors.torch import load_file
            except ImportError as e:
                print(f"Missing required packages: {e}")
                print("Please run: pip install transformers safetensors")
                return False

            if not self._download_config_files(model_path, repo_id):
                return False

            print(f"\nLoading model: {os.path.basename(model_path)}")

            # Step 1: Check GPU memory (reduced threshold from 7.5GB to 6.0GB)
            is_enough, memory_info = self._check_gpu_memory(required_gb=6.0)
            print(memory_info)

            # Step 2: If insufficient memory, try to free up
            if not is_enough:
                print("Memory insufficient, cleaning up...")
                self._free_gpu_memory()

                # Check again
                is_enough, memory_info = self._check_gpu_memory(required_gb=6.0)
                print(memory_info)

                if not is_enough:
                    print("Still insufficient, will use CPU Offload strategy (slower inference)")

            overall_start = time.time()

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # 1. Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config_dir,
                trust_remote_code=True
            )

            # 2. Load config
            config = AutoConfig.from_pretrained(self.config_dir, trust_remote_code=True)

            # 3. Choose loading strategy based on memory
            import shutil
            temp_model_dir = os.path.join(self.config_dir, "temp_model")
            os.makedirs(temp_model_dir, exist_ok=True)

            # Determine loading strategy
            if torch.cuda.is_available():
                # Use allocated memory instead of reserved for more accurate free memory calculation
                free_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 - torch.cuda.memory_allocated(0) / 1024**3

                # Lower threshold to 6.0GB - model actually needs ~7.5GB but can work with less free space
                # This prevents unnecessary CPU offload when other models are loaded
                if free_memory >= 6.0:
                    # Sufficient memory: Full GPU loading
                    print(f"Strategy: Full GPU loading (Free: {free_memory:.2f}GB)")
                    max_memory_config = None
                    offload_folder = None
                else:
                    # Insufficient memory: Use CPU Offload
                    available_gpu = max(3.0, free_memory - 1.0)  # Reserve at least 1GB
                    print(f"Strategy: CPU Offload (Free: {free_memory:.2f}GB, Allocate: {available_gpu:.2f}GB)")
                    max_memory_config = {
                        0: f"{available_gpu:.1f}GB",
                        "cpu": "16GB"
                    }
                    # Create offload folder
                    offload_folder = os.path.join(self.config_dir, "offload")
                    os.makedirs(offload_folder, exist_ok=True)
            else:
                print("Strategy: CPU mode")
                max_memory_config = None
                offload_folder = None

            load_start = time.time()

            try:
                # Copy config files
                for file in ["config.json", "generation_config.json"]:
                    src = os.path.join(self.config_dir, file)
                    if os.path.exists(src):
                        shutil.copy(src, temp_model_dir)

                # Create link or copy safetensors
                safetensors_target = os.path.join(temp_model_dir, "model.safetensors")
                if os.path.exists(safetensors_target):
                    os.remove(safetensors_target)

                # Windows uses hard links instead of symbolic links
                try:
                    os.link(model_path, safetensors_target)
                except:
                    shutil.copy(model_path, safetensors_target)

                # Use appropriate loading strategy
                load_kwargs = {
                    "pretrained_model_name_or_path": temp_model_dir,
                    "trust_remote_code": True,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                }

                # Add FlashAttention configuration (only if enabled by user)
                if use_flash_attention:
                    if torch.cuda.is_available():
                        try:
                            import flash_attn
                            load_kwargs["attn_implementation"] = "flash_attention_2"
                            print(f"FlashAttention: Enabled (v{flash_attn.__version__})")
                        except ImportError:
                            print("=" * 70)
                            print("WARNING: FlashAttention-2 is enabled but 'flash-attn' is not installed")
                            print("Install with: pip install flash-attn")
                            print("Falling back to standard attention (no performance impact)")
                            print("=" * 70)
                    else:
                        print("FlashAttention: Disabled (GPU required)")

                # Add quantization configuration (only if enabled by user)
                if use_quantization:
                    if torch.cuda.is_available():
                        try:
                            from transformers import BitsAndBytesConfig

                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.0,
                                llm_int8_has_fp16_weight=False,
                            )
                            load_kwargs["quantization_config"] = quantization_config
                            print("Quantization: Enabled (INT8) - ~45% memory reduction")
                        except ImportError:
                            print("=" * 70)
                            print("ERROR: Quantization is enabled but 'bitsandbytes' is not installed")
                            print("Install with: pip install bitsandbytes")
                            print("Falling back to FP16 (using more memory)")
                            print("=" * 70)
                            load_kwargs["torch_dtype"] = dtype
                    else:
                        print("Quantization: Disabled (GPU required)")
                        load_kwargs["torch_dtype"] = dtype
                else:
                    # Default: use FP16 on GPU, FP32 on CPU
                    load_kwargs["torch_dtype"] = dtype

                if max_memory_config is not None:
                    load_kwargs["max_memory"] = max_memory_config

                if offload_folder is not None:
                    load_kwargs["offload_folder"] = offload_folder

                self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

                print(f"Model loaded (Time: {time.time() - load_start:.2f}s)")

            finally:
                # Clean up temp directory
                try:
                    if os.path.exists(temp_model_dir):
                        shutil.rmtree(temp_model_dir)
                except:
                    pass

            self.current_model_path = model_path
            self.current_quantization = use_quantization
            self.current_flash_attention = use_flash_attention
            total_time = time.time() - overall_start

            if torch.cuda.is_available():
                current_allocated = torch.cuda.memory_allocated(0) / 1024**3
                current_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU Memory: Allocated {current_allocated:.2f}GB | Reserved {current_reserved:.2f}GB")

            print(f"Model loaded successfully (Total time: {total_time:.2f}s)")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.tokenizer = None
            self.current_model_path = None
            return False

    def inference(
        self,
        user_prompt: str,
        prompt_template: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        seed: int = 0,
        keep_model_loaded: bool = True,
        use_flash_attention: bool = False,
        use_quantization: bool = False,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Tuple[str, int]:
        """Execute inference with optional FlashAttention-2 and INT8 quantization"""

        # Auto-find Qwen model
        model_path = self._find_qwen_model()
        if model_path is None:
            error_msg = "Error: Qwen model file not found.\nPlease place the correct model file (e.g., qwen_3_4b.safetensors) in ComfyUI's models/text_encoders or models/clip folder."
            print(error_msg)
            return (error_msg, seed)

        if not os.path.exists(model_path):
            error_msg = f"Error: Model file does not exist: {model_path}\nPlease place the correct model file in the text_encoders or clip folder."
            print(error_msg)
            return (error_msg, seed)

        # Use fixed repo_id
        repo_id = "Qwen/Qwen3-4B"
        if not self._load_model(model_path, repo_id, use_quantization, use_flash_attention):
            error_msg = "Error: Model loading failed. Please check the model file and ensure it's properly placed in the text_encoders or clip folder."
            print(error_msg)
            return (error_msg, seed)

        try:
            # Handle seed - ComfyUI's control_after_generate will handle randomization
            if seed > 0:
                torch.manual_seed(seed)
            
            # Load and apply template
            template_content = ""
            if prompt_template != "Custom":
                template_content = self._load_template_content(prompt_template)

            # If template content exists, replace system_prompt with template
            if template_content:
                system_prompt = template_content

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
            print(f"Inference starting (Seed: {seed})...")

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove assistant markers
            if "assistant" in response:
                for separator in ["<|im_start|>assistant\n", "assistant\n", "Assistant:", "assistant:"]:
                    if separator in response:
                        response = response.split(separator)[-1].strip()
                        break

            # Remove <think> tags
            response = self._remove_thinking_tags(response)

            inference_time = time.time() - inference_start

            tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
            tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0

            print(f"Inference completed (Time: {inference_time:.2f}s | Tokens: {tokens_generated} | Speed: {tokens_per_sec:.1f} tokens/s)")

            if torch.cuda.is_available():
                print(f"GPU Memory: Used {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB | Peak {torch.cuda.max_memory_allocated(0) / 1024**3:.2f}GB")

            # Check if we should unload the model to free VRAM
            if not keep_model_loaded:
                print("Explicitly unloading model to free VRAM...")
                del self.model
                del self.tokenizer
                self.model = None
                self.tokenizer = None
                self.current_model_path = None
                self._free_gpu_memory()

            return (response, seed)

        except Exception as e:
            import traceback
            error_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (error_msg, seed)
