import os
import sys
import gc
import time
import re
import subprocess
import platform
import base64
import io
import random
import json
import numpy as np
import folder_paths
import urllib.request
from typing import Optional, Tuple, List

# Import JSON prompt extractor for automatic prompt list extraction
from .json_prompt_extractor import extract_prompts_from_json

# Suggested models for download
SUGGESTED_MODELS = {
    "Download: Z-Image": "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q5_K_M.gguf",
    "Download: Z-Image (Abliterated)": "https://huggingface.co/Mungert/Qwen3-4B-abliterated-GGUF/resolve/main/Qwen3-4B-abliterated-q4_k_m.gguf",
    "Download: Qwen": "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    "Download: Qwen (Abliterated)": "https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-abliterated-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf",
    "Download: QwenVL": "https://huggingface.co/mradermacher/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF/resolve/main/Huihui-Qwen3-VL-4B-Instruct-abliterated.Q4_K_M.gguf",
}

SUGGESTED_MMPROJ = {
    "Download: mmproj": "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-F16.gguf",
    "Download: QwenVL mmproj": "https://huggingface.co/noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-i1-GGUF/resolve/main/mmproj-F16.gguf",
}

class GGUFInference:
    """
    GGUF Model Inference Node with llama-cpp-python
    Supports both text-only and vision models with optional mmproj files
    Auto-searches GGUF files from text_encoders and clip folders
    """

    def __init__(self):
        self.model = None
        self.current_model_path = None
        self.current_mmproj_path = None
        self.clip_model_array = None
        self.llama_cpp_available = False
        self.failed_versions = []  # Track failed installation versions
        self.ggml_error_retry_count = 0  # Track retry attempts for ggml errors
        self.max_ggml_retries = 2  # Maximum retry attempts
        self._check_llama_cpp()

    def _check_llama_cpp(self):
        """Check if llama-cpp-python is available"""
        try:
            import llama_cpp
            self.llama_cpp_available = True
            print("llama-cpp-python is available")
        except ImportError:
            self.llama_cpp_available = False
            print("llama-cpp-python is not installed")

    @classmethod
    def _download_file(cls, url: str, destination: str) -> bool:
        """Download file from URL with progress reporting"""
        try:
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded * 100 / total_size, 100)
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"⬇️ {percent:.0f}% ({downloaded_mb:.0f}MB/{total_mb:.0f}MB)", end='\r')

            urllib.request.urlretrieve(url, destination, reporthook=report_progress)
            print(f"\n✓ Downloaded: {os.path.basename(destination)}")
            return True
        except Exception as e:
            print(f"\n⚠️ Download failed: {e}")
            return False

    @classmethod
    def _get_gguf_files(cls):
        """Get all GGUF files from text_encoders and clip folders"""
        gguf_files = []

        # Search in text_encoders folder
        try:
            text_encoder_paths = folder_paths.get_folder_paths("text_encoders")
            for path in text_encoder_paths:
                if os.path.exists(path):
                    for file in os.listdir(path):
                        if file.lower().endswith('.gguf'):
                            full_path = os.path.join(path, file)
                            if full_path not in gguf_files:
                                gguf_files.append(full_path)
        except:
            pass

        # Search in clip folder
        try:
            clip_paths = folder_paths.get_folder_paths("clip")
            for path in clip_paths:
                if os.path.exists(path):
                    for file in os.listdir(path):
                        if file.lower().endswith('.gguf'):
                            full_path = os.path.join(path, file)
                            if full_path not in gguf_files:
                                gguf_files.append(full_path)
        except:
            pass

        # Add suggested models to the list
        suggested_models = list(SUGGESTED_MODELS.keys())

        if not gguf_files:
            # If no local models, only show suggested models
            return suggested_models

        # Return local models first, then suggested models
        return sorted(gguf_files) + suggested_models

    @classmethod
    def _get_mmproj_files(cls):
        """Get all mmproj files from text_encoders and clip folders"""
        mmproj_files = []

        # Search in text_encoders folder
        try:
            text_encoder_paths = folder_paths.get_folder_paths("text_encoders")
            for path in text_encoder_paths:
                if os.path.exists(path):
                    for file in os.listdir(path):
                        if 'mmproj' in file.lower() and file.lower().endswith('.gguf'):
                            full_path = os.path.join(path, file)
                            if full_path not in mmproj_files:
                                mmproj_files.append(full_path)
        except:
            pass

        # Search in clip folder
        try:
            clip_paths = folder_paths.get_folder_paths("clip")
            for path in clip_paths:
                if os.path.exists(path):
                    for file in os.listdir(path):
                        if 'mmproj' in file.lower() and file.lower().endswith('.gguf'):
                            full_path = os.path.join(path, file)
                            if full_path not in mmproj_files:
                                mmproj_files.append(full_path)
        except:
            pass

        # Add suggested mmproj to the list
        suggested_mmproj = list(SUGGESTED_MMPROJ.keys())

        if not mmproj_files:
            # If no local mmproj files, show "No mmproj files" and suggested option
            return ["No mmproj files"] + suggested_mmproj

        # Return local mmproj files first, then suggested option
        return sorted(mmproj_files) + suggested_mmproj

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
        gguf_files = cls._get_gguf_files()
        # Get file names only for dropdown
        gguf_names = [os.path.basename(f) if f != "No GGUF files found" else f for f in gguf_files]

        # Get mmproj files
        mmproj_files = cls._get_mmproj_files()
        mmproj_names = [os.path.basename(f) if f != "No mmproj files" else f for f in mmproj_files]

        # Get prompt templates
        templates = cls._get_prompt_templates()
        template_options = ["Custom"] + templates

        # Set default model to QwenVL download option if available
        default_model = "Download: QwenVL" if "Download: QwenVL" in gguf_names else (gguf_names[0] if gguf_names else "No GGUF files found")

        # Set default mmproj to QwenVL mmproj download option if available
        default_mmproj = "Download: QwenVL mmproj" if "Download: QwenVL mmproj" in mmproj_names else (mmproj_names[0] if mmproj_names else "No mmproj files")

        # Set default template to image_to_prompt.md if available, otherwise Custom
        default_template = "image_to_prompt.md" if "image_to_prompt.md" in template_options else (template_options[0] if template_options else "Custom")

        return {
            "required": {
                "model": (gguf_names, {
                    "default": default_model
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "prompt_template": (template_options, {
                    "default": default_template
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "max_tokens": ("INT", {
                    "default": 3072,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model in memory after inference"
                }),
                "mmproj_file": (mmproj_names, {
                    "default": default_mmproj,
                    "tooltip": "Vision model mmproj file (auto-enabled when any image is provided and model is VL type)"
                }),
                "image_1": ("IMAGE", {
                    "tooltip": "First input image for vision model (auto-enables vision mode for VL models)"
                }),
                "image_2": ("IMAGE", {
                    "tooltip": "Second input image for vision model (optional, for multi-image analysis)"
                }),
                "image_3": ("IMAGE", {
                    "tooltip": "Third input image for vision model (optional, for multi-image analysis)"
                }),
                "auto_install_llama_cpp": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-install/repair llama-cpp-python (supports all platforms/CUDA versions, prioritizes Basic version for compatibility)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("text", "used_seed", "prompts")
    OUTPUT_IS_LIST = (False, False, True)  # prompts 是列表
    FUNCTION = "inference"
    CATEGORY = "ListHelper/LLM"

    def _free_memory(self):
        """Free memory with robust handling for Vision Handlers & Zombie Wrappers"""
        print("🧹 Releasing resources...")
        try:
            import llama_cpp
            
            # 1. Vision Cleanup (Qwen2.5-VL / Llava)
            if self.clip_model_array is not None:
                # Capture pointer (legacy handlers)
                clip_ptr = None
                if hasattr(self.clip_model_array, 'clip_model'):
                    clip_ptr = self.clip_model_array.clip_model
                elif hasattr(self.clip_model_array, 'vision_model'):
                    clip_ptr = self.clip_model_array.vision_model

                # A. Close Exit Stack (Crucial for Qwen25VLChatHandler)
                if hasattr(self.clip_model_array, '_exit_stack') and self.clip_model_array._exit_stack:
                    try: self.clip_model_array._exit_stack.close()
                    except: pass

                # B. Standard Close
                if hasattr(self.clip_model_array, 'close'):
                    try: self.clip_model_array.close()
                    except: pass
                
                # C. Force Kill (Legacy/Direct Pointer)
                if clip_ptr:
                     try: llama_cpp.llama_free_model(clip_ptr)
                     except: pass

                del self.clip_model_array
                self.clip_model_array = None

            # 2. Main Model Cleanup
            if self.model is not None:
                # --- CAPTURE RAW POINTERS ---
                raw_model_ptr = None
                raw_ctx_ptr = None
                
                if hasattr(self.model, '_model') and self.model._model:
                    if hasattr(self.model._model, 'model'):
                        raw_model_ptr = self.model._model.model
                    else:
                        raw_model_ptr = self.model._model
                elif hasattr(self.model, 'model') and self.model.model:
                     raw_model_ptr = self.model.model

                if hasattr(self.model, '_ctx') and self.model._ctx:
                    if hasattr(self.model._ctx, 'ctx'):
                        raw_ctx_ptr = self.model._ctx.ctx
                    else:
                        raw_ctx_ptr = self.model._ctx
                
                # A. Try Standard Close
                if hasattr(self.model, 'close'):
                    try: self.model.close()
                    except: pass
                
                # B. FORCE KILL using Captured Pointers
                if raw_model_ptr:
                    try: llama_cpp.llama_free_model(raw_model_ptr)
                    except: pass

                if raw_ctx_ptr:
                    try: llama_cpp.llama_free(raw_ctx_ptr)
                    except: pass

                # C. Nullify Wrappers
                if hasattr(self.model, '_model'): self.model._model = None
                if hasattr(self.model, '_ctx'): self.model._ctx = None
                
                del self.model
                self.model = None

            self.current_model_path = None
            self.current_mmproj_path = None

            # 3. System Cleanup
            import gc
            gc.collect() 
            gc.collect() 
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except: pass

            print("✓ Resources released")

        except Exception as e:
            print(f"⚠️ Error freeing memory: {e}")

    def _free_image_memory(self):
        """Free image-related memory without unloading model"""
        try:
            # Run garbage collection multiple times
            for _ in range(2):
                gc.collect()

            # Try to free CUDA memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Empty cache again after sync
                    torch.cuda.empty_cache()
            except:
                pass

            # Image memory freed silently
            pass
        except Exception as e:
            print(f"⚠️ Error freeing image memory: {e}")

    def _is_ggml_error(self, exception) -> bool:
        """Detect if exception is related to ggml/CUDA compatibility issues

        Args:
            exception: The exception object to check

        Returns:
            True if this is a ggml-related error that might be fixed by reinstalling
        """
        error_str = str(exception).lower()

        # Check for common ggml-related error patterns
        ggml_patterns = [
            'ggml',
            'cannot load library',
            'dll load failed',
            'cuda',
            'shared library',
            'importerror',
            'oserror',
        ]

        return any(pattern in error_str for pattern in ggml_patterns)

    def _handle_ggml_error_and_reinstall(self) -> Tuple[bool, Optional[str]]:
        """Handle ggml error by detecting system and reinstalling compatible version

        This method is called when a ggml-related error occurs during model loading.
        It will:
        1. Re-detect system configuration (Python/CUDA versions)
        2. Uninstall current llama-cpp-python
        3. Search for and install compatible version

        Returns:
            Tuple of (success: bool, message: str or None)
        """
        print("=" * 70)
        print("GGML ERROR DETECTED - Attempting automatic recovery")
        print("=" * 70)

        # Increment retry count
        self.ggml_error_retry_count += 1

        # Check if we've exceeded max retries
        if self.ggml_error_retry_count > self.max_ggml_retries:
            error_msg = (
                f"Maximum retry attempts ({self.max_ggml_retries}) reached.\n\n"
                "Unable to automatically resolve the ggml compatibility issue.\n"
                "This usually means:\n"
                "1. Your CUDA/Python version changed and no compatible wheel is available\n"
                "2. System libraries are missing or corrupted\n\n"
                "Please manually install from:\n"
                "https://github.com/JamePeng/llama-cpp-python/releases"
            )
            print(error_msg)
            print("=" * 70)
            return (False, error_msg)

        print(f"Retry attempt {self.ggml_error_retry_count}/{self.max_ggml_retries}")
        print("Re-detecting system configuration...")

        # Get current installed version to add to skip list
        try:
            import pkg_resources
            current_version = pkg_resources.get_distribution("llama-cpp-python").version
            # Extract tag from version if possible
            # Version format like: 0.3.18+cu128
            if current_version:
                print(f"Current version: {current_version}")
                # Add current version to failed list if not already there
                # We need to find the matching tag, but for now just track by version
                pass
        except:
            current_version = None

        # Uninstall current version
        print("Uninstalling incompatible version...")
        if not self._uninstall_llama_cpp():
            error_msg = (
                "Failed to uninstall current llama-cpp-python.\n"
                "Please manually uninstall: pip uninstall llama-cpp-python"
            )
            print(error_msg)
            print("=" * 70)
            return (False, error_msg)

        # Try to install compatible version, skipping previously failed versions
        print("Searching for compatible version...")
        print("System will be re-detected automatically during installation")
        print("=" * 70)

        success, new_version = self._install_llama_cpp(skip_versions=set(self.failed_versions))

        if success and new_version:
            # Add this version to failed list (since we got here, it had issues)
            if new_version not in self.failed_versions:
                self.failed_versions.append(new_version)

            success_msg = (
                f"Successfully installed llama-cpp-python {new_version}\n\n"
                "IMPORTANT: Please restart ComfyUI to use the new version.\n"
                "The current session may still have compatibility issues."
            )
            print("=" * 70)
            print(success_msg)
            print("=" * 70)
            return (True, success_msg)
        else:
            # Add to failed list even if we couldn't install
            if new_version and new_version not in self.failed_versions:
                self.failed_versions.append(new_version)

            error_msg = (
                "Failed to find/install compatible llama-cpp-python version.\n\n"
                "Please visit: https://github.com/JamePeng/llama-cpp-python/releases\n"
                "And manually install a version compatible with:\n"
                f"- Python {sys.version_info.major}.{sys.version_info.minor}\n"
            )

            # Add CUDA info if available
            cuda_version = self._detect_cuda_version()
            if cuda_version:
                error_msg += f"- CUDA {cuda_version}\n"

            print(error_msg)
            print("=" * 70)
            return (False, error_msg)

    def _detect_cuda_version(self):
        """Detect CUDA version from system"""
        try:
            # Try to get CUDA version from torch
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                if cuda_version:
                    # Convert 12.8 to 128, 12.1 to 121, etc.
                    major, minor = cuda_version.split('.')
                    return f"cu{major}{minor}"
        except:
            pass

        # Try to get CUDA version from nvcc
        try:
            result = subprocess.run(['nvcc', '--version'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5)
            if result.returncode == 0:
                # Parse output like "Cuda compilation tools, release 12.8, V12.8.89"
                match = re.search(r'release (\d+)\.(\d+)', result.stdout)
                if match:
                    major, minor = match.groups()
                    return f"cu{major}{minor}"
        except:
            pass

        return None

    def _parse_release_info(self, release):
        """Parse release tag to extract CUDA version, AVX support, and OS info

        Example tag formats:
        - v0.3.18-cu130-AVX2-win-20251220 -> CUDA 13.0, AVX2, Windows
        - v0.3.18-cu128-AVX2-linux-20251220 -> CUDA 12.8, AVX2, Linux
        - v0.3.16+cu128avx2 -> CUDA 12.8, AVX2
        - v0.3.16+cu121 -> CUDA 12.1, no AVX
        - v0.3.16+avx2 -> no CUDA, AVX2
        - v0.3.16 -> no CUDA, no AVX
        """
        tag_name = release.get('tag_name', '')

        # Extract CUDA version (e.g., cu128, cu121, cu130)
        # Support both formats: -cu130- and +cu128
        cuda_match = re.search(r'[-+]cu(\d+)', tag_name)
        cuda_version = f"cu{cuda_match.group(1)}" if cuda_match else None

        # Check for AVX2 support (case-insensitive)
        has_avx2 = 'avx2' in tag_name.lower()

        # Check for AVX (non-AVX2)
        has_avx = 'avx' in tag_name.lower() and not has_avx2

        return {
            'tag_name': tag_name,
            'cuda_version': cuda_version,
            'has_avx2': has_avx2,
            'has_avx': has_avx,
            'release': release
        }

    def _get_github_releases(self, max_releases=15):
        """Get recent releases from GitHub"""
        try:
            api_url = f"https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page={max_releases}"

            # Create request with User-Agent header
            req = urllib.request.Request(api_url)
            req.add_header('User-Agent', 'ComfyUI-GGUFInference/1.0')

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data  # Returns list of releases
        except Exception as e:
            print(f"Failed to fetch GitHub releases: {e}")
            return None

    def _find_matching_release(self, releases_list, skip_versions=None):
        """Find matching release and wheel URL from releases list based on system configuration

        Args:
            releases_list: List of GitHub releases
            skip_versions: Set of version tags to skip (previously failed versions)
        """
        if not releases_list:
            return None

        if skip_versions is None:
            skip_versions = set()

        # Get system info
        system = platform.system().lower()  # 'windows', 'linux', 'darwin'
        py_version = sys.version_info
        py_ver = f"cp{py_version.major}{py_version.minor}"

        # Detect CUDA version
        cuda_version = self._detect_cuda_version()

        print("=" * 70)
        print("Detecting system configuration:")
        print(f"  OS: {system}")
        print(f"  Python: {py_version.major}.{py_version.minor} ({py_ver})")
        print(f"  CUDA: {cuda_version if cuda_version else 'Not detected'}")
        if skip_versions:
            print(f"  Skipping versions: {', '.join(sorted(skip_versions))}")
        print("=" * 70)

        # Determine platform suffix
        if system == 'windows':
            platform_suffix = 'win_amd64.whl'
        elif system == 'linux':
            platform_suffix = 'linux_x86_64.whl'
        elif system == 'darwin':
            # macOS - try both arm64 and x86_64
            import platform as plt
            machine = plt.machine().lower()
            if 'arm' in machine or 'aarch64' in machine:
                platform_suffix = 'macosx_11_0_arm64.whl'
            else:
                platform_suffix = 'macosx_10_9_x86_64.whl'
        else:
            print(f"Unsupported OS: {system}")
            return None

        # Build priority list for release matching
        # Priority: CUDA Basic > CPU Basic
        # Note: Only use Basic versions (no AVX/AVX2) for maximum compatibility
        #       AVX and AVX2 versions are skipped due to compatibility issues
        print("=" * 70)
        print("Searching through releases...")
        print("Note: Skipping AVX/AVX2 versions for better compatibility")
        print("=" * 70)

        # Try each priority level
        priority_filters = []

        if cuda_version and system != 'darwin':
            # Priority 1: CUDA exact version match (Basic only - no AVX/AVX2)
            priority_filters.append({
                'name': f'CUDA {cuda_version} Basic (exact match)',
                'cuda': cuda_version,
                'require_basic': True,  # Only match Basic (no AVX, no AVX2)
                'allow_cuda_upgrade': False
            })
            # Priority 2: CUDA compatible version (Basic only - allows newer CUDA)
            # CUDA is forward compatible, e.g., cu130 wheels work on cu128 systems
            priority_filters.append({
                'name': f'CUDA {cuda_version}+ Basic (compatible)',
                'cuda': cuda_version,
                'require_basic': True,
                'allow_cuda_upgrade': True
            })

        # Priority 3: CPU Basic version (no AVX, no AVX2)
        priority_filters.append({
            'name': 'CPU Basic',
            'cuda': None,
            'require_basic': True
        })

        # Search through releases with priority
        for priority in priority_filters:
            print(f"Trying priority: {priority['name']}")

            for release in releases_list:
                release_info = self._parse_release_info(release)

                # Skip this version if it's in the skip list
                if release_info['tag_name'] in skip_versions:
                    continue

                # Check if this release matches current priority
                if priority.get('allow_cuda_upgrade', False):
                    # Allow CUDA version upgrade (e.g., cu130 wheel on cu128 system)
                    # Extract numeric CUDA version for comparison
                    if priority['cuda'] and release_info['cuda_version']:
                        try:
                            required_cuda = int(priority['cuda'][2:])  # cu128 -> 128
                            release_cuda = int(release_info['cuda_version'][2:])  # cu130 -> 130
                            # Match if release CUDA >= required CUDA (forward compatibility)
                            cuda_match = release_cuda >= required_cuda
                        except:
                            cuda_match = False
                    else:
                        cuda_match = (priority['cuda'] is None and release_info['cuda_version'] is None)
                else:
                    # Exact CUDA version match
                    cuda_match = (priority['cuda'] is None and release_info['cuda_version'] is None) or \
                                (priority['cuda'] == release_info['cuda_version'])

                # Handle require_basic flag: skip all AVX and AVX2 versions
                if priority.get('require_basic', False):
                    # Only match Basic versions (no AVX, no AVX2)
                    basic_match = not release_info['has_avx'] and not release_info['has_avx2']
                else:
                    # Old logic for backward compatibility (not used in new priority system)
                    basic_match = True

                if cuda_match and basic_match:
                    # Found matching release, now find wheel for this Python version
                    print(f"  Found matching release: {release_info['tag_name']}")

                    # Search for wheel with matching Python version
                    for asset in release.get('assets', []):
                        name = asset.get('name', '')
                        download_url = asset.get('browser_download_url', '')

                        if not name.endswith('.whl'):
                            continue

                        # Check if wheel matches Python version and platform
                        if f"-{py_ver}-{py_ver}-" in name and platform_suffix in name:
                            print("=" * 70)
                            print(f"SUCCESS: Found matching wheel!")
                            print(f"  Release: {release_info['tag_name']}")
                            print(f"  Wheel: {name}")
                            print(f"  Download URL: {download_url}")
                            print("=" * 70)
                            return (download_url, release_info['tag_name'])

                    # Release matched but no wheel for this Python version
                    print(f"    No wheel found for Python {py_ver} in this release")

        # No matching release found
        print("=" * 70)
        print("No matching release found for your system configuration")
        print("Available releases:")
        for release in releases_list[:5]:  # Show first 5 releases
            release_info = self._parse_release_info(release)
            print(f"  - {release_info['tag_name']} (CUDA: {release_info['cuda_version']}, AVX: {release_info['has_avx']}, AVX2: {release_info['has_avx2']})")
        print("=" * 70)

        return None

    def _uninstall_llama_cpp(self):
        """Uninstall existing llama-cpp-python"""
        print("=" * 70)
        print("Uninstalling existing llama-cpp-python...")
        print("=" * 70)

        try:
            # Use pip uninstall with -y flag to auto-confirm
            result = subprocess.run([
                sys.executable, "-m", "pip", "uninstall", "llama-cpp-python", "-y"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("=" * 70)
                print("SUCCESS: llama-cpp-python uninstalled successfully")
                print("=" * 70)
                return True
            else:
                print("=" * 70)
                print(f"Uninstall completed with warnings:")
                print(result.stdout)
                print(result.stderr)
                print("=" * 70)
                return True  # Still consider it successful

        except Exception as e:
            print("=" * 70)
            print(f"ERROR: Failed to uninstall llama-cpp-python: {e}")
            print("Please manually uninstall using: pip uninstall llama-cpp-python")
            print("=" * 70)
            return False

    def _install_llama_cpp(self, skip_versions=None):
        """Auto-installation: Auto-detect system and download from GitHub releases

        Supports all platforms (Windows/Linux/macOS) and CUDA versions.
        Only uses Basic versions (no AVX/AVX2) for maximum compatibility.

        Args:
            skip_versions: Set of version tags to skip (previously failed versions)

        Returns:
            Tuple of (success: bool, installed_version: str or None)
        """
        if skip_versions is None:
            skip_versions = set()

        print("=" * 70)
        print("Auto-Installing llama-cpp-python")
        print("Detecting system configuration and searching through releases...")
        print("=" * 70)

        # Get recent releases (up to 15)
        releases_list = self._get_github_releases(max_releases=15)
        if not releases_list:
            print("=" * 70)
            print("ERROR: Failed to fetch releases from GitHub")
            print("Please visit: https://github.com/JamePeng/llama-cpp-python/releases")
            print("=" * 70)
            return (False, None)

        # Find matching release and wheel
        result = self._find_matching_release(releases_list, skip_versions=skip_versions)
        if not result:
            print("=" * 70)
            print("ERROR: Could not find a matching wheel for your system")
            print("Please visit: https://github.com/JamePeng/llama-cpp-python/releases")
            print("And manually download the appropriate wheel for your system")
            print("=" * 70)
            return (False, None)

        wheel_url, version_tag = result

        # Download and install
        try:
            print("=" * 70)
            print("Installing llama-cpp-python...")
            print(f"Version: {version_tag}")
            print("=" * 70)

            subprocess.check_call([
                sys.executable, "-m", "pip", "install", wheel_url
            ])

            print("=" * 70)
            print("SUCCESS: llama-cpp-python installed successfully")
            print("IMPORTANT: Please restart ComfyUI to use the GGUF node")
            print("=" * 70)
            return (True, version_tag)

        except Exception as e:
            print("=" * 70)
            print(f"ERROR: Failed to install llama-cpp-python: {e}")
            print("")
            print("Please visit: https://github.com/JamePeng/llama-cpp-python/releases")
            print("=" * 70)
            return (False, version_tag)


    def _is_vision_model(self, model_path: str) -> bool:
        """Check if model is a vision model based on filename"""
        model_name = os.path.basename(model_path).lower()
        return 'vl' in model_name

    def _load_model(self, model_path: str, enable_vision: bool = False, mmproj_path: Optional[str] = None, force_reload: bool = False) -> bool:
        """Load GGUF model with llama-cpp-python

        Args:
            model_path: Path to GGUF model file
            enable_vision: Whether to enable vision mode
            mmproj_path: Path to mmproj file (for vision models)
            force_reload: Force reload even if model is already loaded
        """
        try:
            # Check if model and mmproj are already loaded
            if (not force_reload and
                self.model is not None and
                self.current_model_path == model_path and
                self.current_mmproj_path == mmproj_path):
                # Model already loaded, skip verbose output
                return True

            # Unload previous model if model or mmproj changed, or force reload
            if self.model is not None:
                # Unload silently
                self._free_memory()
                # Wait a bit for memory to be fully released
                time.sleep(0.5)

            if not self.llama_cpp_available:
                return False

            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            # Check if this is a vision model
            is_vision_model = self._is_vision_model(model_path)

            # Only print model name for initial load
            print(f"Loading: {os.path.basename(model_path)}")

            load_start = time.time()

            # Load model with appropriate settings
            load_kwargs = {
                "model_path": model_path,
                "n_ctx": 8192,
                "n_gpu_layers": -1,  # Use GPU if available
                "verbose": False,
                "use_mmap": False, # Disable mmap for clean unload
                "use_mlock": False,
            }

            # Load vision model if it's a VL model and vision is enabled
            if is_vision_model and enable_vision and mmproj_path and mmproj_path != "No mmproj files":
                try:
                    chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                    load_kwargs["chat_handler"] = chat_handler
                    self.clip_model_array = chat_handler
                except Exception as e:
                    print(f"⚠️ Failed to load mmproj: {e}")

            self.model = Llama(**load_kwargs)
            self.current_model_path = model_path
            self.current_mmproj_path = mmproj_path

            load_time = time.time() - load_start
            print(f"✓ Loaded ({load_time:.1f}s)")
            return True

        except Exception as e:
            print(f"⚠️ Load failed: {e}")

            # Clean up all resources on load failure (silently)
            self._free_memory()

            # Check if this is a ggml-related error and raise it for handling
            if self._is_ggml_error(e):
                # Re-raise with ggml indicator for upstream handling
                raise RuntimeError(f"GGML_ERROR: {str(e)}") from e

            return False

    def _remove_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags and their content"""
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        return cleaned_text.strip()

    def _detect_excessive_repetition(self, text: str, max_repeat_ratio: float = 0.3) -> Tuple[bool, str]:
        """Detect if text has excessive repetition

        Returns:
            Tuple of (has_repetition, cleaned_text)
        """
        if not text or len(text) < 100:
            return (False, text)

        # Split into sentences/lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if len(lines) < 3:
            return (False, text)

        # Count line repetitions
        line_counts = {}
        for line in lines:
            if len(line) > 10:  # Only count substantial lines
                line_counts[line] = line_counts.get(line, 0) + 1

        # Check for excessive repetition
        total_lines = len(lines)
        max_repeats = max(line_counts.values()) if line_counts else 0
        repeat_ratio = max_repeats / total_lines if total_lines > 0 else 0

        if repeat_ratio > max_repeat_ratio and max_repeats > 2:
            # Remove duplicate lines, keeping only first occurrence
            seen = set()
            cleaned_lines = []
            for line in lines:
                if line not in seen:
                    cleaned_lines.append(line)
                    seen.add(line)
                elif len(line) <= 10:  # Keep short lines (like separators)
                    cleaned_lines.append(line)

            cleaned_text = '\n'.join(cleaned_lines)
            print(f"⚠️ Repetition detected, cleaned {len(lines)} → {len(cleaned_lines)} lines")
            return (True, cleaned_text)

        return (False, text)

    def _tensor_to_base64(self, image_tensor) -> str:
        """Convert ComfyUI IMAGE tensor to base64 string (single image)"""
        try:
            from PIL import Image

            # ComfyUI IMAGE format: [B, H, W, C] with values in [0, 1]
            # Take first image if batch
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor[0]

            # Convert from [0, 1] to [0, 255]
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

            # Create PIL Image
            pil_image = Image.fromarray(image_np)

            # Convert to JPEG bytes
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95)
            img_bytes = buffered.getvalue()

            # Encode to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            return f"data:image/jpeg;base64,{img_base64}"

        except Exception as e:
            print(f"Error converting image to base64: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _tensors_to_base64_list(self, image_tensor) -> List[str]:
        """Convert ComfyUI IMAGE tensor(s) to list of base64 strings
        
        Args:
            image_tensor: ComfyUI IMAGE tensor [B, H, W, C] or [H, W, C]
            
        Returns:
            List of base64 encoded image URLs
        """
        try:
            from PIL import Image
            
            # Handle single image [H, W, C]
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            # ComfyUI IMAGE format: [B, H, W, C] with values in [0, 1]
            batch_size = image_tensor.shape[0]
            base64_list = []
            
            for i in range(batch_size):
                # Get single image from batch
                single_image = image_tensor[i]
                
                # Convert from [0, 1] to [0, 255]
                image_np = (single_image.cpu().numpy() * 255).astype(np.uint8)
                
                # Create PIL Image
                pil_image = Image.fromarray(image_np)
                
                # Convert to JPEG bytes
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG", quality=95)
                img_bytes = buffered.getvalue()
                
                # Encode to base64
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                base64_list.append(f"data:image/jpeg;base64,{img_base64}")
            
            return base64_list
            
        except Exception as e:
            print(f"Error converting images to base64: {e}")
            import traceback
            traceback.print_exc()
            return None

    def inference(
        self,
        model: str,
        prompt: str,
        prompt_template: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int = 0,
        keep_model_loaded: bool = False,
        mmproj_file: str = "No mmproj files",
        image_1 = None,
        image_2 = None,
        image_3 = None,
        auto_install_llama_cpp: bool = True,
    ) -> Tuple[str, int]:
        """Execute GGUF model inference"""

        # Check if llama-cpp-python is available
        if not self.llama_cpp_available:
            if auto_install_llama_cpp:
                print("llama-cpp-python not found, attempting auto-installation...")
                success, version = self._install_llama_cpp(skip_versions=set(self.failed_versions))
                if success:
                    error_msg = "llama-cpp-python installed successfully!\n\nPlease restart ComfyUI to use the GGUF node."
                else:
                    error_msg = "Failed to install llama-cpp-python.\n\nPlease visit: https://github.com/JamePeng/llama-cpp-python/releases"
            else:
                error_msg = "ERROR: llama-cpp-python is not installed.\n\nPlease either:\n1. Enable 'auto_install_llama_cpp' option (recommended)\n2. Install manually: pip install llama-cpp-python"
            print(error_msg)
            return (error_msg, seed)

        # Check if model is a suggested download
        if model in SUGGESTED_MODELS:
            download_url = SUGGESTED_MODELS[model]
            filename = os.path.basename(download_url)

            # Get clip folder path for downloading
            try:
                clip_paths = folder_paths.get_folder_paths("clip")
                if clip_paths and len(clip_paths) > 0:
                    download_dir = clip_paths[0]
                else:
                    error_msg = "Error: Cannot find clip folder for downloading model."
                    print(error_msg)
                    return (error_msg, seed)
            except:
                error_msg = "Error: Cannot access clip folder for downloading model."
                print(error_msg)
                return (error_msg, seed)

            model_path = os.path.join(download_dir, filename)

            # Check if file already exists
            if not os.path.exists(model_path):
                print(f"Downloading model...")
                if not self._download_file(download_url, model_path):
                    error_msg = f"⚠️ Download failed: {download_url}"
                    return (error_msg, seed)
        else:
            # Get full path from model name
            gguf_files = self._get_gguf_files()
            model_path = None
            for full_path in gguf_files:
                if os.path.basename(full_path) == model:
                    model_path = full_path
                    break

            if model_path is None:
                error_msg = f"Error: Model not found: {model}\nPlease place GGUF files in text_encoders or clip folder."
                print(error_msg)
                return (error_msg, seed)

            if not os.path.exists(model_path):
                error_msg = f"Error: Model file does not exist: {model_path}"
                print(error_msg)
                return (error_msg, seed)

        # Check if this is a vision model
        is_vision_model = self._is_vision_model(model_path)

        # Collect all provided images into a list
        images = []
        if image_1 is not None:
            images.append(image_1)
        if image_2 is not None:
            images.append(image_2)
        if image_3 is not None:
            images.append(image_3)

        # Auto-detect vision mode: enable if any image is provided and model is VL type
        enable_vision = False
        if len(images) > 0 and is_vision_model:
            enable_vision = True
            print(f"📸 Detected {len(images)} image(s) for vision analysis")
        elif len(images) > 0 and not is_vision_model:
            print(f"⚠️ {len(images)} image(s) ignored (model is text-only)")

        # Get mmproj path if vision is enabled
        mmproj_path = None
        if enable_vision:
            if mmproj_file == "No mmproj files":
                print("⚠️ No mmproj file, falling back to text-only mode")
                enable_vision = False
            else:
                # Check if mmproj is a suggested download
                if mmproj_file in SUGGESTED_MMPROJ:
                    download_url = SUGGESTED_MMPROJ[mmproj_file]
                    filename = os.path.basename(download_url)

                    # Get clip folder path for downloading
                    try:
                        clip_paths = folder_paths.get_folder_paths("clip")
                        if clip_paths and len(clip_paths) > 0:
                            download_dir = clip_paths[0]
                        else:
                            print("⚠️ Cannot find clip folder")
                            enable_vision = False
                            mmproj_path = None
                    except:
                        print("⚠️ Cannot access clip folder")
                        enable_vision = False
                        mmproj_path = None

                    if enable_vision:
                        mmproj_path = os.path.join(download_dir, filename)

                        # Check if file already exists
                        if not os.path.exists(mmproj_path):
                            print(f"Downloading mmproj...")
                            if not self._download_file(download_url, mmproj_path):
                                print(f"⚠️ Download failed")
                                enable_vision = False
                                mmproj_path = None
                else:
                    # Find full path for mmproj from text_encoders and clip folders
                    mmproj_files = self._get_mmproj_files()
                    for full_path in mmproj_files:
                        if os.path.basename(full_path) == mmproj_file:
                            mmproj_path = full_path
                            break

                    if mmproj_path is None or not os.path.exists(mmproj_path):
                        print(f"⚠️ mmproj not found: {mmproj_file}")
                        enable_vision = False

        # Load model with ggml error handling
        load_result = False
        ggml_error_occurred = False

        try:
            # First attempt to load model
            load_result = self._load_model(model_path, enable_vision, mmproj_path)
        except RuntimeError as e:
            # Check if this is a ggml error
            if "GGML_ERROR" in str(e):
                ggml_error_occurred = True
                print("⚠️ GGML compatibility error detected during model loading")
            else:
                # Re-raise if not a ggml error
                raise

        # If loading failed (but not ggml error), try force reload once
        if not load_result and not ggml_error_occurred:
            try:
                print("⚠️ Retrying with force reload...")
                load_result = self._load_model(model_path, enable_vision, mmproj_path, force_reload=True)
            except RuntimeError as e:
                if "GGML_ERROR" in str(e):
                    ggml_error_occurred = True
                    print("⚠️ GGML compatibility error detected during force reload")
                else:
                    raise

        # --- NEW: Clear KV Cache to prevent context overflow on re-runs ---
        try:
             import llama_cpp
             # Attempt to find the low-level context pointer
             ctx_ptr = getattr(self.model, 'ctx', None)
             if not ctx_ptr and hasattr(self.model, '_ctx'):
                 ctx_ptr = getattr(self.model._ctx, 'ctx', self.model._ctx)

             if ctx_ptr:
                 # Strategy A: Modern API (v0.3+)
                 if hasattr(llama_cpp, 'llama_get_memory'):
                     mem = llama_cpp.llama_get_memory(ctx_ptr)
                     llama_cpp.llama_memory_seq_rm(mem, -1, 0, -1)
                 # Strategy B: Legacy API
                 elif hasattr(llama_cpp, 'llama_kv_cache_seq_rm'):
                     llama_cpp.llama_kv_cache_seq_rm(ctx_ptr, -1, 0, -1)
        except Exception as e:
             print(f"Warning: Could not clear KV cache at start: {e}")
        # ---------------------------------------------------------------

        # Handle ggml error with auto-reinstall
        if ggml_error_occurred and auto_install_llama_cpp:
            print("Attempting automatic recovery from GGML error...")
            success, message = self._handle_ggml_error_and_reinstall()
            # Return the message to user (success or failure)
            return (message, seed, [])

        if not load_result:
            # Model loading failed - get the error details from the exception
            if auto_install_llama_cpp:
                # Check for specific error types
                import traceback
                error_trace = traceback.format_exc()

                # Check for WinError
                has_winerror = 'WinError' in error_trace or 'WindowsError' in error_trace

                # Check for ggml.dll error
                has_ggml_dll_error = 'ggml.dll' in error_trace.lower() or 'cannot load library' in error_trace.lower()

                print("=" * 70)
                print("ERROR: Model loading failed!")

                if has_ggml_dll_error:
                    # Special handling for ggml.dll error
                    error_msg = (
                        "ERROR: Cannot load ggml.dll\n\n"
                        "This error typically occurs when:\n"
                        "1. Your CUDA version is incompatible with the installed llama-cpp-python\n"
                        "2. CUDA runtime libraries are missing or outdated\n\n"
                        "Recommended solutions:\n"
                        "1. Update your NVIDIA GPU drivers and CUDA toolkit\n"
                        "2. Or manually install a compatible llama-cpp-python version from:\n"
                        "   https://github.com/JamePeng/llama-cpp-python/releases\n\n"
                        "The auto-installer will now try to find an older compatible version..."
                    )
                    print(error_msg)
                    print("=" * 70)

                # Get current installed version info
                try:
                    import pkg_resources
                    current_version = pkg_resources.get_distribution("llama-cpp-python").version
                    print(f"Current llama-cpp-python version: {current_version}")
                except:
                    current_version = None

                if has_winerror or has_ggml_dll_error:
                    print("Detected compatibility issue - will try previous version")
                    print("Attempting to uninstall and install previous version...")
                else:
                    print("This may indicate llama-cpp-python is incompatible or corrupted.")
                    print("Attempting to uninstall and reinstall llama-cpp-python...")
                print("=" * 70)

                # Try to uninstall and reinstall with previous version
                if self._uninstall_llama_cpp():
                    print("\nNow attempting to install compatible version...")

                    # Try to install, skipping failed versions
                    success, new_version = self._install_llama_cpp(skip_versions=set(self.failed_versions))

                    if success and new_version:
                        # Track this version if it was just installed
                        if new_version not in self.failed_versions:
                            self.failed_versions.append(new_version)

                        error_msg = f"llama-cpp-python has been reinstalled (version: {new_version})!\n\nIMPORTANT: Please restart ComfyUI to use the new version."
                        print("=" * 70)
                        print(error_msg)
                        print("=" * 70)
                        return (error_msg, seed)
                    else:
                        # Installation failed - add to failed versions if we got a version
                        if new_version and new_version not in self.failed_versions:
                            self.failed_versions.append(new_version)

                        error_msg = (
                            "Failed to install compatible llama-cpp-python version.\n\n"
                            "Please visit: https://github.com/JamePeng/llama-cpp-python/releases\n"
                            "And manually install the appropriate version for your system.\n\n"
                        )
                        if has_ggml_dll_error:
                            error_msg += (
                                "Note: For ggml.dll errors, ensure your CUDA toolkit is up to date:\n"
                                "- Update NVIDIA GPU drivers\n"
                                "- Install latest CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
                            )
                else:
                    error_msg = (
                        "Failed to uninstall llama-cpp-python.\n\n"
                        "Please manually uninstall using: pip uninstall llama-cpp-python\n"
                        "Then enable auto_install_llama_cpp option to reinstall."
                    )
            else:
                error_msg = (
                    "Error: Model loading failed.\n\n"
                    "This may indicate llama-cpp-python is incompatible.\n"
                    "Please enable 'auto_install_llama_cpp' to automatically fix this,\n"
                    "or manually reinstall llama-cpp-python."
                )

            print(error_msg)
            return (error_msg, seed)

        try:
            # Load and apply template
            template_content = ""
            if prompt_template != "Custom":
                template_content = self._load_template_content(prompt_template)

            # If template content exists, replace system_prompt with template
            if template_content:
                system_prompt = template_content

            # Prepare messages
            messages = []
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})

            # Add image(s) if vision is enabled and model supports it
            if enable_vision and is_vision_model and len(images) > 0:
                # Build content with text first
                content = [{"type": "text", "text": prompt}]
                
                # Convert each image to base64 and add to content
                for idx, img in enumerate(images, 1):
                    # Convert single image tensor to base64
                    image_url = self._tensor_to_base64(img)
                    if image_url is None:
                        error_msg = f"Error: Failed to convert image_{idx} to base64 format."
                        print(error_msg)
                        return (error_msg, seed)
                    content.append({"type": "image_url", "image_url": image_url})
                
                # Log number of images being processed
                print(f"🎬 Processing {len(images)} image(s) with vision model")
                
                messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                messages.append({"role": "user", "content": prompt})

            # Run inference (silently)
            inference_start = time.time()

            # Prepare generation parameters
            # Set repeat_penalty internally to prevent repetitive output (1.15 is a good balance)
            gen_params = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": 1.15,  # Internal setting to reduce repetition
                "stop": ["<|im_end|>", "<|endoftext|>", "</s>", "\n\n\n"],  # Common stop sequences
            }

            # Only add seed if it's set (> 0)
            if seed > 0:
                gen_params["seed"] = seed

            response = self.model.create_chat_completion(**gen_params)

            # Extract response text
            response_text = response["choices"][0]["message"]["content"]

            # Remove thinking tags
            response_text = self._remove_thinking_tags(response_text)

            # Detect and clean excessive repetition
            has_repetition, response_text = self._detect_excessive_repetition(response_text)

            inference_time = time.time() - inference_start
            tokens_generated = response["usage"]["completion_tokens"]
            tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0

            print(f"✓ Done ({inference_time:.1f}s, {tokens_generated} tokens, {tokens_per_sec:.0f}t/s)")

            # Always free image-related memory after inference
            if len(images) > 0:
                self._free_image_memory()

            # Unload model if requested
            if not keep_model_loaded:
                self._free_memory()

            # 自動提取提示詞列表（如果輸出包含 JSON 格式的雜誌數據）
            try:
                prompts = extract_prompts_from_json(response_text)
                if prompts:
                    print(f"✅ 自動提取到 {len(prompts)} 個圖片提示詞")
            except Exception as e:
                print(f"ℹ️ 提示詞提取失敗（這是正常的，如果輸出不是雜誌 JSON）: {e}")
                prompts = []

            return (response_text, seed, prompts)

        except Exception as e:
            import traceback
            error_msg = f"⚠️ Inference failed: {str(e)}"
            print(error_msg)

            # Clean up resources on error (silently)
            if len(images) > 0:
                self._free_image_memory()

            # If error suggests memory/loading issue, unload model completely
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['memory', 'load', 'allocation', 'cuda', 'out of memory']):
                self._free_memory()

            return (error_msg, seed, [])  # 錯誤時返回空的提示詞列表
