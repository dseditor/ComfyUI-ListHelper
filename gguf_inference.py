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
            print(f"Downloading: {url}")
            print(f"Destination: {destination}")

            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded * 100 / total_size, 100)
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"Progress: {percent:.1f}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)", end='\r')

            urllib.request.urlretrieve(url, destination, reporthook=report_progress)
            print(f"\nDownload completed: {os.path.basename(destination)}")
            return True
        except Exception as e:
            print(f"\nDownload failed: {e}")
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
                    "default": 4096,
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
                    "tooltip": "Vision model mmproj file (auto-enabled when image is provided and model is VL type)"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image for vision model (auto-enables vision mode for VL models)"
                }),
                "auto_install_llama_cpp": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-install/repair llama-cpp-python (supports all platforms/CUDA versions, prioritizes Basic version for compatibility)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("text", "used_seed",)
    FUNCTION = "inference"
    CATEGORY = "ListHelper"

    def _free_memory(self):
        """Free GPU and system memory"""
        try:
            # Clear model
            if self.model is not None:
                del self.model
                self.model = None

            self.current_model_path = None
            self.current_mmproj_path = None

            if self.clip_model_array is not None:
                del self.clip_model_array
                self.clip_model_array = None

            # Run garbage collection
            gc.collect()

            # Try to free CUDA memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass

            print("Memory freed successfully")
        except Exception as e:
            print(f"Error freeing memory: {e}")

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

    def _find_matching_release(self, releases_list):
        """Find matching release and wheel URL from releases list based on system configuration"""
        if not releases_list:
            return None

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
        # Priority: CUDA Basic > CUDA+AVX2 > CPU Basic > CPU+AVX2
        # Note: Basic versions are prioritized for better compatibility
        #       AVX2 requires CPU support and may cause "Illegal Instruction" errors
        print("=" * 70)
        print("Searching through releases...")
        print("=" * 70)

        # Try each priority level
        priority_filters = []

        if cuda_version and system != 'darwin':
            # Priority 1: CUDA exact version match (Basic - better compatibility)
            priority_filters.append({
                'name': f'CUDA {cuda_version} Basic (exact match)',
                'cuda': cuda_version,
                'avx2': False,
                'exclude_avx2': True,  # Only match releases without AVX2
                'allow_cuda_upgrade': False
            })
            # Priority 2: CUDA compatible version (Basic - allows newer CUDA)
            # CUDA is forward compatible, e.g., cu130 wheels work on cu128 systems
            priority_filters.append({
                'name': f'CUDA {cuda_version}+ Basic (compatible)',
                'cuda': cuda_version,
                'avx2': False,
                'exclude_avx2': True,
                'allow_cuda_upgrade': True
            })
            # Priority 3: CUDA exact version match + AVX2 (requires AVX2 CPU support)
            priority_filters.append({
                'name': f'CUDA {cuda_version} + AVX2 (exact match)',
                'cuda': cuda_version,
                'avx2': True,
                'exclude_avx2': False,
                'allow_cuda_upgrade': False
            })
            # Priority 4: CUDA compatible version + AVX2
            priority_filters.append({
                'name': f'CUDA {cuda_version}+ AVX2 (compatible)',
                'cuda': cuda_version,
                'avx2': True,
                'exclude_avx2': False,
                'allow_cuda_upgrade': True
            })

        if system != 'darwin':
            # Priority 3: CPU Basic version
            priority_filters.append({
                'name': 'CPU Basic',
                'cuda': None,
                'avx2': False,
                'exclude_avx2': True
            })
            # Priority 4: CPU AVX2 version
            priority_filters.append({
                'name': 'CPU + AVX2',
                'cuda': None,
                'avx2': True,
                'exclude_avx2': False
            })
        else:
            # macOS: no AVX2 priority needed
            priority_filters.append({
                'name': 'CPU Basic',
                'cuda': None,
                'avx2': False,
                'exclude_avx2': False
            })

        # Search through releases with priority
        for priority in priority_filters:
            print(f"Trying priority: {priority['name']}")

            for release in releases_list:
                release_info = self._parse_release_info(release)

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

                # Handle exclude_avx2 flag
                if priority.get('exclude_avx2', False):
                    # Only match if release does NOT have AVX2
                    avx2_match = not release_info['has_avx2']
                else:
                    # Match if either AVX2 is not required or release has AVX2
                    avx2_match = (not priority['avx2']) or release_info['has_avx2']

                if cuda_match and avx2_match:
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
                            return download_url

                    # Release matched but no wheel for this Python version
                    print(f"    No wheel found for Python {py_ver} in this release")

        # No matching release found
        print("=" * 70)
        print("No matching release found for your system configuration")
        print("Available releases:")
        for release in releases_list[:5]:  # Show first 5 releases
            release_info = self._parse_release_info(release)
            print(f"  - {release_info['tag_name']} (CUDA: {release_info['cuda_version']}, AVX2: {release_info['has_avx2']})")
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

    def _install_llama_cpp(self):
        """Auto-installation: Auto-detect system and download from GitHub releases

        Supports all platforms (Windows/Linux/macOS) and CUDA versions.
        Prioritizes Basic (non-AVX2) versions for better compatibility.
        """
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
            return False

        # Find matching release and wheel
        wheel_url = self._find_matching_release(releases_list)
        if not wheel_url:
            print("=" * 70)
            print("ERROR: Could not find a matching wheel for your system")
            print("Please visit: https://github.com/JamePeng/llama-cpp-python/releases")
            print("And manually download the appropriate wheel for your system")
            print("=" * 70)
            return False

        # Download and install
        try:
            print("=" * 70)
            print("Installing llama-cpp-python...")
            print("=" * 70)

            subprocess.check_call([
                sys.executable, "-m", "pip", "install", wheel_url
            ])

            print("=" * 70)
            print("SUCCESS: llama-cpp-python installed successfully")
            print("IMPORTANT: Please restart ComfyUI to use the GGUF node")
            print("=" * 70)
            return True

        except Exception as e:
            print("=" * 70)
            print(f"ERROR: Failed to install llama-cpp-python: {e}")
            print("")
            print("Please visit: https://github.com/JamePeng/llama-cpp-python/releases")
            print("=" * 70)
            return False


    def _is_vision_model(self, model_path: str) -> bool:
        """Check if model is a vision model based on filename"""
        model_name = os.path.basename(model_path).lower()
        return 'vl' in model_name

    def _load_model(self, model_path: str, enable_vision: bool = False, mmproj_path: Optional[str] = None) -> bool:
        """Load GGUF model with llama-cpp-python"""
        try:
            # Check if model and mmproj are already loaded
            if (self.model is not None and
                self.current_model_path == model_path and
                self.current_mmproj_path == mmproj_path):
                print(f"Model already loaded: {os.path.basename(model_path)}")
                if mmproj_path:
                    print(f"  with mmproj: {os.path.basename(mmproj_path)}")
                return True

            # Unload previous model if model or mmproj changed
            if self.model is not None:
                if self.current_model_path != model_path:
                    print("Unloading previous model (model changed)...")
                elif self.current_mmproj_path != mmproj_path:
                    print("Unloading previous model (mmproj changed)...")
                self._free_memory()

            if not self.llama_cpp_available:
                return False

            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            # Check if this is a vision model
            is_vision_model = self._is_vision_model(model_path)

            print(f"Loading GGUF model: {os.path.basename(model_path)}")
            if is_vision_model:
                print("  Detected: Vision model (VL)")
            else:
                print("  Detected: Text-only model")

            load_start = time.time()

            # Load model with appropriate settings
            load_kwargs = {
                "model_path": model_path,
                "n_ctx": 8192,
                "n_gpu_layers": -1,  # Use GPU if available
                "verbose": False,
            }

            # Load vision model if it's a VL model and vision is enabled
            if is_vision_model and enable_vision and mmproj_path and mmproj_path != "No mmproj files":
                print(f"Loading vision model with mmproj: {os.path.basename(mmproj_path)}")
                try:
                    chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                    load_kwargs["chat_handler"] = chat_handler
                    self.clip_model_array = chat_handler
                    print("  Vision mode: Enabled")
                except Exception as e:
                    print(f"  WARNING: Failed to load mmproj, falling back to text-only mode: {e}")
                    print("  Vision mode: Disabled (fallback)")
            elif is_vision_model and not enable_vision:
                print("  Vision mode: Disabled (vision not enabled)")
            elif not is_vision_model and enable_vision:
                print("  Vision mode: Ignored (not a vision model)")

            self.model = Llama(**load_kwargs)
            self.current_model_path = model_path
            self.current_mmproj_path = mmproj_path

            load_time = time.time() - load_start
            print(f"Model loaded successfully (Time: {load_time:.2f}s)")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.current_model_path = None
            self.current_mmproj_path = None
            return False

    def _remove_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags and their content"""
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        return cleaned_text.strip()

    def _tensor_to_base64(self, image_tensor) -> str:
        """Convert ComfyUI IMAGE tensor to base64 string"""
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
        image = None,
        auto_install_llama_cpp: bool = True,
    ) -> Tuple[str, int]:
        """Execute GGUF model inference"""

        # Check if llama-cpp-python is available
        if not self.llama_cpp_available:
            if auto_install_llama_cpp:
                print("llama-cpp-python not found, attempting auto-installation...")
                if self._install_llama_cpp():
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
            if os.path.exists(model_path):
                print(f"Model already exists: {filename}")
            else:
                print(f"Downloading suggested model: {model}")
                if not self._download_file(download_url, model_path):
                    error_msg = f"Error: Failed to download model from {download_url}"
                    print(error_msg)
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

        # Auto-detect vision mode: enable if image is provided and model is VL type
        enable_vision = False
        if image is not None and is_vision_model:
            enable_vision = True
            print("=" * 70)
            print("Auto-detected: Vision mode enabled")
            print(f"  - Image input: Provided")
            print(f"  - Model type: VL (Vision-Language)")
            print("=" * 70)
        elif image is not None and not is_vision_model:
            print("=" * 70)
            print("WARNING: Image provided but model is not a VL (Vision-Language) type.")
            print(f"Model: {os.path.basename(model_path)}")
            print("Vision mode will NOT be enabled. Image will be ignored.")
            print("=" * 70)

        # Get mmproj path if vision is enabled
        mmproj_path = None
        if enable_vision:
            if mmproj_file == "No mmproj files":
                print("=" * 70)
                print("WARNING: Vision mode detected but no mmproj file selected.")
                print("Falling back to text-only mode.")
                print("=" * 70)
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
                            print("=" * 70)
                            print("WARNING: Cannot find clip folder for downloading mmproj.")
                            print("Falling back to text-only mode.")
                            print("=" * 70)
                            enable_vision = False
                            mmproj_path = None
                    except:
                        print("=" * 70)
                        print("WARNING: Cannot access clip folder for downloading mmproj.")
                        print("Falling back to text-only mode.")
                        print("=" * 70)
                        enable_vision = False
                        mmproj_path = None

                    if enable_vision:
                        mmproj_path = os.path.join(download_dir, filename)

                        # Check if file already exists
                        if os.path.exists(mmproj_path):
                            print(f"mmproj already exists: {filename}")
                        else:
                            print(f"Downloading suggested mmproj: {mmproj_file}")
                            if not self._download_file(download_url, mmproj_path):
                                print("=" * 70)
                                print(f"WARNING: Failed to download mmproj from {download_url}")
                                print("Falling back to text-only mode.")
                                print("=" * 70)
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
                        print("=" * 70)
                        print(f"WARNING: mmproj file not found: {mmproj_file}")
                        print("Falling back to text-only mode.")
                        print("=" * 70)
                        enable_vision = False

        # Load model
        if not self._load_model(model_path, enable_vision, mmproj_path):
            # Model loading failed
            if auto_install_llama_cpp:
                print("=" * 70)
                print("ERROR: Model loading failed!")
                print("This may indicate llama-cpp-python is incompatible or corrupted.")
                print("Attempting to uninstall and reinstall llama-cpp-python...")
                print("=" * 70)

                # Try to uninstall and reinstall
                if self._uninstall_llama_cpp():
                    print("\nNow attempting to install compatible version...")
                    if self._install_llama_cpp():
                        error_msg = "llama-cpp-python has been reinstalled!\n\nIMPORTANT: Please restart ComfyUI to use the new version."
                        print("=" * 70)
                        print(error_msg)
                        print("=" * 70)
                        return (error_msg, seed)
                    else:
                        error_msg = "Failed to reinstall llama-cpp-python.\n\nPlease visit: https://github.com/JamePeng/llama-cpp-python/releases\nAnd manually install the appropriate version for your system."
                else:
                    error_msg = "Failed to uninstall llama-cpp-python.\n\nPlease manually uninstall using: pip uninstall llama-cpp-python\nThen enable auto_install_llama_cpp option to reinstall."
            else:
                error_msg = "Error: Model loading failed.\n\nThis may indicate llama-cpp-python is incompatible.\nPlease enable 'auto_install_llama_cpp' to automatically fix this,\nor manually reinstall llama-cpp-python."

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

            # Add image if vision is enabled and model supports it
            if enable_vision and is_vision_model and image is not None:
                # Convert image tensor to base64
                image_url = self._tensor_to_base64(image)
                if image_url is None:
                    error_msg = "Error: Failed to convert image to base64 format."
                    print(error_msg)
                    return (error_msg, seed)

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": image_url}
                    ]
                })
                print("Using vision mode with image input")
            else:
                messages.append({"role": "user", "content": prompt})
                if image is not None and not enable_vision:
                    print("Note: Image input provided but vision mode is disabled, ignoring image")

            # Run inference
            print(f"Starting inference (Seed: {seed})...")
            inference_start = time.time()

            # Prepare generation parameters
            gen_params = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }

            # Only add seed if it's set (> 0)
            if seed > 0:
                gen_params["seed"] = seed

            response = self.model.create_chat_completion(**gen_params)

            # Extract response text
            response_text = response["choices"][0]["message"]["content"]

            # Remove thinking tags
            response_text = self._remove_thinking_tags(response_text)

            inference_time = time.time() - inference_start
            tokens_generated = response["usage"]["completion_tokens"]
            tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0

            print(f"Inference completed (Time: {inference_time:.2f}s | Tokens: {tokens_generated} | Speed: {tokens_per_sec:.1f} tokens/s)")

            # Unload model if requested
            if not keep_model_loaded:
                print("Unloading model to free memory...")
                self._free_memory()

            return (response_text, seed)

        except Exception as e:
            import traceback
            error_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (error_msg, seed)
