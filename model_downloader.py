import os
import sys
import re
import subprocess
import folder_paths
from typing import List, Dict, Tuple


class ModelDownloader:
    """
    Model Downloader Node
    Download models to ComfyUI/models folders using S3impleClient
    Supports HuggingFace Hub acceleration (if huggingface_hub is installed)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "download_list": ("STRING", {
                    "multiline": True,
                    "default": "diffusion_models\nhttps://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf\nCLIP\nhttps://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
                    "tooltip": "Download list format:\nFolder name\nURL1\nURL2\n..."
                }),
                "use_s3c": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "使用 S3impleClient 進行加速下載（如停用則使用普通 HTTP 下載）"
                }),
                "use_custom_path": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "使用自訂路徑（如停用則使用 HuggingFace Hub 原始快取路徑）"
                }),
                "chunk_size_mb": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Chunk size per HTTP request (MB)"
                }),
                "max_workers": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Maximum number of parallel download workers"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_message",)
    FUNCTION = "download_models"
    OUTPUT_NODE = True
    CATEGORY = "ListHelper/Tools"

    def download_models(self, download_list: str, use_s3c: bool, use_custom_path: bool,
                       chunk_size_mb: int, max_workers: int) -> Tuple[str]:
        """
        Download models to ComfyUI/models folders

        Args:
            download_list: Download list text
            use_s3c: Use S3impleClient for accelerated download
            use_custom_path: Use custom path (if False, use HuggingFace Hub cache path)
            chunk_size_mb: Chunk size (MB)
            max_workers: Maximum parallel workers

        Returns:
            Status message
        """
        # Ensure S3impleClient is installed (only if use_s3c is enabled)
        s3c_installed = False
        if use_s3c:
            s3c_installed = self._ensure_s3impleclient()
            if not s3c_installed:
                return ("錯誤：無法安裝或匯入 S3impleClient。請手動安裝：pip install s3impleclient",)

        # Parse download list
        download_map = self._parse_download_list(download_list)

        if not download_map:
            return ("錯誤：下載列表格式不正確或為空",)

        # Get ComfyUI models root directory (only if using custom path)
        models_root = None
        if use_custom_path:
            models_root = self._get_models_root()
            print(f"ModelDownloader: Models root directory = {models_root}")

        # Check for duplicate files before download (only if using custom path)
        skipped_files = []
        if use_custom_path:
            for folder_name, urls in download_map.items():
                target_folder = self._get_target_folder(models_root, folder_name)
                for url in urls:
                    filename = self._extract_filename(url)
                    dest_path = os.path.join(target_folder, filename)
                    if os.path.exists(dest_path):
                        skipped_files.append(f"{folder_name}/{filename}")
                        print(f"ModelDownloader: 檔案已存在，跳過：{dest_path}")

            # Remove URLs that point to existing files
            if skipped_files:
                filtered_download_map = {}
                for folder_name, urls in download_map.items():
                    target_folder = self._get_target_folder(models_root, folder_name)
                    filtered_urls = []
                    for url in urls:
                        filename = self._extract_filename(url)
                        dest_path = os.path.join(target_folder, filename)
                        if not os.path.exists(dest_path):
                            filtered_urls.append(url)
                    if filtered_urls:
                        filtered_download_map[folder_name] = filtered_urls
                download_map = filtered_download_map

            if not download_map:
                skip_msg = f"所有檔案已存在。已跳過：\n" + "\n".join(f"  - {f}" for f in skipped_files)
                print(f"ModelDownloader: {skip_msg}")
                return (skip_msg,)

        # Auto-detect if should use HuggingFace patch
        all_hf_urls = all(
            self._is_huggingface_url(url)
            for urls in download_map.values()
            for url in urls
        )

        # Initialize downloader (only if using S3C)
        s3c = None
        if use_s3c:
            try:
                import s3impleclient as s3c

                # Configure download settings
                chunk_size_bytes = chunk_size_mb * 1024 * 1024
                s3c.configure_download(s3c.DownloadConfig(
                    chunk_size=chunk_size_bytes,
                    max_workers=max_workers,
                    timeout=300.0,
                    max_retries=5,
                ))

                print(f"ModelDownloader: S3C 設定完成 - chunk_size={chunk_size_mb}MB, workers={max_workers}")

            except Exception as e:
                return (f"錯誤：無法設定 S3impleClient：{str(e)}",)

        # Auto-detect and apply HuggingFace patch if conditions are met
        hf_available = False
        patch_applied = False
        if all_hf_urls:
            try:
                import huggingface_hub
                hf_available = True

                # Only apply patch if using S3C
                if use_s3c and s3c:
                    try:
                        s3c.patch_huggingface_hub()
                        patch_applied = True
                        print("ModelDownloader: HuggingFace Hub 加速已啟用")
                    except Exception as patch_error:
                        print(f"ModelDownloader: 無法套用 HF patch ({str(patch_error)})，使用標準 HTTP 下載")
                        hf_available = False

            except ImportError:
                print("ModelDownloader: huggingface_hub 未安裝，使用標準 HTTP 下載")
                hf_available = False

        # Start downloading
        results = []
        total_files = sum(len(urls) for urls in download_map.values())
        current_file = 0

        try:
            for folder_name, urls in download_map.items():
                # Get target folder path (only if using custom path)
                target_folder = None
                if use_custom_path:
                    target_folder = self._get_target_folder(models_root, folder_name)
                    # Ensure folder exists
                    os.makedirs(target_folder, exist_ok=True)
                    print(f"\nModelDownloader: 目標資料夾 = {target_folder}")

                for url in urls:
                    current_file += 1
                    filename = self._extract_filename(url)

                    print(f"\n[{current_file}/{total_files}] 正在下載：{filename}")
                    print(f"  URL: {url}")

                    try:
                        # Determine download method based on settings
                        if not use_custom_path and self._is_huggingface_url(url):
                            # Use HuggingFace Hub default cache path
                            if not hf_available:
                                results.append(f"✗ {filename} - 錯誤：需要 huggingface_hub 但未安裝")
                                print(f"  ✗ 下載失敗：需要 huggingface_hub")
                                continue

                            from huggingface_hub import hf_hub_download
                            repo_id, file_path = self._parse_hf_url(url)

                            # Use HF Hub default cache (no local_dir specified)
                            downloaded_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=file_path,
                                force_download=False,
                            )

                            results.append(f"✓ {filename} (HF 快取路徑)")
                            print(f"  ✓ 下載成功至 HF 快取：{downloaded_path}")

                        elif use_custom_path:
                            # Use custom path
                            dest_path = os.path.join(target_folder, filename)
                            print(f"  目標：{dest_path}")

                            # Determine if should use hf_hub_download with custom path
                            use_hf_hub = False
                            if hf_available and patch_applied and self._is_huggingface_url(url):
                                # Check if file path contains subdirectories
                                repo_id, file_path = self._parse_hf_url(url)

                                # Only use hf_hub_download if file is at root level (no subdirs)
                                # This prevents hf_hub_download from creating unwanted subdirectories
                                if '/' not in file_path:
                                    use_hf_hub = True

                            if use_hf_hub:
                                # Use hf_hub_download with custom path
                                from huggingface_hub import hf_hub_download
                                downloaded_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename=file_path,
                                    force_download=False,
                                    local_dir=target_folder,
                                    local_dir_use_symlinks=False,
                                )

                                results.append(f"✓ {filename} (HF 加速)")
                                print(f"  ✓ 下載成功（HF 加速）")

                            elif use_s3c and s3c:
                                # Use S3impleClient download
                                result = s3c.download(
                                    url=url,
                                    dest=dest_path,
                                )

                                if result.success:
                                    size_mb = result.total_bytes / (1024 * 1024)
                                    method = "(S3C + HF patch)" if patch_applied else "(S3C)"
                                    results.append(f"✓ {filename} ({size_mb:.2f} MB) {method}")
                                    print(f"  ✓ 下載成功 - {size_mb:.2f} MB {method}")
                                else:
                                    results.append(f"✗ {filename} - 失敗")
                                    print(f"  ✗ 下載失敗")

                            else:
                                # Use plain HTTP download
                                success = self._download_http(url, dest_path)
                                if success:
                                    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                                    results.append(f"✓ {filename} ({size_mb:.2f} MB) (HTTP)")
                                    print(f"  ✓ 下載成功 - {size_mb:.2f} MB (HTTP)")
                                else:
                                    results.append(f"✗ {filename} - HTTP 下載失敗")
                                    print(f"  ✗ HTTP 下載失敗")

                        else:
                            # Non-HF URL without custom path - error
                            results.append(f"✗ {filename} - 錯誤：非 HF URL 需要使用自訂路徑")
                            print(f"  ✗ 錯誤：非 HuggingFace URL 必須啟用自訂路徑")

                    except Exception as e:
                        error_msg = f"✗ {filename} - 錯誤：{str(e)}"
                        results.append(error_msg)
                        print(f"  {error_msg}")

        finally:
            # Restore HF patch
            if patch_applied:
                try:
                    s3c.unpatch_huggingface_hub()
                    print("\nModelDownloader: HuggingFace Hub 原始設定已還原")
                except Exception as unpatch_error:
                    print(f"\nModelDownloader: 無法還原 HF patch ({str(unpatch_error)})")

        # Generate final status message
        success_count = sum(1 for r in results if r.startswith("✓"))
        fail_count = sum(1 for r in results if r.startswith("✗"))

        # Include skipped files in status message
        status_lines = [f"下載完成！"]
        if skipped_files:
            status_lines.append(f"已跳過（檔案已存在）：{len(skipped_files)}")
        status_lines.append(f"成功：{success_count}/{total_files}")
        status_lines.append(f"失敗：{fail_count}")
        status_lines.append("")
        status_lines.append("詳細資訊：")
        if skipped_files:
            status_lines.append("  已跳過的檔案：")
            for f in skipped_files:
                status_lines.append(f"    ○ {f}")
        status_lines.extend(f"  {r}" for r in results)

        status_msg = "\n".join(status_lines)

        print(f"\n{status_msg}")
        return (status_msg,)

    def _ensure_s3impleclient(self) -> bool:
        """Ensure S3impleClient is installed"""
        try:
            import s3impleclient
            print("ModelDownloader: S3impleClient is installed")
            return True
        except ImportError:
            print("ModelDownloader: S3impleClient not installed, attempting auto-install...")

            try:
                python_exe = sys.executable
                install_cmd = [python_exe, "-m", "pip", "install", "s3impleclient"]

                print(f"ModelDownloader: Running install command: {' '.join(install_cmd)}")

                result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:
                    print("ModelDownloader: S3impleClient installation successful!")
                    print(result.stdout)

                    # Try to re-import
                    try:
                        import s3impleclient
                        return True
                    except ImportError:
                        print("ModelDownloader: Installation succeeded but import failed, may need to restart ComfyUI")
                        return False
                else:
                    print(f"ModelDownloader: Installation failed\nStdout: {result.stdout}\nStderr: {result.stderr}")
                    return False

            except Exception as e:
                print(f"ModelDownloader: Auto-installation failed: {str(e)}")
                return False

    def _parse_download_list(self, text: str) -> Dict[str, List[str]]:
        """
        Parse download list

        Format:
        Folder name
        URL1
        URL2
        Folder name2
        URL3

        Returns:
            {folder_name: [URL1, URL2, ...]}
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        download_map = {}
        current_folder = None

        for line in lines:
            # Check if line is a URL
            if line.startswith('http://') or line.startswith('https://'):
                if current_folder:
                    download_map[current_folder].append(line)
                else:
                    print(f"Warning: URL {line} has no corresponding folder name, skipped")
            else:
                # This is a folder name
                current_folder = line
                if current_folder not in download_map:
                    download_map[current_folder] = []

        print(f"ModelDownloader: Parsed {len(download_map)} folders")
        for folder, urls in download_map.items():
            print(f"  {folder}: {len(urls)} files")

        return download_map

    def _get_models_root(self) -> str:
        """Get ComfyUI models root directory"""
        # Use folder_paths to get models directory
        # Usually ComfyUI/models
        try:
            # Try to get any folder path, then find models directory
            checkpoints_path = folder_paths.get_folder_paths("checkpoints")
            if checkpoints_path:
                # checkpoints is usually in models/checkpoints
                models_root = os.path.dirname(checkpoints_path[0])
                return models_root
        except:
            pass

        # Fallback: navigate up from current project
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        models_path = os.path.join(current_dir, "models")

        if os.path.exists(models_path):
            return models_path

        # Final fallback: create new one
        os.makedirs(models_path, exist_ok=True)
        return models_path

    def _get_target_folder(self, models_root: str, folder_name: str) -> str:
        """
        Get target folder path
        Supports multi-level paths (e.g., "clip/split_files")
        Base folder name is case-insensitive
        """
        # Split folder path into parts
        # Support both forward slash and backslash
        folder_parts = folder_name.replace('\\', '/').split('/')

        # Get base folder (first part)
        base_folder = folder_parts[0]
        base_folder_lower = base_folder.lower()

        # Check if base folder already exists (case-insensitive)
        actual_base_folder = base_folder
        if os.path.exists(models_root):
            for existing_folder in os.listdir(models_root):
                existing_path = os.path.join(models_root, existing_folder)
                if os.path.isdir(existing_path) and existing_folder.lower() == base_folder_lower:
                    actual_base_folder = existing_folder
                    print(f"  Found existing base folder: {existing_folder}")
                    break

        # Construct full path with all parts
        if len(folder_parts) > 1:
            # Multi-level path: models_root/base_folder/sub1/sub2/...
            full_path = os.path.join(models_root, actual_base_folder, *folder_parts[1:])
        else:
            # Single level: models_root/base_folder
            full_path = os.path.join(models_root, actual_base_folder)

        return full_path

    def _is_huggingface_url(self, url: str) -> bool:
        """Check if URL is a HuggingFace URL"""
        return 'huggingface.co' in url.lower()

    def _parse_hf_url(self, url: str) -> Tuple[str, str]:
        """
        Parse HuggingFace URL

        Example: https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf
        Returns: ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q8_0.gguf")
        """
        # Match HuggingFace URL format
        # https://huggingface.co/{org}/{repo}/resolve/{branch}/{filepath}
        pattern = r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)'
        match = re.search(pattern, url)

        if match:
            repo_id = match.group(1)
            file_path = match.group(2)
            return repo_id, file_path

        # If unable to parse, raise error
        raise ValueError(f"Unable to parse HuggingFace URL: {url}")

    def _extract_filename(self, url: str) -> str:
        """Extract filename from URL"""
        # Remove query parameters
        url = url.split('?')[0]

        # Get last path segment
        filename = url.split('/')[-1]

        return filename

    def _download_http(self, url: str, dest_path: str) -> bool:
        """
        Download file using plain HTTP (urllib)

        Args:
            url: URL to download
            dest_path: Destination file path

        Returns:
            True if successful, False otherwise
        """
        try:
            import urllib.request
            import shutil

            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Download with progress indication
            print(f"  正在使用 HTTP 下載...")

            with urllib.request.urlopen(url, timeout=300) as response:
                total_size = int(response.headers.get('Content-Length', 0))

                # Write to temporary file first
                temp_path = dest_path + '.tmp'

                with open(temp_path, 'wb') as out_file:
                    downloaded = 0
                    chunk_size = 8192
                    last_progress = 0

                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        out_file.write(chunk)
                        downloaded += len(chunk)

                        # Print progress every 10%
                        if total_size > 0:
                            progress = int(downloaded * 100 / total_size)
                            if progress >= last_progress + 10:
                                print(f"  進度：{progress}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                                last_progress = progress

                # Move temp file to final destination
                shutil.move(temp_path, dest_path)

            return True

        except Exception as e:
            print(f"  HTTP 下載錯誤：{str(e)}")
            # Clean up temp file if it exists
            temp_path = dest_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ModelDownloader": ModelDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelDownloader": "Model Downloader",
}
