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
    CATEGORY = "ListHelper"

    def download_models(self, download_list: str, chunk_size_mb: int, max_workers: int) -> Tuple[str]:
        """
        Download models to ComfyUI/models folders

        Args:
            download_list: Download list text
            chunk_size_mb: Chunk size (MB)
            max_workers: Maximum parallel workers

        Returns:
            Status message
        """
        # Ensure S3impleClient is installed
        s3c_installed = self._ensure_s3impleclient()
        if not s3c_installed:
            return ("Error: Failed to install or import S3impleClient. Please install manually: pip install s3impleclient",)

        # Parse download list
        download_map = self._parse_download_list(download_list)

        if not download_map:
            return ("Error: Download list format is incorrect or empty",)

        # Get ComfyUI models root directory
        models_root = self._get_models_root()
        print(f"ModelDownloader: Models root directory = {models_root}")

        # Check for duplicate files before download
        skipped_files = []
        for folder_name, urls in download_map.items():
            target_folder = self._get_target_folder(models_root, folder_name)
            for url in urls:
                filename = self._extract_filename(url)
                dest_path = os.path.join(target_folder, filename)
                if os.path.exists(dest_path):
                    skipped_files.append(f"{folder_name}/{filename}")
                    print(f"ModelDownloader: File already exists, skipping: {dest_path}")

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
            skip_msg = f"All files already exist. Skipped:\n" + "\n".join(f"  - {f}" for f in skipped_files)
            print(f"ModelDownloader: {skip_msg}")
            return (skip_msg,)

        # Auto-detect if should use HuggingFace patch
        all_hf_urls = all(
            self._is_huggingface_url(url)
            for urls in download_map.values()
            for url in urls
        )

        # Initialize downloader
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

            print(f"ModelDownloader: Configuration complete - chunk_size={chunk_size_mb}MB, workers={max_workers}")

        except Exception as e:
            return (f"Error: Failed to configure S3impleClient: {str(e)}",)

        # Auto-detect and apply HuggingFace patch if conditions are met
        hf_available = False
        patch_applied = False
        if all_hf_urls:
            try:
                import huggingface_hub
                hf_available = True

                # Try to apply patch, but continue with HTTP if it fails
                try:
                    s3c.patch_huggingface_hub()
                    patch_applied = True
                    print("ModelDownloader: HuggingFace Hub acceleration enabled")
                except Exception as patch_error:
                    print(f"ModelDownloader: Failed to apply HF patch ({str(patch_error)}), using standard HTTP download")
                    hf_available = False

            except ImportError:
                print("ModelDownloader: huggingface_hub not installed, using standard HTTP download")
                hf_available = False

        # Start downloading
        results = []
        total_files = sum(len(urls) for urls in download_map.values())
        current_file = 0

        try:
            for folder_name, urls in download_map.items():
                # Get target folder path
                target_folder = self._get_target_folder(models_root, folder_name)

                # Ensure folder exists
                os.makedirs(target_folder, exist_ok=True)
                print(f"\nModelDownloader: Target folder = {target_folder}")

                for url in urls:
                    current_file += 1
                    filename = self._extract_filename(url)
                    dest_path = os.path.join(target_folder, filename)

                    print(f"\n[{current_file}/{total_files}] Downloading: {filename}")
                    print(f"  URL: {url}")
                    print(f"  Destination: {dest_path}")

                    try:
                        # Determine download method
                        use_hf_hub = False
                        if hf_available and patch_applied and self._is_huggingface_url(url):
                            # Check if file path contains subdirectories
                            repo_id, file_path = self._parse_hf_url(url)

                            # Only use hf_hub_download if file is at root level (no subdirs)
                            # This prevents hf_hub_download from creating unwanted subdirectories
                            if '/' not in file_path:
                                use_hf_hub = True

                        if use_hf_hub:
                            # Use hf_hub_download (only for root-level files)
                            from huggingface_hub import hf_hub_download
                            downloaded_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=file_path,
                                force_download=False,
                                local_dir=target_folder,
                                local_dir_use_symlinks=False,
                            )

                            results.append(f"✓ {filename} (HF accelerated)")
                            print(f"  ✓ Download successful (with HF acceleration)")

                        else:
                            # Use S3impleClient direct download
                            # (Also accelerated if HF patch is applied)
                            result = s3c.download(
                                url=url,
                                dest=dest_path,
                            )

                            if result.success:
                                size_mb = result.total_bytes / (1024 * 1024)
                                method = "(S3C + HF patch)" if patch_applied else "(S3C)"
                                results.append(f"✓ {filename} ({size_mb:.2f} MB) {method}")
                                print(f"  ✓ Download successful - {size_mb:.2f} MB {method}")
                            else:
                                results.append(f"✗ {filename} - Failed")
                                print(f"  ✗ Download failed")

                    except Exception as e:
                        error_msg = f"✗ {filename} - Error: {str(e)}"
                        results.append(error_msg)
                        print(f"  {error_msg}")

        finally:
            # Restore HF patch
            if patch_applied:
                try:
                    s3c.unpatch_huggingface_hub()
                    print("\nModelDownloader: HuggingFace Hub original settings restored")
                except Exception as unpatch_error:
                    print(f"\nModelDownloader: Failed to unpatch HF ({str(unpatch_error)})")

        # Generate final status message
        success_count = sum(1 for r in results if r.startswith("✓"))
        fail_count = sum(1 for r in results if r.startswith("✗"))

        # Include skipped files in status message
        status_lines = [f"Download complete!"]
        if skipped_files:
            status_lines.append(f"Skipped (already exists): {len(skipped_files)}")
        status_lines.append(f"Success: {success_count}/{total_files}")
        status_lines.append(f"Failed: {fail_count}")
        status_lines.append("")
        status_lines.append("Details:")
        if skipped_files:
            status_lines.append("  Skipped files:")
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


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ModelDownloader": ModelDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelDownloader": "Model Downloader",
}
