import os
import re
import shutil
import json
import folder_paths
from typing import List, Dict, Tuple, Optional


class ModelDownloader:
    """
    Model Downloader Node
    下載模型到 ComfyUI/models 資料夾
    支援 HuggingFace Hub 加速下載（需安裝 huggingface_hub，可選 s3impleclient 加速）
    """

    # 範本配置檔路徑
    TEMPLATES_FILE = os.path.join(os.path.dirname(__file__), "model_templates.json")

    @classmethod
    def _load_template_names(cls) -> List[str]:
        """載入範本名稱列表"""
        template_names = ["無"]  # 預設選項
        try:
            if os.path.exists(cls.TEMPLATES_FILE):
                with open(cls.TEMPLATES_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    templates = data.get("templates", {})
                    template_names.extend(templates.keys())
        except Exception as e:
            print(f"ModelDownloader: 載入範本列表失敗：{e}")
        return template_names

    @classmethod
    def INPUT_TYPES(cls):
        template_names = cls._load_template_names()
        return {
            "required": {
                "download_list": ("STRING", {
                    "multiline": True,
                    "default": "diffusion_models\nhttps://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf\nCLIP\nhttps://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
                    "tooltip": "下載列表格式：\n資料夾名稱\nURL1\nURL2\n...\n\n注意：若選擇了模型範本，此輸入將被忽略。"
                }),
                "use_s3c": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用 S3impleClient 加速下載（需安裝 s3impleclient）。\n僅在開啟 HF 下載時有效。"
                }),
                "use_hf_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用 HuggingFace Hub 下載。\n關閉時使用普通 HTTP 下載（相容性最高）。\n開啟時檔案會先下載到 HF 快取，再自動搬移到 ComfyUI models 資料夾。"
                }),
                "chunk_size_mb": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "每次 HTTP 請求的區塊大小 (MB)"
                }),
                "max_workers": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "最大平行下載執行緒數"
                }),
                "model_template": (template_names, {
                    "default": "None",
                    "tooltip": "選擇模型範本。選擇範本後將使用範本中的下載列表，忽略上方手動輸入的內容。"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_message",)
    FUNCTION = "download_models"
    OUTPUT_NODE = True
    CATEGORY = "ListHelper/Tools"

    def download_models(self, download_list: str, use_s3c: bool, use_hf_download: bool,
                        chunk_size_mb: int, max_workers: int, model_template: str) -> Tuple[str]:
        """
        下載模型到 ComfyUI/models 資料夾

        Args:
            download_list: 下載列表文字
            use_s3c: 使用 S3impleClient 加速（僅在 HF 下載時有效）
            use_hf_download: 使用 HuggingFace Hub 下載（開啟時會自動搬移到 models 資料夾）
            chunk_size_mb: 區塊大小 (MB)
            max_workers: 最大平行執行緒數
            model_template: 模型範本名稱

        Returns:
            狀態訊息
        """
        # 若選擇了範本，使用範本的下載列表
        if model_template and model_template != "無":
            template_download_list = self._get_template_download_list(model_template)
            if template_download_list:
                download_list = template_download_list
                print(f"ModelDownloader: 使用範本「{model_template}」的下載列表")
            else:
                return (f"錯誤：無法載入範本「{model_template}」",)

        # 解析下載列表
        download_map = self._parse_download_list(download_list)
        if not download_map:
            return ("錯誤：下載列表格式不正確或為空",)

        # 取得 ComfyUI models 根目錄
        models_root = self._get_models_root()
        print(f"ModelDownloader: Models 根目錄 = {models_root}")

        # 檢查相依套件
        hf_available = self._check_huggingface_hub()
        s3c_available = self._check_s3impleclient()

        if use_hf_download and not hf_available:
            return ("錯誤：使用 HF 下載需要安裝 huggingface_hub。\n"
                    "請執行：pip install huggingface_hub\n"
                    "或關閉「使用 HF 下載」選項使用普通 HTTP 下載。",)

        # 設定 S3C（如果開啟 use_s3c、可用且使用 HF 下載）
        s3c = None
        patch_applied = False
        if use_hf_download and use_s3c and s3c_available:
            s3c, patch_applied = self._setup_s3c(chunk_size_mb, max_workers)
        elif use_s3c and not s3c_available:
            print("ModelDownloader: 警告 - use_s3c 已開啟但 s3impleclient 未安裝")

        # 檢查檔案是否已存在且完整
        files_to_download, skipped_files, incomplete_files = self._check_existing_files(
            download_map, models_root
        )

        if not files_to_download:
            skip_msg = "所有檔案已存在且完整。\n已跳過：\n" + "\n".join(f"  - {f}" for f in skipped_files)
            print(f"ModelDownloader: {skip_msg}")
            return (skip_msg,)

        # 開始下載
        results = []
        total_files = sum(len(urls) for urls in files_to_download.values())
        current_file = 0

        try:
            for folder_name, urls in files_to_download.items():
                target_folder = self._get_target_folder(models_root, folder_name)
                os.makedirs(target_folder, exist_ok=True)
                print(f"\nModelDownloader: 目標資料夾 = {target_folder}")

                for url in urls:
                    current_file += 1
                    filename = self._extract_filename(url)
                    dest_path = os.path.join(target_folder, filename)

                    print(f"\n[{current_file}/{total_files}] 正在下載：{filename}")
                    print(f"  URL: {url}")
                    print(f"  目標：{dest_path}")

                    try:
                        if use_hf_download and self._is_huggingface_url(url):
                            # 使用 HF Hub 下載，然後搬移到目標資料夾
                            result = self._download_with_hf_hub(url, dest_path, s3c, patch_applied)
                        else:
                            # 使用普通 HTTP 下載
                            result = self._download_with_http(url, dest_path)

                        results.append(result)

                    except Exception as e:
                        error_msg = f"✗ {filename} - 錯誤：{str(e)}"
                        results.append(error_msg)
                        print(f"  {error_msg}")

        finally:
            # 還原 HF patch
            if patch_applied and s3c:
                try:
                    s3c.unpatch_huggingface_hub()
                    print("\nModelDownloader: HuggingFace Hub 原始設定已還原")
                except Exception as e:
                    print(f"\nModelDownloader: 無法還原 HF patch ({str(e)})")

        # 產生最終狀態訊息
        return self._generate_status_message(results, skipped_files, incomplete_files, total_files)

    # ==================== 範本處理 ====================

    def _get_template_download_list(self, template_name: str) -> Optional[str]:
        """
        從範本配置檔取得下載列表

        Args:
            template_name: 範本名稱

        Returns:
            下載列表文字格式，如果範本不存在則返回 None
        """
        try:
            if not os.path.exists(self.TEMPLATES_FILE):
                print(f"ModelDownloader: 範本檔案不存在：{self.TEMPLATES_FILE}")
                return None

            with open(self.TEMPLATES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            templates = data.get("templates", {})
            if template_name not in templates:
                print(f"ModelDownloader: 找不到範本「{template_name}」")
                return None

            template = templates[template_name]
            download_list_data = template.get("download_list", [])

            # 將 JSON 格式轉換為文字格式
            lines = []
            for item in download_list_data:
                folder = item.get("folder", "")
                urls = item.get("urls", [])
                if folder and urls:
                    lines.append(folder)
                    lines.extend(urls)

            download_list_text = "\n".join(lines)
            print(f"ModelDownloader: 範本「{template_name}」包含 {len(download_list_data)} 個資料夾")
            return download_list_text

        except json.JSONDecodeError as e:
            print(f"ModelDownloader: 範本檔案格式錯誤：{e}")
            return None
        except Exception as e:
            print(f"ModelDownloader: 載入範本失敗：{e}")
            return None

    # ==================== 相依套件檢查 ====================

    def _check_huggingface_hub(self) -> bool:
        """檢查 huggingface_hub 是否已安裝"""
        try:
            import huggingface_hub
            print("ModelDownloader: huggingface_hub 已安裝")
            return True
        except ImportError:
            print("ModelDownloader: huggingface_hub 未安裝")
            return False

    def _check_s3impleclient(self) -> bool:
        """檢查 s3impleclient 是否已安裝"""
        try:
            import s3impleclient
            print("ModelDownloader: s3impleclient 已安裝")
            return True
        except ImportError:
            print("ModelDownloader: s3impleclient 未安裝（可選，用於加速下載）")
            return False

    def _setup_s3c(self, chunk_size_mb: int, max_workers: int) -> Tuple[Optional[object], bool]:
        """
        設定 S3impleClient 並套用 HF patch

        Returns:
            (s3c module, patch_applied)
        """
        try:
            import s3impleclient as s3c

            # 設定下載參數
            chunk_size_bytes = chunk_size_mb * 1024 * 1024
            s3c.configure_download(s3c.DownloadConfig(
                chunk_size=chunk_size_bytes,
                max_workers=max_workers,
                timeout=300.0,
                max_retries=5,
            ))
            print(f"ModelDownloader: S3C 設定完成 - chunk_size={chunk_size_mb}MB, workers={max_workers}")

            # 套用 HF patch
            try:
                s3c.patch_huggingface_hub()
                print("ModelDownloader: HuggingFace Hub 加速已啟用（S3C patch 已套用）")
                return s3c, True
            except Exception as e:
                print(f"ModelDownloader: 警告 - 無法套用 HF patch ({str(e)})")
                return s3c, False

        except Exception as e:
            print(f"ModelDownloader: 警告 - S3C 設定失敗 ({str(e)})")
            return None, False

    # ==================== 檔案檢查 ====================

    def _check_existing_files(self, download_map: Dict[str, List[str]],
                              models_root: str) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
        """
        檢查檔案是否已存在且完整

        透過比對遠端檔案大小來判斷檔案是否完整，
        避免下載中斷後的不完整檔案被誤判為已存在。

        Returns:
            (files_to_download, skipped_files, incomplete_files)
        """
        files_to_download = {}
        skipped_files = []
        incomplete_files = []

        for folder_name, urls in download_map.items():
            target_folder = self._get_target_folder(models_root, folder_name)
            urls_to_download = []

            for url in urls:
                filename = self._extract_filename(url)
                dest_path = os.path.join(target_folder, filename)

                if os.path.exists(dest_path):
                    # 檔案存在，檢查大小是否與遠端一致
                    local_size = os.path.getsize(dest_path)
                    remote_size = self._get_remote_file_size(url)

                    if remote_size is not None and local_size != remote_size:
                        # 檔案大小不一致，需要重新下載
                        incomplete_files.append(f"{folder_name}/{filename} (本地: {local_size / (1024*1024):.2f}MB, 遠端: {remote_size / (1024*1024):.2f}MB)")
                        print(f"ModelDownloader: 檔案不完整，將重新下載：{dest_path}")
                        print(f"  本地大小：{local_size} bytes, 遠端大小：{remote_size} bytes")
                        # 刪除不完整的檔案
                        try:
                            os.remove(dest_path)
                        except Exception as e:
                            print(f"  警告：無法刪除不完整檔案：{e}")
                        urls_to_download.append(url)
                    elif remote_size is None:
                        # 無法取得遠端大小，假設檔案完整
                        skipped_files.append(f"{folder_name}/{filename} (無法驗證)")
                        print(f"ModelDownloader: 檔案已存在（無法驗證完整性），跳過：{dest_path}")
                    else:
                        # 檔案大小一致，跳過
                        skipped_files.append(f"{folder_name}/{filename}")
                        print(f"ModelDownloader: 檔案已存在且完整，跳過：{dest_path}")
                else:
                    urls_to_download.append(url)

            if urls_to_download:
                files_to_download[folder_name] = urls_to_download

        return files_to_download, skipped_files, incomplete_files

    def _get_remote_file_size(self, url: str) -> Optional[int]:
        """
        取得遠端檔案大小（透過 HEAD 請求）

        Returns:
            檔案大小（bytes），如果無法取得則返回 None
        """
        try:
            import urllib.request

            request = urllib.request.Request(url, method='HEAD')
            request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

            with urllib.request.urlopen(request, timeout=30) as response:
                content_length = response.headers.get('Content-Length')
                if content_length:
                    return int(content_length)
        except Exception as e:
            print(f"ModelDownloader: 無法取得遠端檔案大小：{e}")

        return None

    # ==================== 下載方法 ====================

    def _download_with_hf_hub(self, url: str, dest_path: str,
                              s3c: Optional[object], patch_applied: bool) -> str:
        """
        使用 HuggingFace Hub 下載，然後搬移到目標資料夾

        Args:
            url: HuggingFace URL
            dest_path: 最終目標路徑
            s3c: s3impleclient module（如果可用）
            patch_applied: S3C patch 是否已套用

        Returns:
            結果訊息
        """
        from huggingface_hub import hf_hub_download

        filename = self._extract_filename(url)
        repo_id, file_path = self._parse_hf_url(url)

        print(f"  使用 HF Hub 下載...")
        print(f"  Repo: {repo_id}, File: {file_path}")

        # 使用 HF Hub 下載到快取
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            force_download=False,
        )

        print(f"  HF 快取路徑：{downloaded_path}")

        # 搬移到目標資料夾
        if downloaded_path != dest_path:
            print(f"  搬移檔案到：{dest_path}")
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # 複製檔案（保留 HF 快取中的原始檔案）
            shutil.copy2(downloaded_path, dest_path)

            # 驗證複製成功
            if os.path.exists(dest_path):
                size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                method = "HF Hub + S3C 加速" if patch_applied else "HF Hub"
                result = f"✓ {filename} ({size_mb:.2f} MB) ({method})"
                print(f"  ✓ 下載並搬移成功 - {size_mb:.2f} MB")
                return result
            else:
                result = f"✗ {filename} - 搬移失敗"
                print(f"  ✗ 搬移失敗")
                return result
        else:
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            method = "HF Hub + S3C 加速" if patch_applied else "HF Hub"
            result = f"✓ {filename} ({size_mb:.2f} MB) ({method})"
            print(f"  ✓ 下載成功 - {size_mb:.2f} MB")
            return result

    def _download_with_http(self, url: str, dest_path: str) -> str:
        """
        使用普通 HTTP 下載

        Args:
            url: 下載 URL
            dest_path: 目標路徑

        Returns:
            結果訊息
        """
        import urllib.request

        filename = self._extract_filename(url)
        print(f"  使用 HTTP 下載...")

        # 確保目標資料夾存在
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # 使用暫存檔案下載
        temp_path = dest_path + '.tmp'

        try:
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

            with urllib.request.urlopen(request, timeout=300) as response:
                total_size = int(response.headers.get('Content-Length', 0))

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

                        # 每 10% 顯示進度
                        if total_size > 0:
                            progress = int(downloaded * 100 / total_size)
                            if progress >= last_progress + 10:
                                print(f"  進度：{progress}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                                last_progress = progress

            # 下載完成，搬移到最終位置
            shutil.move(temp_path, dest_path)

            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            result = f"✓ {filename} ({size_mb:.2f} MB) (HTTP)"
            print(f"  ✓ 下載成功 - {size_mb:.2f} MB")
            return result

        except Exception as e:
            # 清理暫存檔案
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise e

    # ==================== 工具方法 ====================

    def _parse_download_list(self, text: str) -> Dict[str, List[str]]:
        """
        解析下載列表

        格式：
        資料夾名稱
        URL1
        URL2
        資料夾名稱2
        URL3

        Returns:
            {folder_name: [URL1, URL2, ...]}
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        download_map = {}
        current_folder = None

        for line in lines:
            if line.startswith('http://') or line.startswith('https://'):
                if current_folder:
                    download_map[current_folder].append(line)
                else:
                    print(f"警告：URL {line} 沒有對應的資料夾名稱，已跳過")
            else:
                current_folder = line
                if current_folder not in download_map:
                    download_map[current_folder] = []

        print(f"ModelDownloader: 解析到 {len(download_map)} 個資料夾")
        for folder, urls in download_map.items():
            print(f"  {folder}: {len(urls)} 個檔案")

        return download_map

    def _get_models_root(self) -> str:
        """取得 ComfyUI models 根目錄"""
        try:
            checkpoints_path = folder_paths.get_folder_paths("checkpoints")
            if checkpoints_path:
                return os.path.dirname(checkpoints_path[0])
        except:
            pass

        # 備用方案
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        models_path = os.path.join(current_dir, "models")

        if not os.path.exists(models_path):
            os.makedirs(models_path, exist_ok=True)

        return models_path

    def _get_target_folder(self, models_root: str, folder_name: str) -> str:
        """
        取得目標資料夾路徑
        支援多層路徑（例如 "clip/split_files"）
        基礎資料夾名稱不區分大小寫
        """
        folder_parts = folder_name.replace('\\', '/').split('/')
        base_folder = folder_parts[0]
        base_folder_lower = base_folder.lower()

        # 尋找已存在的資料夾（不區分大小寫）
        actual_base_folder = base_folder
        if os.path.exists(models_root):
            for existing_folder in os.listdir(models_root):
                existing_path = os.path.join(models_root, existing_folder)
                if os.path.isdir(existing_path) and existing_folder.lower() == base_folder_lower:
                    actual_base_folder = existing_folder
                    break

        if len(folder_parts) > 1:
            return os.path.join(models_root, actual_base_folder, *folder_parts[1:])
        else:
            return os.path.join(models_root, actual_base_folder)

    def _is_huggingface_url(self, url: str) -> bool:
        """檢查是否為 HuggingFace URL"""
        return 'huggingface.co' in url.lower()

    def _parse_hf_url(self, url: str) -> Tuple[str, str]:
        """
        解析 HuggingFace URL

        範例: https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf
        返回: ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q8_0.gguf")
        """
        pattern = r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)'
        match = re.search(pattern, url)

        if match:
            return match.group(1), match.group(2)

        raise ValueError(f"無法解析 HuggingFace URL：{url}")

    def _extract_filename(self, url: str) -> str:
        """從 URL 提取檔案名稱"""
        url = url.split('?')[0]
        return url.split('/')[-1]

    def _generate_status_message(self, results: List[str], skipped_files: List[str],
                                  incomplete_files: List[str], total_files: int) -> Tuple[str]:
        """產生最終狀態訊息"""
        success_count = sum(1 for r in results if r.startswith("✓"))
        fail_count = sum(1 for r in results if r.startswith("✗"))

        status_lines = ["下載完成！"]

        if skipped_files:
            status_lines.append(f"已跳過（檔案已存在）：{len(skipped_files)}")
        if incomplete_files:
            status_lines.append(f"重新下載（檔案不完整）：{len(incomplete_files)}")

        status_lines.append(f"成功：{success_count}/{total_files}")
        status_lines.append(f"失敗：{fail_count}")
        status_lines.append("")
        status_lines.append("詳細資訊：")

        if skipped_files:
            status_lines.append("  已跳過的檔案：")
            for f in skipped_files:
                status_lines.append(f"    ○ {f}")

        if incomplete_files:
            status_lines.append("  重新下載的檔案：")
            for f in incomplete_files:
                status_lines.append(f"    ⟳ {f}")

        status_lines.extend(f"  {r}" for r in results)

        status_msg = "\n".join(status_lines)
        print(f"\n{status_msg}")
        return (status_msg,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ModelDownloader": ModelDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelDownloader": "Model Downloader",
}
