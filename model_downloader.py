import os
import re
import shutil
import json
import time
import folder_paths
from typing import List, Dict, Tuple, Optional

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False


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
                    "default": "無",
                    "tooltip": "選擇模型範本。選擇範本後將使用範本中的下載列表，忽略上方手動輸入的內容。\n選擇「無」時使用上方手動輸入的下載列表。"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_message",)
    FUNCTION = "download_models"
    OUTPUT_NODE = True
    CATEGORY = "ListHelper/Tools"

    def _send_progress(self, unique_id, data):
        """透過 WebSocket 發送下載進度事件"""
        if HAS_SERVER and unique_id:
            try:
                PromptServer.instance.send_sync("model_download_progress", {
                    "node_id": str(unique_id),
                    **data
                })
            except Exception as e:
                print(f"ModelDownloader: 進度推送失敗：{e}")

    def download_models(self, download_list: str, use_s3c: bool, use_hf_download: bool,
                        chunk_size_mb: int, max_workers: int, model_template: str,
                        unique_id=None) -> Tuple[str]:
        """
        下載模型到 ComfyUI/models 資料夾

        Args:
            download_list: 下載列表文字
            use_s3c: 使用 S3impleClient 加速（僅在 HF 下載時有效）
            use_hf_download: 使用 HuggingFace Hub 下載（開啟時會自動搬移到 models 資料夾）
            chunk_size_mb: 區塊大小 (MB)
            max_workers: 最大平行執行緒數
            model_template: 模型範本名稱
            unique_id: 節點唯一 ID（由 ComfyUI 自動傳入）

        Returns:
            狀態訊息
        """
        # 若選擇了範本，使用範本的下載列表；否則使用手動輸入的 download_list
        no_template_values = {"無", "none", "None", ""}
        if model_template and model_template not in no_template_values:
            template_download_list = self._get_template_download_list(model_template)
            if template_download_list:
                download_list = template_download_list
                print(f"ModelDownloader: 使用範本「{model_template}」的下載列表")
            else:
                return (f"錯誤：無法載入範本「{model_template}」",)
        else:
            print(f"ModelDownloader: 未選擇範本，使用手動輸入的下載列表")

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

        # 發送檢查狀態
        self._send_progress(unique_id, {
            "status": "checking",
            "message": "正在檢查現有檔案...",
            "progress": 0,
            "current_file": 0,
            "total_files": 0,
        })

        # 檢查檔案是否已存在且完整
        files_to_download, skipped_files, incomplete_files = self._check_existing_files(
            download_map, models_root
        )

        if not files_to_download:
            skip_msg = "所有檔案已存在且完整。\n已跳過：\n" + "\n".join(f"  - {f}" for f in skipped_files)
            print(f"ModelDownloader: {skip_msg}")
            self._send_progress(unique_id, {
                "status": "complete",
                "message": "所有檔案已存在，無需下載",
                "progress": 100,
                "current_file": 0,
                "total_files": 0,
            })
            return (skip_msg,)

        # 開始下載
        results = []
        total_files = sum(len(urls) for urls in files_to_download.values())
        current_file = 0

        # 建立進度回調
        def progress_callback(filename, downloaded, total, cur_file, tot_files):
            progress = (downloaded / total * 100) if total > 0 else 0
            self._send_progress(unique_id, {
                "status": "downloading",
                "filename": filename,
                "progress": round(progress, 1),
                "downloaded": downloaded,
                "total": total,
                "current_file": cur_file,
                "total_files": tot_files,
                "message": f"正在下載：{filename}",
            })

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

                    # 發送開始下載事件
                    self._send_progress(unique_id, {
                        "status": "downloading",
                        "filename": filename,
                        "progress": 0,
                        "downloaded": 0,
                        "total": 0,
                        "current_file": current_file,
                        "total_files": total_files,
                        "message": f"正在下載：{filename}",
                    })

                    try:
                        if use_hf_download and self._is_huggingface_url(url):
                            # 使用 HF Hub 下載，然後搬移到目標資料夾
                            result = self._download_with_hf_hub(url, dest_path, s3c, patch_applied,
                                                                 progress_callback=progress_callback,
                                                                 current_file=current_file,
                                                                 total_files=total_files)
                        else:
                            # 使用普通 HTTP 下載
                            result = self._download_with_http(url, dest_path,
                                                               progress_callback=progress_callback,
                                                               current_file=current_file,
                                                               total_files=total_files)

                        results.append(result)

                    except Exception as e:
                        error_msg = f"✗ {filename} - 錯誤：{str(e)}"
                        results.append(error_msg)
                        print(f"  {error_msg}")
                        self._send_progress(unique_id, {
                            "status": "error",
                            "filename": filename,
                            "progress": 0,
                            "current_file": current_file,
                            "total_files": total_files,
                            "message": f"下載失敗：{str(e)}",
                        })

        finally:
            # 還原 HF patch
            if patch_applied and s3c:
                try:
                    s3c.unpatch_huggingface_hub()
                    print("\nModelDownloader: HuggingFace Hub 原始設定已還原")
                except Exception as e:
                    print(f"\nModelDownloader: 無法還原 HF patch ({str(e)})")

        # 發送下載完成事件
        success_count = sum(1 for r in results if r.startswith("✓"))
        fail_count = sum(1 for r in results if r.startswith("✗"))
        self._send_progress(unique_id, {
            "status": "complete",
            "message": f"下載完成！成功：{success_count}/{total_files}，失敗：{fail_count}",
            "progress": 100,
            "current_file": total_files,
            "total_files": total_files,
            "has_downloads": True,
        })

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
        支援 Civitai 等需要特殊處理的網站

        Returns:
            檔案大小（bytes），如果無法取得則返回 None
        """
        try:
            import urllib.request
            import ssl

            # 建立自訂的 opener 以處理重定向
            cookie_handler = urllib.request.HTTPCookieProcessor()
            redirect_handler = urllib.request.HTTPRedirectHandler()
            ssl_context = ssl.create_default_context()
            https_handler = urllib.request.HTTPSHandler(context=ssl_context)

            opener = urllib.request.build_opener(
                https_handler,
                cookie_handler,
                redirect_handler
            )

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': '*/*',
            }

            # Civitai 有時 HEAD 請求不返回 Content-Length，需要用 GET 請求的 Range header
            is_civitai = 'civitai.com' in url.lower()

            if is_civitai:
                # 對 Civitai 使用 GET 請求但只取 0 bytes 來獲取 headers
                request = urllib.request.Request(url, headers=headers)
                request.add_header('Range', 'bytes=0-0')
                request.add_header('Referer', 'https://civitai.com/')

                with opener.open(request, timeout=30) as response:
                    # 從 Content-Range 取得總大小
                    content_range = response.headers.get('Content-Range')
                    if content_range:
                        # 格式：bytes 0-0/12345678
                        import re
                        match = re.search(r'/(\d+)', content_range)
                        if match:
                            return int(match.group(1))

                    # 備用：嘗試 Content-Length
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        return int(content_length)
            else:
                # 非 Civitai 使用標準 HEAD 請求
                request = urllib.request.Request(url, method='HEAD', headers=headers)

                with opener.open(request, timeout=30) as response:
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        return int(content_length)

        except Exception as e:
            print(f"ModelDownloader: 無法取得遠端檔案大小：{e}")

        return None

    # ==================== 下載方法 ====================

    def _download_with_hf_hub(self, url: str, dest_path: str,
                              s3c: Optional[object], patch_applied: bool,
                              progress_callback=None, current_file: int = 0,
                              total_files: int = 0) -> str:
        """
        使用 HuggingFace Hub 下載，然後搬移到目標資料夾

        Args:
            url: HuggingFace URL
            dest_path: 最終目標路徑
            s3c: s3impleclient module（如果可用）
            patch_applied: S3C patch 是否已套用
            progress_callback: 進度回調函數
            current_file: 當前檔案序號
            total_files: 總檔案數

        Returns:
            結果訊息
        """
        from huggingface_hub import hf_hub_download

        filename = self._extract_filename(url)
        repo_id, file_path = self._parse_hf_url(url)

        print(f"  使用 HF Hub 下載...")
        print(f"  Repo: {repo_id}, File: {file_path}")

        # HF Hub 下載不支援細粒度進度，發送一個「下載中」狀態
        if progress_callback:
            progress_callback(filename, 0, 0, current_file, total_files)

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
                size_bytes = os.path.getsize(dest_path)
                method = "HF Hub + S3C 加速" if patch_applied else "HF Hub"
                if progress_callback:
                    progress_callback(filename, size_bytes, size_bytes, current_file, total_files)
                result = f"✓ {filename} ({size_mb:.2f} MB) ({method})"
                print(f"  ✓ 下載並搬移成功 - {size_mb:.2f} MB")
                return result
            else:
                result = f"✗ {filename} - 搬移失敗"
                print(f"  ✗ 搬移失敗")
                return result
        else:
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            size_bytes = os.path.getsize(dest_path)
            method = "HF Hub + S3C 加速" if patch_applied else "HF Hub"
            if progress_callback:
                progress_callback(filename, size_bytes, size_bytes, current_file, total_files)
            result = f"✓ {filename} ({size_mb:.2f} MB) ({method})"
            print(f"  ✓ 下載成功 - {size_mb:.2f} MB")
            return result

    def _download_with_http(self, url: str, dest_path: str,
                            progress_callback=None, current_file: int = 0,
                            total_files: int = 0) -> str:
        """
        使用普通 HTTP 下載（支援 Civitai 等需要特殊處理的網站）

        Args:
            url: 下載 URL
            dest_path: 目標路徑
            progress_callback: 進度回調函數
            current_file: 當前檔案序號
            total_files: 總檔案數

        Returns:
            結果訊息
        """
        import urllib.request
        import urllib.error
        import ssl

        filename = self._extract_filename(url)
        print(f"  使用 HTTP 下載...")

        # 確保目標資料夾存在
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # 使用暫存檔案下載
        temp_path = dest_path + '.tmp'

        try:
            # 建立自訂的 opener 以處理重定向和 cookies
            # Civitai 需要完整的瀏覽器 headers 和 cookie 支援
            cookie_handler = urllib.request.HTTPCookieProcessor()
            redirect_handler = urllib.request.HTTPRedirectHandler()

            # 建立 SSL context（某些網站需要）
            ssl_context = ssl.create_default_context()
            https_handler = urllib.request.HTTPSHandler(context=ssl_context)

            opener = urllib.request.build_opener(
                https_handler,
                cookie_handler,
                redirect_handler
            )

            # 設定完整的瀏覽器 headers（Civitai 會檢查這些）
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'identity',  # 不使用壓縮，方便計算進度
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
            }

            request = urllib.request.Request(url, headers=headers)

            # 檢查是否為 Civitai URL，需要特殊處理
            is_civitai = 'civitai.com' in url.lower()
            if is_civitai:
                print(f"  偵測到 Civitai 連結，使用特殊處理...")
                # Civitai 有時需要 Referer
                request.add_header('Referer', 'https://civitai.com/')

            with opener.open(request, timeout=300) as response:
                # 取得最終 URL（處理重定向後的實際下載位置）
                final_url = response.geturl()
                if final_url != url:
                    print(f"  重定向到：{final_url[:100]}...")
                    # 從最終 URL 取得真正的檔案名稱（如果可能）
                    if is_civitai:
                        # Civitai 重定向後的 URL 可能包含真正的檔案名稱
                        content_disposition = response.headers.get('Content-Disposition', '')
                        if content_disposition:
                            # 嘗試從 Content-Disposition 取得檔案名稱
                            import re
                            cd_match = re.search(r'filename[*]?=["\']?([^"\';\r\n]+)', content_disposition)
                            if cd_match:
                                real_filename = cd_match.group(1).strip()
                                # URL 解碼
                                real_filename = urllib.request.unquote(real_filename)
                                if real_filename and real_filename != filename:
                                    print(f"  實際檔案名稱：{real_filename}")
                                    # 更新目標路徑
                                    dest_path = os.path.join(os.path.dirname(dest_path), real_filename)
                                    temp_path = dest_path + '.tmp'
                                    filename = real_filename

                total_size = int(response.headers.get('Content-Length', 0))

                if total_size == 0 and is_civitai:
                    print(f"  警告：無法取得檔案大小，可能是 Civitai 連結問題")

                with open(temp_path, 'wb') as out_file:
                    downloaded = 0
                    chunk_size = 8192 * 4  # 增加 chunk size 提升效能
                    last_progress = 0
                    last_callback_time = time.time()

                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        out_file.write(chunk)
                        downloaded += len(chunk)

                        # 每 10% 顯示進度（終端機）
                        if total_size > 0:
                            progress = int(downloaded * 100 / total_size)
                            if progress >= last_progress + 10:
                                print(f"  進度：{progress}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                                last_progress = progress
                        elif downloaded % (10 * 1024 * 1024) == 0:  # 每 10MB 顯示一次
                            print(f"  已下載：{downloaded / (1024*1024):.2f} MB")

                        # WebSocket 進度推送（限制頻率，每 0.5 秒最多一次）
                        now = time.time()
                        if progress_callback and (now - last_callback_time >= 0.5):
                            progress_callback(filename, downloaded, total_size, current_file, total_files)
                            last_callback_time = now

            # 檢查下載的檔案是否有效（不是錯誤頁面）
            downloaded_size = os.path.getsize(temp_path)
            if downloaded_size < 1024 and is_civitai:
                # 檔案太小，可能是錯誤訊息
                with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if 'error' in content.lower() or 'unauthorized' in content.lower() or '<html' in content.lower():
                    os.remove(temp_path)
                    raise ValueError(f"Civitai 下載失敗，可能需要登入或 API Token。回應內容：{content[:200]}")

            # 下載完成，搬移到最終位置
            shutil.move(temp_path, dest_path)

            size_bytes = os.path.getsize(dest_path)
            size_mb = size_bytes / (1024 * 1024)
            if progress_callback:
                progress_callback(filename, size_bytes, size_bytes, current_file, total_files)
            result = f"✓ {filename} ({size_mb:.2f} MB) (HTTP)"
            print(f"  ✓ 下載成功 - {size_mb:.2f} MB")
            return result

        except urllib.error.HTTPError as e:
            # 清理暫存檔案
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

            if e.code == 401 or e.code == 403:
                raise ValueError(f"HTTP {e.code}：需要認證。請確認 Civitai URL 包含有效的 API Token（?token=xxx）")
            elif e.code == 404:
                raise ValueError(f"HTTP 404：找不到檔案。請確認 URL 是否正確")
            else:
                raise ValueError(f"HTTP 錯誤 {e.code}：{e.reason}")

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
        """
        從 URL 提取檔案名稱
        對於 Civitai 等特殊 URL，會返回臨時檔名（下載時會從 Content-Disposition 取得真正檔名）
        """
        # 檢查是否為 Civitai API 下載連結
        if 'civitai.com/api/download' in url.lower():
            # Civitai API URL 格式：/api/download/models/123456?...
            # 從 URL 取得 model version ID 作為臨時檔名
            url_path = url.split('?')[0]
            model_id = url_path.split('/')[-1]
            # 返回臨時檔名，下載時會從 Content-Disposition 取得真正檔名
            return f"civitai_model_{model_id}.safetensors"

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
