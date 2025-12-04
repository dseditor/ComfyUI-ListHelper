"""
測試 QwenGPUInference 節點
"""

import sys
import os

# 添加當前目錄和 ComfyUI 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, current_dir)
sys.path.insert(0, comfy_path)

# 模擬 folder_paths
class MockFolderPaths:
    @staticmethod
    def get_folder_paths(folder_type):
        if folder_type == "text_encoders":
            return [os.path.join(comfy_path, "models", "text_encoders")]
        return []

# 替換 folder_paths
import folder_paths as real_folder_paths
try:
    # 嘗試使用真實的 folder_paths
    pass
except:
    # 如果失敗，使用模擬的
    sys.modules['folder_paths'] = MockFolderPaths()

# 現在可以導入節點
from qwen_inference import QwenGPUInference

def test_node():
    """測試節點基本功能"""

    print("=" * 60)
    print("測試 QwenGPUInference 節點")
    print("=" * 60)

    # 創建節點實例
    node = QwenGPUInference()

    # 測試獲取檔案列表
    print("\n1. 測試獲取 safetensors 檔案列表...")
    files = node._get_safetensors_files()
    print(f"   找到 {len(files)} 個檔案:")
    for f in files:
        print(f"   - {os.path.basename(f) if f != 'No safetensors files found' else f}")

    if files[0] == "No safetensors files found":
        print("\n⚠️ 未找到 safetensors 檔案，無法繼續測試")
        print("請確保在 models/text_encoders 目錄中有 .safetensors 檔案")
        return

    # 測試推理
    print("\n2. 測試推理功能...")
    model_file = os.path.basename(files[0])

    result = node.inference(
        model_file=model_file,
        user_prompt="你好，請介紹一下你自己。",
        system_prompt="你是一個有用的 AI 助手。",
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repo_id="Qwen/Qwen3-4B"
    )

    print("\n推理結果:")
    print("-" * 60)
    print(result[0])
    print("-" * 60)

    print("\n✓ 測試完成!")

if __name__ == "__main__":
    test_node()
