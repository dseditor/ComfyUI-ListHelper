"""
人物特徵提取節點
用於從圖片中提取人物特徵描述
"""

class PersonFeatureExtractor:
    """
    人物特徵提取器
    接收圖片，輸出特徵提取提示詞給 LLM
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("feature_prompt",)
    FUNCTION = "extract_features"
    CATEGORY = "DesignPack"
    
    def extract_features(self, image):
        """生成人物特徵提取提示詞"""
        try:
            # 生成用於 LLM 的特徵提取提示詞
            feature_prompt = """請仔細分析這張圖片中的人物，提取以下特徵並用簡潔的中文描述（50-80字）：

1. **國籍/種族特徵**：判斷人物的種族特徵（例如：亞洲人、歐美人等）
2. **臉型**：描述臉型（例如：圓臉、瓜子臉、方臉、鵝蛋臉等）
3. **五官特徵**：
   - 眼睛：大小、形狀、顏色
   - 鼻子：高挺或扁平
   - 嘴巴：大小、唇形
4. **妝容風格**：描述妝容（例如：自然妝、濃妝、裸妝等）
5. **髮型和髮色**：詳細描述髮型和顏色
6. **其他明顯特徵**：眼鏡、飾品、特殊標記等

請直接輸出特徵描述，不要包含其他說明文字。

範例格式：
"亞洲女性，瓜子臉，大眼睛雙眼皮，高挺鼻梁，自然妝容，黑色長直髮，戴黑框眼鏡"
"""
            
            print("📸 人物特徵提取提示詞已生成")
            print("💡 請將此提示詞連接到支援圖片的 LLM 節點（如 GGUF_LLM 或 OpenAI Helper）")
            print("   並將圖片也連接到 LLM 節點")
            
            return (feature_prompt,)
            
        except Exception as e:
            import traceback
            error_msg = f"錯誤：{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (error_msg,)


class PersonFeatureParser:
    """
    人物特徵解析器
    接收 LLM 輸出的特徵描述，清理並輸出
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_output": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("features",)
    FUNCTION = "parse"
    CATEGORY = "DesignPack"
    
    def parse(self, llm_output):
        """解析 LLM 輸出的人物特徵"""
        try:
            if not llm_output or not llm_output.strip():
                return ("",)
            
            # 清理輸出（移除可能的 markdown 標記等）
            features = llm_output.strip()
            
            # 移除常見的前綴
            prefixes_to_remove = [
                "人物特徵：",
                "特徵描述：",
                "描述：",
                "特徵：",
            ]
            
            for prefix in prefixes_to_remove:
                if features.startswith(prefix):
                    features = features[len(prefix):].strip()
            
            # 移除 markdown 標記
            features = features.replace("**", "").replace("*", "")
            
            # 限制長度（最多 150 字）
            if len(features) > 150:
                features = features[:150] + "..."
            
            print("✅ 人物特徵解析完成")
            print(f"   特徵: {features}")
            print("💡 請將此特徵連接到寫真雜誌提示詞注入器的 features 參數")
            
            return (features,)
            
        except Exception as e:
            import traceback
            error_msg = f"錯誤：{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return ("",)


# 節點註冊
NODE_CLASS_MAPPINGS = {
    "PersonFeatureExtractor": PersonFeatureExtractor,
    "PersonFeatureParser": PersonFeatureParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonFeatureExtractor": "👤 人物特徵提取器",
    "PersonFeatureParser": "📋 人物特徵解析器",
}
