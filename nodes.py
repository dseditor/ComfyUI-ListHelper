import copy
import torch
import torch.nn.functional as F
import os
import re
import sys
import json
import math
import subprocess
import codecs
import time
import datetime
import random as rnd
import torchaudio
import folder_paths
import json
from comfy.comfy_types import IO
from comfy_api.input_impl import VideoFromFile, VideoFromComponents
from comfy_api.util import VideoContainer, VideoCodec, VideoComponents
from fractions import Fraction
from typing import Optional
from comfy.cli_args import args
from typing import List, Dict, Any, Tuple
from random import Random
from datetime import datetime
from .qwen_inference import QwenGPUInference
from .gguf_inference import GGUFInference
from .model_downloader import ModelDownloader
from .openai_helper import OpenAIHelper
from .openrouter_llm import OpenRouterLLM

class AudioListGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "waveform": ("AUDIO",),
                "videofps": ("FLOAT", {"default": 23.976, "min": 1.0, "step": 0.001}),
                "samplefps": ("INT", {"default": 81, "min": 1}),
                "pad_last_segment": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "crossfade_duration": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "crossfade_type": (["linear", "cosine", "equal_power"], {"default": "cosine"}),
            }
        }

    RETURN_TYPES = ("INT", "AUDIO",)
    OUTPUT_IS_LIST = (False, True)
    RETURN_NAMES = ("cycle", "audio_list")
    FUNCTION = "split"
    CATEGORY = "ListHelper/Audio"

    def split(self, waveform, videofps, samplefps, pad_last_segment, crossfade_duration=0.1, crossfade_type="cosine"):
        audio_tensor = waveform["waveform"]         # shape: [1, C, N]
        sample_rate = waveform["sample_rate"]
        total_samples = audio_tensor.shape[-1]

        segment_duration_seconds = samplefps / videofps
        samples_per_segment = int(segment_duration_seconds * sample_rate)
        crossfade_samples = int(crossfade_duration * sample_rate)

        audio_list = []

        # 確保交叉淡化時間不會超過段落長度的一半
        crossfade_samples = min(crossfade_samples, samples_per_segment // 2)

        for i in range(0, total_samples, samples_per_segment):
            end_idx = min(i + samples_per_segment, total_samples)
            
            # 計算實際的開始和結束位置，考慮交叉淡化
            actual_start = max(0, i - crossfade_samples) if i > 0 else 0
            actual_end = min(total_samples, end_idx + crossfade_samples) if end_idx < total_samples else end_idx
            
            # 提取包含交叉淡化部分的音頻段
            extended_segment = audio_tensor[:, :, actual_start:actual_end].clone()
            
            # 應用交叉淡化效果
            if crossfade_samples > 0:
                extended_segment = self._apply_crossfade(
                    extended_segment, 
                    crossfade_samples, 
                    crossfade_type,
                    actual_start, 
                    i, 
                    end_idx, 
                    actual_end
                )

            # 如果需要填充最後一個段落
            segment_len = extended_segment.shape[-1]
            if pad_last_segment and end_idx == total_samples and segment_len < samples_per_segment:
                pad_len = samples_per_segment - segment_len
                extended_segment = F.pad(extended_segment, (0, pad_len))

            audio_obj = {
                "waveform": extended_segment,
                "sample_rate": sample_rate
            }

            audio_list.append(copy.deepcopy(audio_obj))

        return len(audio_list), audio_list

    def _apply_crossfade(self, segment, crossfade_samples, crossfade_type, actual_start, segment_start, segment_end, actual_end):
        """
        對音頻段應用交叉淡化效果
        
        Args:
            segment: 音頻段張量 [1, C, T]
            crossfade_samples: 交叉淡化的樣本數
            crossfade_type: 交叉淡化類型
            actual_start: 實際開始位置
            segment_start: 段落開始位置
            segment_end: 段落結束位置
            actual_end: 實際結束位置
        """
        if crossfade_samples == 0:
            return segment

        segment_length = segment.shape[-1]
        
        # 創建淡化曲線
        fade_curve = self._create_fade_curve(crossfade_samples, crossfade_type)
        
        # 應用淡入效果（段落開始處）
        if actual_start < segment_start:
            fade_in_length = min(crossfade_samples, segment_length)
            fade_in_curve = fade_curve[:fade_in_length]
            
            # 擴展維度以匹配音頻張量 [1, C, fade_in_length]
            fade_in_curve = fade_in_curve.unsqueeze(0).unsqueeze(0)
            fade_in_curve = fade_in_curve.expand(segment.shape[0], segment.shape[1], -1)
            
            segment[:, :, :fade_in_length] *= fade_in_curve

        # 應用淡出效果（段落結束處）
        if actual_end > segment_end:
            fade_out_length = min(crossfade_samples, segment_length)
            fade_out_curve = fade_curve[:fade_out_length].flip(0)  # 反轉淡化曲線
            
            # 擴展維度以匹配音頻張量
            fade_out_curve = fade_out_curve.unsqueeze(0).unsqueeze(0)
            fade_out_curve = fade_out_curve.expand(segment.shape[0], segment.shape[1], -1)
            
            segment[:, :, -fade_out_length:] *= fade_out_curve

        return segment

    def _create_fade_curve(self, length, fade_type):
        """
        創建淡化曲線
        
        Args:
            length: 淡化長度（樣本數）
            fade_type: 淡化類型 ("linear", "cosine", "equal_power")
        
        Returns:
            淡化曲線張量
        """
        import torch
        import math
        
        if fade_type == "linear":
            # 線性淡化：從0到1
            curve = torch.linspace(0.0, 1.0, length)
            
        elif fade_type == "cosine":
            # 餘弦淡化：更平滑的過渡
            t = torch.linspace(0.0, math.pi/2, length)
            curve = torch.sin(t)
            
        elif fade_type == "equal_power":
            # 等功率淡化：保持總功率恆定
            t = torch.linspace(0.0, math.pi/2, length)
            curve = torch.sin(t)
            
        else:
            # 預設使用線性淡化
            curve = torch.linspace(0.0, 1.0, length)
        
        return curve

class AudioToFrameCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "calculate"
    CATEGORY = "ListHelper/Audio"

    def calculate(self, audio, fps):
        waveform = audio["waveform"]         # shape: [1, channels, samples]
        sample_rate = audio["sample_rate"]   # e.g., 44100

        total_samples = waveform.shape[-1]
        duration_sec = total_samples / sample_rate
        total_frames = int(duration_sec * fps)

        return (total_frames,)
        


        
class PromptListGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "delimiter": ("STRING", {"multiline": False, "default": "", "dynamicPrompts": False}),
                "use_regex": ("BOOLEAN", {"default": False}),
                "keep_delimiter": ("BOOLEAN", {"default": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "skip_every": ("INT", {"default": 0, "min": 0, "max": 10}),
                "max_count": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "skip_first_index": ("BOOLEAN", {"default": False}),
                "random_order": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    
    INPUT_IS_LIST = False
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text_list", "total_index")
    FUNCTION = "run"
    OUTPUT_IS_LIST = (True, False)
    CATEGORY = "ListHelper/Tools"
    
    def run(self, text, delimiter, use_regex, keep_delimiter, start_index, skip_every, max_count, skip_first_index, random_order, seed):
        # 處理多個換行符號為一個換行符號
        text = re.sub(r'\n+', '\n', text)
        
        # 如果delimiter為空，則使用換行符號作為分隔符
        if not delimiter.strip():
            delimiter = '\n'
        # 直接使用delimiter進行搜尋分割，支援中日韓文字如"章"、"節"等
        
        # 如果需要跳過第一個無分隔符號的部分
        if skip_first_index:
            if use_regex:
                # 使用正規表示式搜尋第一個匹配
                match = re.search(delimiter, text)
                if match:
                    # 跳過第一個匹配之前的內容
                    text = text[match.start():]
            elif delimiter in text:
                # 找到第一個分隔符號的位置
                first_delimiter_pos = text.find(delimiter)
                if first_delimiter_pos > 0:
                    # 跳過第一個分隔符號之前的內容
                    text = text[first_delimiter_pos:]
        
        # 分割文本 - 支援正規表示式或一般字符串，並可選擇保留分隔符
        if use_regex:
            try:
                if keep_delimiter:
                    # 使用正規表示式分割並保留分隔符
                    arr = re.split(f'({delimiter})', text)
                    # 重新組合，讓每個片段都包含其前面的分隔符（除了第一個）
                    result = []
                    for i in range(0, len(arr)):
                        if i == 0:
                            # 第一個片段
                            if arr[i]:  # 如果不為空
                                result.append(arr[i])
                        elif i % 2 == 1:
                            # 這是分隔符，與下一個片段合併
                            if i + 1 < len(arr):
                                combined = arr[i] + arr[i + 1]
                                if combined.strip():  # 如果合併後不為空
                                    result.append(combined)
                        # i % 2 == 0 且 i > 0 的情況已經在上面處理過了
                    arr = result
                else:
                    # 使用正規表示式分割，不保留分隔符
                    arr = re.split(delimiter, text)
            except re.error:
                # 如果正規表示式有錯誤，回退到一般字符串分割
                if keep_delimiter:
                    arr = self._split_with_delimiter(text, delimiter)
                else:
                    arr = text.split(delimiter)
        else:
            # 使用一般字符串分割
            if keep_delimiter:
                arr = self._split_with_delimiter(text, delimiter)
            else:
                arr = text.split(delimiter)
        
        # 過濾空白項目並去除首尾空格
        arr = [item.strip() for item in arr if item.strip()]
        
        # 計算總數
        total_index = len(arr)
        
        # 根據random_order參數決定是否隨機排序
        if arr:
            if random_order:
                # 使用種子創建隨機數生成器並打亂順序
                rng = Random(seed)
                rng.shuffle(arr)
            
            # 根據參數選取項目
            selected_arr = arr[start_index:start_index + max_count * (skip_every + 1):(skip_every + 1)]
        else:
            selected_arr = []
        
        return (selected_arr, total_index)
    
    def _split_with_delimiter(self, text, delimiter):
        """輔助方法：用一般字符串分割並保留分隔符"""
        if delimiter not in text:
            return [text] if text.strip() else []
        
        parts = text.split(delimiter)
        result = []
        
        for i, part in enumerate(parts):
            if i == 0:
                # 第一個部分
                if part.strip():
                    result.append(part)
            else:
                # 其他部分都加上分隔符
                combined = delimiter + part
                if combined.strip():
                    result.append(combined)
        
        return result
        


class NumberListGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("FLOAT", {
                    "default": 0.0, 
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "max_value": ("FLOAT", {
                    "default": 10.0, 
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "step": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.01,
                    "max": 1000.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "count": ("INT", {
                    "default": 10, 
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "random": ("BOOLEAN", {
                    "default": False
                })
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1, 
                    "min": -1, 
                    "max": 1000000,
                    "step": 1,
                    "display": "number"
                })
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "INT")
    RETURN_NAMES = ("int_list", "float_list", "total_count")
    FUNCTION = "generate_number_list"
    CATEGORY = "ListHelper/Math"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True, True, False)
    
    def generate_number_list(self, min_value, max_value, step, count, random, seed=-1):
        """
        生成數字列表
        
        Args:
            min_value: 起始值
            max_value: 最大值
            step: 步長
            count: 數量
            random: 是否隨機排列
            seed: 隨機種子
        """
        print(f"Generating number list - min: {min_value}, max: {max_value}, step: {step}, count: {count}, random: {random}, seed: {seed}")
        
        # 生成基礎數字列表
        float_list = []
        current_value = min_value
        
        for i in range(count):
            if current_value > max_value:
                break
            float_list.append(current_value)
            current_value += step
        
        # 生成整數列表
        int_list = [int(val) for val in float_list]
        
        # 如果啟用隨機排列
        if random:
            # 設定隨機種子
            if seed >= 0:
                rnd.seed(seed)
            
            # 隨機打亂兩個列表（保持對應關係）
            combined = list(zip(int_list, float_list))
            rnd.shuffle(combined)
            int_list, float_list = zip(*combined)
            int_list = list(int_list)
            float_list = list(float_list)
        
        # 總數量
        total_count = len(float_list)
        
        print(f"Generated {total_count} numbers")
        return (int_list, float_list, total_count)


def create_number_list(min_value, max_value, step, count, random=False, seed=-1):
    """
    獨立的數字列表生成函數
    """
    node = NumberListGeneratorNode()
    return node.generate_number_list(min_value, max_value, step, count, random, seed)
 
class AudioListCombine:
    """
    合併音檔清單為單一音檔的節點
    將多個音檔按順序串接，或進行混音處理
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO",),  # 接收音檔清單
                "combine_mode": (["concatenate", "mix", "overlay"], {"default": "concatenate"}),
            },
            "optional": {
                "fade_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "target_sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "combine_audio_list"
    CATEGORY = "ListHelper/Audio"
    
    # 標記此節點接收清單輸入
    INPUT_IS_LIST = True
    
    def combine_audio_list(self, audio_list: List[Dict], combine_mode: List[str], 
                          fade_duration: List[float] = [0.0], 
                          normalize_output: List[bool] = [True],
                          target_sample_rate: List[int] = [44100]) -> Tuple[Dict]:
        """
        合併音檔清單
        
        Args:
            audio_list: 音檔清單，每個元素包含 'waveform' 和 'sample_rate'
            combine_mode: 合併模式 - concatenate(串接), mix(混音), overlay(覆疊)
            fade_duration: 淡入淡出時長（秒）
            normalize_output: 是否標準化輸出
            target_sample_rate: 目標採樣率
            
        Returns:
            合併後的音檔字典
        """
        
        # 取得參數（因為 INPUT_IS_LIST=True，所有參數都是清單）
        mode = combine_mode[0]
        fade_dur = fade_duration[0]
        normalize = normalize_output[0]
        target_sr = target_sample_rate[0]
        
        if not audio_list:
            raise ValueError("音檔清單不能為空")
        
        # 預處理：統一採樣率和聲道數
        processed_audio = []
        for audio_dict in audio_list:
            waveform = audio_dict['waveform']  # [B, C, T]
            sample_rate = audio_dict['sample_rate']
            
            # 重新採樣到目標採樣率
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            processed_audio.append(waveform)
        
        # 統一聲道數（取最大聲道數）
        max_channels = max(audio.shape[1] for audio in processed_audio)
        for i, audio in enumerate(processed_audio):
            if audio.shape[1] < max_channels:
                # 單聲道轉雙聲道或補齊聲道
                if audio.shape[1] == 1 and max_channels == 2:
                    processed_audio[i] = audio.repeat(1, 2, 1)
                else:
                    # 用零填充缺少的聲道
                    pad_channels = max_channels - audio.shape[1]
                    padding = torch.zeros(audio.shape[0], pad_channels, audio.shape[2])
                    processed_audio[i] = torch.cat([audio, padding], dim=1)
        
        # 根據模式合併音檔
        if mode == "concatenate":
            combined_waveform = self._concatenate_audio(processed_audio, fade_dur)
        elif mode == "mix":
            combined_waveform = self._mix_audio(processed_audio)
        elif mode == "overlay":
            combined_waveform = self._overlay_audio(processed_audio)
        else:
            raise ValueError(f"不支援的合併模式: {mode}")
        
        # 標準化輸出
        if normalize:
            combined_waveform = self._normalize_audio(combined_waveform)
        
        # 確保輸出格式正確
        if combined_waveform.dim() == 2:
            combined_waveform = combined_waveform.unsqueeze(0)  # 添加批次維度
        
        result_dict = {
            'waveform': combined_waveform,
            'sample_rate': target_sr
        }
        
        return (result_dict,)
    
    def _concatenate_audio(self, audio_list: List[torch.Tensor], fade_duration: float) -> torch.Tensor:
        """串接音檔"""
        if len(audio_list) == 1:
            return audio_list[0]
        
        result = audio_list[0]
        
        for next_audio in audio_list[1:]:
            if fade_duration > 0:
                result = self._crossfade_concat(result, next_audio, fade_duration)
            else:
                result = torch.cat([result, next_audio], dim=2)  # 在時間維度串接
        
        return result
    
    def _mix_audio(self, audio_list: List[torch.Tensor]) -> torch.Tensor:
        """混音（平均）"""
        # 找出最長的音檔長度
        max_length = max(audio.shape[2] for audio in audio_list)
        batch_size = audio_list[0].shape[0]
        channels = audio_list[0].shape[1]
        
        # 將所有音檔填充到相同長度
        padded_audio = []
        for audio in audio_list:
            if audio.shape[2] < max_length:
                padding = torch.zeros(batch_size, channels, max_length - audio.shape[2])
                audio = torch.cat([audio, padding], dim=2)
            padded_audio.append(audio)
        
        # 疊加並平均
        mixed = torch.stack(padded_audio, dim=0).mean(dim=0)
        return mixed
    
    def _overlay_audio(self, audio_list: List[torch.Tensor]) -> torch.Tensor:
        """覆疊音檔（直接相加）"""
        # 找出最長的音檔長度
        max_length = max(audio.shape[2] for audio in audio_list)
        batch_size = audio_list[0].shape[0]
        channels = audio_list[0].shape[1]
        
        # 初始化結果張量
        result = torch.zeros(batch_size, channels, max_length)
        
        # 逐個添加音檔
        for audio in audio_list:
            result[:, :, :audio.shape[2]] += audio
        
        return result
    
    def _crossfade_concat(self, audio1: torch.Tensor, audio2: torch.Tensor, 
                         fade_duration: float, sample_rate: int = 44100) -> torch.Tensor:
        """交叉淡化串接"""
        fade_samples = int(fade_duration * sample_rate)
        
        if fade_samples == 0 or audio1.shape[2] < fade_samples:
            return torch.cat([audio1, audio2], dim=2)
        
        # 創建淡出和淡入曲線
        fade_out = torch.linspace(1.0, 0.0, fade_samples).unsqueeze(0).unsqueeze(0)
        fade_in = torch.linspace(0.0, 1.0, fade_samples).unsqueeze(0).unsqueeze(0)
        
        # 分割音檔
        audio1_main = audio1[:, :, :-fade_samples]
        audio1_tail = audio1[:, :, -fade_samples:]
        
        if audio2.shape[2] >= fade_samples:
            audio2_head = audio2[:, :, :fade_samples]
            audio2_main = audio2[:, :, fade_samples:]
        else:
            audio2_head = audio2
            audio2_main = torch.zeros(audio2.shape[0], audio2.shape[1], 0)
        
        # 應用交叉淡化
        crossfade_section = audio1_tail * fade_out + audio2_head * fade_in
        
        # 合併結果
        result = torch.cat([audio1_main, crossfade_section, audio2_main], dim=2)
        return result
    
    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """標準化音檔到 [-1, 1] 範圍"""
        max_val = waveform.abs().max()
        if max_val > 0:
            return waveform / max_val
        return waveform
        
class CeilDivide:
    """
    將 a/b 的結果無條件進位為整數
    例如: 21.02 -> 22, 21.99 -> 22, 21.00 -> 21
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 1, "min": -999999, "max": 999999}),
                "b": ("INT", {"default": 1, "min": -999999, "max": 999999}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "ceil_divide"
    CATEGORY = "ListHelper/Math"
    
    def ceil_divide(self, a: int, b: int) -> tuple:
        """
        計算 a/b 並無條件進位為整數
        
        Args:
            a: 被除數
            b: 除數
            
        Returns:
            無條件進位後的整數結果
        """
        if b == 0:
            raise ValueError("除數不能為零")
        
        # 計算除法結果
        division_result = a / b
        
        # 使用 math.ceil 進行無條件進位
        result = math.ceil(division_result)
        
        return (result,)     
        
class FrameMatch:
    """
    調整圖像序列到指定幀數的節點
    如果目標幀數大於輸入幀數，會重複最後一幀來補齊
    如果目標幀數小於輸入幀數，會截取前面的幀
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 輸入圖像序列
                "target_frames": ("INT", {
                    "default": 100, 
                    "min": 1, 
                    "max": 10000, 
                    "step": 1,
                    "tooltip": "目標幀數"
                }),
            },
            "optional": {
                "fill_mode": (["repeat_last", "loop", "bounce"], {
                    "default": "repeat_last",
                    "tooltip": "填充模式：repeat_last=重複最後一幀，loop=循環播放，bounce=來回播放"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "match_frames"
    CATEGORY = "ListHelper/Tools"
    
    def match_frames(self, images, target_frames, fill_mode="repeat_last"):
        """
        調整圖像序列到目標幀數
        
        Args:
            images: 輸入圖像張量 [N, H, W, C]
            target_frames: 目標幀數
            fill_mode: 填充模式
            
        Returns:
            調整後的圖像序列
        """
        import torch
        
        if images is None or images.shape[0] == 0:
            raise ValueError("輸入圖像序列不能為空")
        
        current_frames = images.shape[0]
        
        print(f"FrameMatch: 當前幀數 {current_frames} -> 目標幀數 {target_frames}")
        
        # 如果當前幀數等於目標幀數，直接返回
        if current_frames == target_frames:
            return (images,)
        
        # 如果目標幀數小於當前幀數，截取前面的幀
        elif target_frames < current_frames:
            matched_images = images[:target_frames]
            print(f"FrameMatch: 截取前 {target_frames} 幀")
            
        # 如果目標幀數大於當前幀數，需要填充
        else:
            additional_frames_needed = target_frames - current_frames
            
            if fill_mode == "repeat_last":
                # 重複最後一幀
                last_frame = images[-1:].clone()  # 保持維度 [1, H, W, C]
                repeated_frames = last_frame.repeat(additional_frames_needed, 1, 1, 1)
                matched_images = torch.cat([images, repeated_frames], dim=0)
                print(f"FrameMatch: 重複最後一幀 {additional_frames_needed} 次")
                
            elif fill_mode == "loop":
                # 循環播放整個序列
                loops_needed = (additional_frames_needed + current_frames - 1) // current_frames
                looped_images = images.repeat(loops_needed + 1, 1, 1, 1)
                matched_images = looped_images[:target_frames]
                print(f"FrameMatch: 循環播放 {loops_needed} 次")
                
            elif fill_mode == "bounce":
                # 來回播放（正向 -> 反向 -> 正向...）
                additional_images = []
                remaining_frames = additional_frames_needed
                forward = True
                
                while remaining_frames > 0:
                    if forward:
                        # 正向播放（跳過第一幀以避免重複）
                        frames_to_add = min(remaining_frames, current_frames - 1)
                        if frames_to_add > 0:
                            additional_images.append(images[1:frames_to_add + 1])
                            remaining_frames -= frames_to_add
                    else:
                        # 反向播放（跳過最後一幀以避免重複）
                        frames_to_add = min(remaining_frames, current_frames - 1)
                        if frames_to_add > 0:
                            # 反轉順序，並跳過最後一幀
                            reversed_frames = torch.flip(images[:-1], dims=[0])
                            additional_images.append(reversed_frames[:frames_to_add])
                            remaining_frames -= frames_to_add
                    
                    forward = not forward
                
                if additional_images:
                    bounced_frames = torch.cat(additional_images, dim=0)
                    matched_images = torch.cat([images, bounced_frames], dim=0)
                else:
                    matched_images = images
                
                print(f"FrameMatch: 來回播放模式，添加 {additional_frames_needed} 幀")
            
            else:
                # 預設使用重複最後一幀
                last_frame = images[-1:].clone()
                repeated_frames = last_frame.repeat(additional_frames_needed, 1, 1, 1)
                matched_images = torch.cat([images, repeated_frames], dim=0)
                print(f"FrameMatch: 使用預設模式，重複最後一幀 {additional_frames_needed} 次")
        
        # 確保輸出幀數正確
        final_frames = matched_images.shape[0]
        if final_frames != target_frames:
            # 如果還是不匹配，進行最終調整
            if final_frames > target_frames:
                matched_images = matched_images[:target_frames]
            else:
                # 補齊差異
                diff = target_frames - final_frames
                last_frame = matched_images[-1:].clone()
                extra_frames = last_frame.repeat(diff, 1, 1, 1)
                matched_images = torch.cat([matched_images, extra_frames], dim=0)
        
        print(f"FrameMatch: 完成，最終幀數 {matched_images.shape[0]}")
        
        return (matched_images,)


class SimpleWildCardPlayer:
    """
    簡單的 Wildcard 抽取節點
    從 WildCard 資料夾中隨機抽取內容，支援批次生成和自訂範本
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 掃描 wildcard 資料夾取得所有範本
        wildcard_root = os.path.join(os.path.dirname(__file__), "wildcard")
        templates = ["basic"]  # 預設範本

        if os.path.exists(wildcard_root):
            templates = [d for d in os.listdir(wildcard_root)
                        if os.path.isdir(os.path.join(wildcard_root, d))]

        if not templates:
            templates = ["basic"]

        return {
            "required": {
                "basic_prompt": ("STRING", {
                    "multiline": True,
                    "default": "(masterpiece, best quality, photorealistic, 8k, highly detailed, solo, 1girl, realistic photography:1.2), natural skin texture, well-rested appearance, detailed eyelashes, with blunt fringe, see-through bangs, Long sideburns, side locks framing face, detailed hair strands, over fringe, with, on neckline, (she wearing a full and proper, lace undergarment, tones, fully clothed, fully covering legs, complete outfit.)",
                    "tooltip": "固定的基礎提示詞"
                }),
                "wildcard_template": (templates, {
                    "default": "basic",
                    "tooltip": "選擇 wildcard 範本資料夾"
                }),
                "wildcard_files": ("STRING", {
                    "multiline": True,
                    "default": "years, age, pretit, contory, cm, face, realface_skin, bodytype, breasts, expression, eyeiled, eye_shape, eye_quality, eye_effect, eyecolor, hh, colors4, hairsize, hairlong, colors3, headwear, colors2, earring, earring-visibility, neck, mate, grid, wetdry, clothestight, tt-clothes, garter-tights, colors5, socks, shoeme, shoes, 169, light, RandomPose-light, backsence",
                    "tooltip": "要抽取的 Wildcard 檔案名稱（不含 .txt），用逗號分隔"
                }),
                "batch_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "要生成的批次數量"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "隨機種子"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_wildcards"
    CATEGORY = "ListHelper/Tools"

    def generate_wildcards(self, basic_prompt, wildcard_template, wildcard_files, batch_count, seed):
        """
        生成 Wildcard 提示詞列表

        Args:
            basic_prompt: 基礎提示詞
            wildcard_template: 範本資料夾名稱
            wildcard_files: 要抽取的檔案列表（逗號分隔）
            batch_count: 批次數量
            seed: 隨機種子

        Returns:
            提示詞列表
        """
        # 解析 wildcard 檔案名稱
        file_names = [name.strip() for name in wildcard_files.split(",") if name.strip()]

        # 載入所有 wildcard 檔案內容
        wildcard_root = os.path.join(os.path.dirname(__file__), "wildcard", wildcard_template)
        wildcard_data = {}

        for file_name in file_names:
            file_path = os.path.join(wildcard_root, f"{file_name}.txt")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = [line.strip() for line in f.readlines() if line.strip()]
                        if content:
                            wildcard_data[file_name] = content
                            print(f"Loaded wildcard: {file_name} ({len(content)} entries)")
                except Exception as e:
                    print(f"Error loading {file_name}.txt: {e}")
            else:
                print(f"Warning: {file_name}.txt not found in {wildcard_template}")

        if not wildcard_data:
            print("No wildcard files loaded, returning basic prompt only")
            return ([basic_prompt] * batch_count,)

        # 生成批次提示詞
        prompt_list = []

        for i in range(batch_count):
            current_seed = seed + i
            rng = Random(current_seed)

            # 建構提示詞
            wildcard_parts = []
            for file_name in file_names:
                if file_name in wildcard_data:
                    selected = rng.choice(wildcard_data[file_name])
                    wildcard_parts.append(selected)

            # 組合最終提示詞
            if wildcard_parts:
                wildcard_text = ", ".join(wildcard_parts)
                final_prompt = f"{basic_prompt}, {wildcard_text}"
            else:
                final_prompt = basic_prompt

            prompt_list.append(final_prompt)

            print(f"Generated prompt {i+1}/{batch_count} with seed {current_seed}")

        return (prompt_list,)


class BatchToPSD:
    """
    Convert batch PNG images to a multi-layer PSD file.
    The first image in the batch is skipped (merged result),
    and subsequent images become layers in the PSD file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI_PSD",
                    "tooltip": "Prefix for the output PSD filename"
                }),
            },
            "optional": {
                "reverse_layer_order": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the order of layers in PSD (bottom to top)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("message",)
    FUNCTION = "convert_to_psd"
    OUTPUT_NODE = True
    CATEGORY = "ListHelper/Tools"

    def convert_to_psd(self, images, filename_prefix, reverse_layer_order=False):
        """
        Convert batch images to multi-layer PSD file.

        Args:
            images: Batch of images [N, H, W, C]
            filename_prefix: Filename prefix for output PSD
            reverse_layer_order: Whether to reverse layer order

        Returns:
            Status message with file path or installation instruction
        """
        import numpy as np

        # Try to import psd-tools
        try:
            from psd_tools import PSDImage
            from psd_tools.api.layers import PixelLayer
            from PIL import Image
        except ImportError:
            # Auto-install psd-tools
            print("BatchToPSD: psd-tools not found, attempting auto-installation...")
            try:
                # Get the Python executable path
                python_exe = sys.executable

                # Install psd-tools
                install_cmd = [python_exe, "-m", "pip", "install", "psd-tools"]
                print(f"BatchToPSD: Running: {' '.join(install_cmd)}")

                result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                if result.returncode == 0:
                    print("BatchToPSD: psd-tools installed successfully!")
                    print(result.stdout)
                    return ("SUCCESS: psd-tools package has been installed successfully. Please RESTART ComfyUI to use the BatchToPSD node.",)
                else:
                    error_msg = f"ERROR: Failed to install psd-tools.\nStdout: {result.stdout}\nStderr: {result.stderr}\n\nPlease manually install: pip install psd-tools"
                    print(f"BatchToPSD: {error_msg}")
                    return (error_msg,)

            except subprocess.TimeoutExpired:
                return ("ERROR: Installation timeout. Please manually install: pip install psd-tools",)
            except Exception as e:
                error_msg = f"ERROR: Failed to auto-install psd-tools: {str(e)}\n\nPlease manually install: pip install psd-tools"
                print(f"BatchToPSD: {error_msg}")
                return (error_msg,)

        # Validate input
        if images is None or images.shape[0] == 0:
            return ("ERROR: No images provided in batch.",)

        if images.shape[0] == 1:
            return ("ERROR: Batch must contain at least 2 images (first image is skipped, remaining become layers).",)

        # Skip first image and get remaining layers
        layer_images = images[1:]
        num_layers = layer_images.shape[0]

        print(f"BatchToPSD: Processing {num_layers} layers (skipped first image in batch)")

        # Create output directory if it doesn't exist
        output_dir = folder_paths.get_output_directory()

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.psd"
        filepath = os.path.join(output_dir, filename)

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else output_dir, exist_ok=True)

        try:
            # Convert tensors to PIL Images and detect if we need alpha channel
            pil_layers = []
            has_alpha = False

            print(f"\n  DEBUG: Input batch shape: {images.shape}")
            print(f"  DEBUG: Layer images shape (after skipping first): {layer_images.shape}")

            for i in range(num_layers):
                # Convert from torch tensor [H, W, C] with values in [0, 1] to numpy array
                img_array = layer_images[i].cpu().numpy()

                # Convert to uint8 [0, 255]
                img_array = (img_array * 255).astype(np.uint8)

                # Detect number of channels
                print(f"  DEBUG: Layer {i+1} array shape: {img_array.shape}, channels: {img_array.shape[2]}")
                
                if img_array.shape[2] == 4:
                    has_alpha = True
                    pil_img = Image.fromarray(img_array, mode='RGBA')
                    # Check actual transparency
                    alpha_channel = img_array[:, :, 3]
                    transparent_pixels = np.sum(alpha_channel == 0)
                    total_pixels = alpha_channel.size
                    print(f"  DEBUG: Layer {i+1} has RGBA, transparent pixels: {transparent_pixels:,} ({transparent_pixels/total_pixels*100:.2f}%)")
                elif img_array.shape[2] == 3:
                    pil_img = Image.fromarray(img_array, mode='RGB')
                    print(f"  DEBUG: Layer {i+1} is RGB (NO alpha channel)")
                else:
                    return (f"ERROR: Unsupported number of channels: {img_array.shape[2]}",)

                pil_layers.append(pil_img)
                print(f"  Converted layer {i+1}/{num_layers} - Size: {pil_img.size}, Mode: {pil_img.mode}")

            # CRITICAL FIX: Use RGBA mode for PSD when any layer has transparency
            # This prevents PixelLayer.frompil() from losing transparency when PSD is in RGB mode
            # Issue: PixelLayer.frompil() mixes RGBA image's transparent areas with RGB colors
            # Solution: Create PSD in RGBA mode to preserve alpha channel correctly
            
            # IMPORTANT: Always use RGBA mode if any layer has alpha channel
            # This matches the successful test script behavior
            target_mode = 'RGBA' if has_alpha else 'RGB'
            print(f"\n  Target PSD mode: {target_mode} (has_alpha={has_alpha})")
            
            # If we detected alpha, we MUST use RGBA mode
            # If no alpha detected but user expects transparency, force RGBA anyway
            # This is safer and matches Photoshop's behavior
            if has_alpha:
                print(f"  ✓ Using RGBA mode to preserve transparency")
            else:
                print(f"  Using RGB mode (no alpha channels detected)")

            # Convert all layers to match PSD mode
            # This is critical - we must convert BEFORE creating the PSD
            for i in range(len(pil_layers)):
                if pil_layers[i].mode != target_mode:
                    original_mode = pil_layers[i].mode
                    pil_layers[i] = pil_layers[i].convert(target_mode)
                    print(f"  Converted layer {i+1} from {original_mode} to {target_mode}")

            # Reverse layer order if requested
            if reverse_layer_order:
                pil_layers.reverse()
                print("  Reversed layer order")

            # Get dimensions from first layer
            width, height = pil_layers[0].size

            # Create PSD with proper color mode (RGBA preserves transparency!)
            psd = PSDImage.new(target_mode, (width, height), depth=8)
            print(f"  Created PSD document: {width}x{height}, mode={target_mode}, depth=8")

            # Add layers to PSD
            for i, pil_img in enumerate(pil_layers):
                layer_name = f"Layer_{i+1}"

                # Create layer from PIL image
                # When PSD is in RGBA mode, PixelLayer.frompil() preserves transparency correctly
                layer = PixelLayer.frompil(pil_img, psd)
                layer.name = layer_name

                # Add to PSD
                psd.append(layer)
                print(f"  Added layer: {layer_name} (mode={pil_img.mode})")

            # Save PSD file
            psd.save(filepath)

            success_msg = f"SUCCESS: PSD file saved to: {filepath} ({num_layers} layers, mode={target_mode})"
            print(f"BatchToPSD: {success_msg}")
            return (success_msg,)

        except Exception as e:
            error_msg = f"ERROR: Failed to create PSD file: {str(e)}"
            print(f"BatchToPSD: {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg,)





NODE_CLASS_MAPPINGS = {
    "AudioListGenerator": AudioListGenerator,
    "AudioToFrameCount": AudioToFrameCount,

    "PromptListGenerator": PromptListGenerator,
    "NumberListGenerator": NumberListGenerator,
    "AudioListCombine": AudioListCombine,
    "CeilDivide": CeilDivide,
    "FrameMatch": FrameMatch,
    "SimpleWildCardPlayer": SimpleWildCardPlayer,
    "QwenGPUInference": QwenGPUInference,
    "GGUFInference": GGUFInference,
    "BatchToPSD": BatchToPSD,
    "ModelDownloader": ModelDownloader,
    "OpenAIHelper": OpenAIHelper,
    "OpenRouterLLM": OpenRouterLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioListGenerator": "Audio Split to List",
    "AudioToFrameCount": "Audio to Frame Count",
    "PromptListGenerator": "PromptListGenerator",
    "NumberListGenerator": "NumberListGenerator",
    "AudioListCombine": "AudioListCombine",
    "CeilDivide": "CeilDivide",
    "FrameMatch": "FrameMatch",
    "SimpleWildCardPlayer": "Simple WildCard Player",
    "QwenGPUInference": "Qwen_TE_LLM",
    "GGUFInference": "GGUF_LLM",
    "BatchToPSD": "Batch to PSD",
    "ModelDownloader": "Model Downloader",
    "OpenAIHelper": "OpenAI Helper",
    "OpenRouterLLM": "OpenRouter LLM",
}

