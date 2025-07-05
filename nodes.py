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
import ffmpeg
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
    CATEGORY = "ListHelper"

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
    CATEGORY = "ListHelper"

    def calculate(self, audio, fps):
        waveform = audio["waveform"]         # shape: [1, channels, samples]
        sample_rate = audio["sample_rate"]   # e.g., 44100

        total_samples = waveform.shape[-1]
        duration_sec = total_samples / sample_rate
        total_frames = int(duration_sec * fps)

        return (total_frames,)
        

class MergeVideoFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True, "default": ""}),
                "total_rounds": ("INT", {"default": 3, "min": 1, "max": 20}),
                "windows_path_format": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_file_path",)
    FUNCTION = "merge_files"
    CATEGORY = "ListHelper"
    
    def __init__(self):
        # 使用相對路徑的狀態檔案
        self.state_file = "merge_state.json"
    
    def merge_files(self, input_text, total_rounds, windows_path_format):
        # 讀取已累積的檔案
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                accumulated_files = state_data.get('files', [])
                file_type = state_data.get('type', None)
        except:
            accumulated_files = []
            file_type = None
        
        # 檢查輸入是否包含-audio.mp4檔案
        audio_files = re.findall(r"'([^']*-audio\.mp4)'", input_text)
        video_files = re.findall(r"'([^']*\.mp4)'", input_text)
        
        # 移除audio檔案，只保留純video檔案
        pure_video_files = [f for f in video_files if not f.endswith('-audio.mp4')]
        
        # 決定要處理的檔案類型
        if audio_files:
            current_files = audio_files
            current_type = 'audio'
        elif pure_video_files:
            current_files = pure_video_files
            current_type = 'video'
        else:
            return ("未找到有效的檔案",)
        
        # 如果檔案類型改變，重置累積的檔案
        if file_type and file_type != current_type:
            accumulated_files = []
        
        file_type = current_type
        
        # 累積新檔案
        for file in current_files:
            if file not in accumulated_files:
                accumulated_files.append(file)
        
        # 儲存狀態
        state_data = {
            'files': accumulated_files,
            'type': file_type
        }
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        
        # 如果只有一個檔案，直接返回
        if total_rounds == 1 and len(accumulated_files) >= 1:
            file_path = accumulated_files[0]
            # 清空狀態檔案
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            return (self._format_path(file_path, windows_path_format),)
        
        # 當累積到指定數量的檔案時合併
        if len(accumulated_files) >= total_rounds:
            output_dir = os.path.dirname(accumulated_files[0])
            
            # 產生帶時間戳的檔案名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if file_type == 'audio':
                output_filename = f"merged_audio_{timestamp}.mp4"
            else:
                output_filename = f"merged_video_{timestamp}.mp4"
            
            output_file = os.path.join(output_dir, output_filename)
            
            # 構建ffmpeg命令
            cmd = ["ffmpeg", "-y"]
            for file in accumulated_files:
                cmd.extend(["-i", file])
            
            # 根據檔案類型選擇不同的合併參數
            if file_type == 'audio':
                # 有音軌的合併
                filter_complex = "".join([f"[{i}:v][{i}:a]" for i in range(len(accumulated_files))])
                filter_complex += f"concat=n={len(accumulated_files)}:v=1:a=1[outv][outa]"
                cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]", "-map", "[outa]"])
            else:
                # 無音軌的合併
                filter_complex = "".join([f"[{i}:v]" for i in range(len(accumulated_files))])
                filter_complex += f"concat=n={len(accumulated_files)}:v=1:a=0[outv]"
                cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]"])
            
            cmd.append(output_file)
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    return (f"合併失敗: {result.stderr}",)
            except Exception as e:
                return (f"執行FFmpeg時發生錯誤: {str(e)}",)
            
            # 清空狀態檔案
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            
            return (self._format_path(output_file, windows_path_format),)
        else:
            # 返回空字串或佔位符，直到收集完成
            return ("",)
    
    def _format_path(self, file_path, windows_format):
        """格式化檔案路徑"""
        if windows_format:
            # Windows格式：先統一為正斜線，再轉換為單反斜線
            normalized_path = file_path.replace('\\\\', '/').replace('\\', '/')
            return normalized_path.replace('/', '\\')
        else:
            # 雙斜線格式：確保使用雙反斜線
            return file_path.replace('\\', '\\\\').replace('/', '\\\\')

        
class PromptListGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "delimiter": ("STRING", {"multiline": False, "default": ",", "dynamicPrompts": False}),
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
    CATEGORY = "ListHelper"
    
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
    CATEGORY = "ListHelper"
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
    CATEGORY = "listhelper"
    
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
    CATEGORY = "ListHelper"
    
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
        
class LoadVideoPath:
    """
    載入視頻檔案，輸出視頻物件和完整檔案路徑
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "file": (sorted(files), {"video_upload": True}),
            }
        }

    CATEGORY = "ListHelper"
    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "load_video_path"
    
    def load_video_path(self, file):
        video_path = folder_paths.get_annotated_filepath(file)
        video_object = VideoFromFile(video_path)
        return (video_object, video_path)

    @classmethod
    def IS_CHANGED(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


class SaveVideoPath:
    """
    保存視頻檔案，輸出保存後的完整檔案路徑
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "要保存的視頻"}),
                "filename_prefix": ("STRING", {"default": "video/ComfyUI", 
                                              "tooltip": "檔案名前綴"}),
                "format": (VideoContainer.as_input(), {"default": "auto", 
                                                      "tooltip": "視頻格式"}),
                "codec": (VideoCodec.as_input(), {"default": "auto", 
                                                 "tooltip": "視頻編碼"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save_video_path"
    OUTPUT_NODE = True
    CATEGORY = "ListHelper"

    def save_video_path(self, video, filename_prefix, format, codec, 
                       prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        width, height = video.get_dimensions()
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            width,
            height
        )
        
        # 準備元數據
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        
        # 生成檔案名和完整路徑
        file = f"{filename}_{counter:05}_.{VideoContainer.get_extension(format)}"
        full_path = os.path.join(full_output_folder, file)
        
        # 保存視頻
        video.save_to(
            full_path,
            format=format,
            codec=codec,
            metadata=saved_metadata
        )
        
        return (full_path,)
        
class TimestampToLrcNode:
    """
    ComfyUI節點：將時間戳格式轉換為LRC歌詞格式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lrc_output",)
    FUNCTION = "convert_to_lrc"
    CATEGORY = "text/processing"
    
    def convert_to_lrc(self, input_text):
        """
        將時間戳格式轉換為LRC格式
        輸入格式: >> 0:00-0:04\n>> 文本內容
        輸出格式: [00:00.00]文本內容
        """
        
        lines = input_text.strip().split('\n')
        lrc_lines = []
        current_time = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            
            # 檢查是否為時間戳行 (格式: >> 0:00-0:04)
            time_match = re.match(r'^>>\s*(\d+):(\d+)-(\d+):(\d+)$', line)
            if time_match:
                # 如果之前有累積的文本，先處理它
                if current_time is not None and current_text.strip():
                    lrc_lines.append(f"[{current_time}]{current_text.strip()}")
                
                # 解析開始時間
                start_min = int(time_match.group(1))
                start_sec = int(time_match.group(2))
                current_time = f"{start_min:02d}:{start_sec:02d}.00"
                current_text = ""
                
            # 檢查是否為文本行 (格式: >> 文本內容)
            elif line.startswith('>> '):
                text_content = line[3:].strip()  # 移除 ">> " 前綴
                if text_content:  # 只添加非空文本
                    if current_text:
                        current_text += " " + text_content
                    else:
                        current_text = text_content
            
            # 處理空行或其他格式
            elif line == '' or line == '>>':
                # 空行保持當前狀態，不做處理
                continue
            else:
                # 其他格式的行，嘗試作為文本處理
                if line and current_time is not None:
                    if current_text:
                        current_text += " " + line
                    else:
                        current_text = line
        
        # 處理最後一段文本
        if current_time is not None and current_text.strip():
            lrc_lines.append(f"[{current_time}]{current_text.strip()}")
        
        # 合併結果
        lrc_output = '\n'.join(lrc_lines)
        
        return (lrc_output,)

try:
    import opencc
except ImportError:
    print("請安裝opencc庫: pip install opencc-python-reimplemented")
    opencc = None

class ChineseConverterNode:
    """
    ComfyUI節點：中文簡繁轉換
    使用opencc庫進行高質量轉換
    布林開關控制：True=簡體轉繁體，False=繁體轉簡體
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "simp_to_trad": ("BOOLEAN", {
                    "default": True,
                    "label_on": "簡體→繁體",
                    "label_off": "繁體→簡體"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("converted_text",)
    FUNCTION = "convert_chinese"
    CATEGORY = "text/processing"
    
    def __init__(self):
        """初始化轉換器"""
        if opencc is None:
            self.s2t_converter = None
            self.t2s_converter = None
            print("錯誤：opencc庫未安裝，請執行: pip install opencc-python-reimplemented")
        else:
            try:
                # 簡體轉繁體轉換器
                self.s2t_converter = opencc.OpenCC('s2t.json')
                # 繁體轉簡體轉換器  
                self.t2s_converter = opencc.OpenCC('t2s.json')
            except Exception as e:
                print(f"opencc初始化失敗: {e}")
                self.s2t_converter = None
                self.t2s_converter = None
    
    def convert_chinese(self, input_text, simp_to_trad):
        """
        轉換中文文本
        
        Args:
            input_text: 輸入文本
            simp_to_trad: True=簡體轉繁體，False=繁體轉簡體
            
        Returns:
            轉換後的文本
        """
        
        if not input_text.strip():
            return ("",)
        
        # 檢查opencc是否可用
        if opencc is None:
            error_msg = "錯誤：請先安裝opencc庫\n執行命令: pip install opencc-python-reimplemented"
            print(error_msg)
            return (error_msg,)
        
        try:
            if simp_to_trad:
                # 簡體轉繁體
                if self.s2t_converter is None:
                    self.s2t_converter = opencc.OpenCC('s2t.json')
                converted_text = self.s2t_converter.convert(input_text)
            else:
                # 繁體轉簡體
                if self.t2s_converter is None:
                    self.t2s_converter = opencc.OpenCC('t2s.json')
                converted_text = self.t2s_converter.convert(input_text)
            
            return (converted_text,)
            
        except Exception as e:
            error_msg = f"轉換失敗: {str(e)}"
            print(error_msg)
            return (error_msg,)


    
NODE_CLASS_MAPPINGS = {
    "AudioListGenerator": AudioListGenerator,
    "AudioToFrameCount": AudioToFrameCount,
    "MergeVideoFilename": MergeVideoFilename,    
    "PromptListGenerator": PromptListGenerator, 
    "NumberListGenerator": NumberListGenerator, 
    "AudioListCombine": AudioListCombine, 
    "CeilDivide": CeilDivide,
    "LoadVideoPath": LoadVideoPath,
    "SaveVideoPath": SaveVideoPath,
    "TimestampToLrcNode": TimestampToLrcNode,
    "ChineseConverterNode": ChineseConverterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioListGenerator": "Audio Split to List",
    "AudioToFrameCount": "Audio to Frame Count",
    "MergeVideoFilename": "MergeVideoFilename",
    "PromptListGenerator": "PromptListGenerator",
    "NumberListGenerator": "NumberListGenerator",
    "AudioListCombine": "AudioListCombine",
    "CeilDivide": "CeilDivide",
    "LoadVideoPath": "LoadVideoPath",
    "SaveVideoPath": "SaveVideoPath",
    "TimestampToLrcNode": "TimestampToLrcNode",
    "ChineseConverterNode": "ChineseConverterNode",
}

