# ListHelper Nodes Collection

[中文版本](#中文版本) | [English Version](#english-version)

---

## English Version

### Overview

The **ListHelper** collection is a comprehensive set of custom nodes for ComfyUI that provides powerful list manipulation capabilities. This collection includes audio processing, text splitting, and number generation tools for enhanced workflow automation.

### Included Nodes

1. [AudioListCombine](#audiolistcombine-node)
2. [NumberListGenerator](#numberlistgenerator-node)
3. [PromptSplitByDelimiter](#promptsplitbydelimiter-node)
4. [Qwen_TE_LLM](#qwen-node) - AI Photo Prompt Optimizer
5. [GGUFInference](#ggufinference-node) - GGUF Model Inference with llama-cpp-python
6. [BatchToPSD](#batchtopsd-node) - Convert Batch Images to Multi-Layer PSD

---

## AudioListCombine Node

![Demo](Readme/demo.jpg)

### Overview

The **AudioListCombine** node is a powerful custom node for ComfyUI that allows you to combine multiple audio files from a list into a single audio output. It supports various combination modes and audio processing options.

### Features

- **Multiple Combination Modes**: Concatenate, mix, or overlay audio files
- **Automatic Sample Rate Conversion**: Unifies different sample rates to target rate
- **Channel Normalization**: Automatically handles mono/stereo conversion
- **Crossfade Support**: Smooth transitions between audio segments
- **Audio Normalization**: Optional output level normalization
- **Flexible Input**: Accepts audio lists from Impact Pack or other list-making nodes

### Requirements

- ComfyUI
- Audio list creation nodes (e.g., Impact Pack's MakeAnyList, or custom list nodes)
- Python libraries: `torch`, `torchaudio`

### Usage

#### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_list` | AUDIO | - | List of audio files (from Impact Pack or other list nodes) |
| `combine_mode` | COMBO | "concatenate" | How to combine audio: concatenate/mix/overlay |
| `fade_duration` | FLOAT | 0.0 | Crossfade duration in seconds (0.0-5.0) |
| `normalize_output` | BOOLEAN | True | Whether to normalize output audio |
| `target_sample_rate` | INT | 44100 | Target sample rate for output |

#### Combine Modes

1. **Concatenate**: Join audio files end-to-end in sequence
   - Supports crossfade transitions
   - Maintains chronological order
   - Best for: Creating audio sequences, podcasts, music playlists

2. **Mix**: Average all audio files together
   - Pads shorter files with silence
   - Equal weight blending
   - Best for: Creating audio mashups, averaging multiple takes

3. **Overlay**: Add all audio files together
   - Direct addition (may cause clipping)
   - Preserves original volumes
   - Best for: Adding sound effects, layering instruments

#### Output

| Output | Type | Description |
|--------|------|-------------|
| `audio` | AUDIO | Combined audio result |

### Examples

#### Example 1: Creating a Music Playlist
```
Audio File 1 → 
Audio File 2 → MakeAnyList → AudioListCombine (concatenate, fade=0.5s) → Save Audio
Audio File 3 → 
```

#### Example 2: Mixing Multiple Recordings
```
Recording 1 → 
Recording 2 → MakeAnyList → AudioListCombine (mix, normalize=True) → Save Audio
Recording 3 → 
```

#### Example 3: Adding Sound Effects
```
Background Music → 
Sound Effect 1  → MakeAnyList → AudioListCombine (overlay) → Save Audio
Sound Effect 2  → 
```

---

## NumberListGenerator Node

![Demo](Readme/demo1.jpg)

### Overview
The NumberListGenerator node creates lists of numbers with customizable parameters, supporting both sequential and randomized output. It's perfect for batch processing, parameter sweeping, or any workflow requiring controlled number sequences.

### Features
- **Dual Output Format**: Generates both integer and float lists simultaneously
- **Flexible Range Control**: Set minimum, maximum values and step size
- **Sequential or Random**: Toggle between ordered and shuffled output
- **Reproducible Results**: Optional seed parameter for consistent random generation
- **Count Tracking**: Returns total number of generated values

### Parameters

**Required Inputs:**
- **min_value** (Float): Starting value for the sequence (Range: -10,000 to 10,000, Default: 0.0)
- **max_value** (Float): Maximum value upper bound (Range: -10,000 to 10,000, Default: 10.0)
- **step** (Float): Increment between consecutive values (Range: 0.01 to 1,000, Default: 1.0)
- **count** (Int): Number of values to generate (Range: 1 to 10,000, Default: 10)
- **random** (Boolean): Enable random shuffling of the generated list (Default: False)

**Optional Inputs:**
- **seed** (Int): Random seed for reproducible results when random=True (Range: -1 to 1,000,000, Default: -1)

**Outputs:**
- **int_list**: List of integer values
- **float_list**: List of float values
- **total_count**: Total number of generated values

### Usage Examples

**Sequential Generation:**
```
min_value: 0, max_value: 20, step: 2, count: 10, random: False
Output: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

**Random Generation:**
```
min_value: 1, max_value: 100, step: 5, count: 8, random: True, seed: 42
Output: [16, 1, 31, 6, 21, 11, 26, 36] (shuffled)
```

---

## PromptSplitByDelimiter Node

![Demo](Readme/demo2.jpg)

### Overview

The **PromptSplitByDelimiter** node is a versatile text processing tool that splits text content using customizable delimiters. It supports both simple string delimiters and advanced regular expressions, with optional random ordering and delimiter preservation.

### Features

- **Flexible Delimiter Support**: Use simple strings or regular expressions as delimiters
- **Multi-language Support**: Native support for CJK (Chinese, Japanese, Korean) characters
- **Regular Expression Mode**: Advanced pattern matching for complex splitting rules
- **Delimiter Preservation**: Option to keep delimiters in the output
- **Random Ordering**: Shuffle results with reproducible seed control
- **Advanced Text Processing**: Handle multiple newlines, skip empty segments
- **Selective Processing**: Skip content before first delimiter occurrence

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|--------|-------------|
| `text` | STRING | - | - | Multiline text input to be split |
| `delimiter` | STRING | "," | - | Delimiter string or regex pattern |
| `use_regex` | BOOLEAN | False | - | Enable regular expression mode |
| `keep_delimiter` | BOOLEAN | False | - | Preserve delimiters in output |
| `start_index` | INT | 0 | 0-1000 | Starting index for selection |
| `skip_every` | INT | 0 | 0-10 | Skip every N items |
| `max_count` | INT | 10 | 1-1000 | Maximum items to return |
| `skip_first_index` | BOOLEAN | False | - | Skip content before first delimiter |
| `random_order` | BOOLEAN | False | - | Randomize output order |
| `seed` | INT | 0 | 0-2147483647 | Random seed for reproducible results |

### Output

| Output | Type | Description |
|--------|------|-------------|
| `text_list` | STRING | List of split text segments |
| `total_index` | INT | Total number of segments found |

### Usage Examples

#### Example 1: Simple Comma Splitting
```
Input: "apple,banana,cherry,date"
Delimiter: ","
Output: ["apple", "banana", "cherry", "date"]
```

#### Example 2: Chinese Chapter Splitting
```
Input: "前言第一章内容第二章内容第三章结尾"
Delimiter: "第.*?章" (regex mode)
Output: ["前言", "内容", "内容", "结尾"]
```

#### Example 3: Delimiter Preservation
```
Input: "AAA//BBB//CCC"
Delimiter: "//"
Keep Delimiter: True
Output: ["AAA", "//BBB", "//CCC"]
```

#### Example 4: Random Chapter Selection
```
Input: "Chapter1\nChapter2\nChapter3\nChapter4"
Delimiter: "\n"
Random Order: True
Max Count: 2
Output: ["Chapter3", "Chapter1"] (randomized)
```

### Advanced Features

#### Regular Expression Support
- **Pattern Matching**: Use regex patterns for complex delimiter rules
- **CJK Character Support**: `第\d+章` matches "第1章", "第2章", etc.
- **Flexible Patterns**: `(章|節|段)` matches any of "章", "節", or "段"

#### Text Processing Rules
- **Newline Normalization**: Multiple consecutive newlines are treated as single newline
- **Empty Delimiter Handling**: Empty delimiter automatically uses newline as fallback
- **Whitespace Trimming**: Automatic trimming of leading/trailing whitespace

#### Selection and Filtering
- **Index Range**: Select specific range of results using `start_index` and `max_count`
- **Skip Pattern**: Use `skip_every` to select every Nth item
- **First Segment Skip**: Use `skip_first_index` to ignore content before first delimiter

### Use Cases

- **Document Processing**: Split books into chapters, articles into sections
- **Data Extraction**: Extract structured data from formatted text
- **Content Management**: Process multilingual content with CJK support
- **Batch Processing**: Generate lists for downstream processing nodes
- **Random Sampling**: Create randomized content selections

---

## Qwen_TE_LLM Node

### Overview

The **Qwen_TE_LLM** node is an AI-powered text generation node that uses the Qwen3-4B language model with intelligent GPU memory management and CPU offload support for seamless integration with ComfyUI. It can transform simple descriptions into detailed, professional prompts using customizable templates.

### Features

- **Automatic Model Detection**: Automatically finds Qwen safetensors files in `text_encoders` or `clip` folders
- **Smart Memory Management**: Automatically detects GPU memory and chooses optimal loading strategy
  - Full GPU mode (≥7.5GB free): ~26-30 tokens/second
  - CPU Offload mode (<7.5GB free): ~1-2 tokens/second, coexists with other models
- **Template System**: Select from pre-made prompt templates in the `Prompt` folder or use custom prompts
- **Bilingual Support**: Handles both Chinese and English inputs
- **Automatic Config Download**: Downloads required model configuration files from HuggingFace
- **Think Tag Removal**: Automatically removes `<think>...</think>` reasoning tags from output

### Requirements

- **GPU**: NVIDIA GPU with CUDA support (12GB+ recommended, works with less using CPU offload)
- **Model Files**:
  - **Required**: `qwen_3_4b.safetensors` (or similar Qwen3-4B safetensors file)
  - **Location**: Place in either `ComfyUI/models/text_encoders/` or `ComfyUI/models/clip/`
- **Python Packages**: transformers, safetensors, torch

### Model Setup

1. Download Qwen3-4B safetensors from [Hugging Face](https://huggingface.co/Qwen/Qwen3-4B)
   - Recommended filename: `qwen_3_4b.safetensors`
2. Place the file in `ComfyUI/models/text_encoders/` or `ComfyUI/models/clip/`
3. First run will automatically download configuration files to a `_config` subfolder

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_prompt` | STRING | "A girl in a coffee shop" | Your input text/description |
| `prompt_template` | COMBO | "Custom" | Select a template from the Prompt folder or use "Custom" |
| `system_prompt` | STRING | "" | Custom system prompt (used when template is "Custom") |
| `max_new_tokens` | INT | 2048 | Maximum length of generated text |
| `temperature` | FLOAT | 0.7 | Creativity level (0.0-2.0) |
| `do_sample` | BOOLEAN | True | Enable sampling for varied outputs |
| `top_p` | FLOAT | 0.9 | Nucleus sampling threshold |
| `top_k` | INT | 50 | Top-k sampling parameter |

### Template System

The node supports customizable prompt templates stored in the `Prompt` folder as `.md` files:
- **Custom**: Use the `system_prompt` parameter directly
- **Template Files**: Select from available `.md` files in the `Prompt` folder
- Templates automatically replace the `system_prompt` parameter when selected

### Loading Strategies

The node automatically selects the best loading strategy based on available GPU memory:

#### Strategy 1: Full GPU Loading
- **Condition**: ≥7.5GB available GPU memory
- **Performance**: ~26-30 tokens/second
- **Use Case**: Standalone usage or minimal ComfyUI memory usage

#### Strategy 2: CPU Offload
- **Condition**: <7.5GB available GPU memory
- **Performance**: ~1-2 tokens/second (slower but reliable)
- **Use Case**: When ComfyUI has loaded large models
- **Advantage**: No memory conflicts, coexists with other models

#### Strategy 3: CPU Mode
- **Condition**: No CUDA available
- **Use Case**: CPU-only environments

### Output

| Output | Type | Description |
|--------|------|-------------|
| `text` | STRING | Detailed professional photography prompt |

### Usage Examples

#### Example 1: Using a Template
```
User Prompt: "一個女孩在咖啡廳"
Template: (photography template from Prompt folder)
Output: "A young woman sitting by the window in a cozy coffee shop, warm afternoon sunlight streaming through large glass windows creating soft shadows, wearing casual outfit, holding a cup of coffee, wooden table with laptop and notebook, blurred background with other customers, shallow depth of field, bokeh effect, warm color temperature, golden hour lighting..."
```

#### Example 2: Custom System Prompt
```
User Prompt: "a cat sitting on a windowsill"
Template: "Custom"
System Prompt: "Describe the scene in poetic detail"
Output: "A calico cat sitting on a sunlit windowsill, its long fur catching golden-hour light from the right, creating soft shadows across its face and body, wearing a quiet expression of peaceful solitude, surrounded by indoor plants and a wooden bookshelf in the background..."
```

### Memory Management

The node includes intelligent memory management:

1. **Pre-Load Check**: Checks GPU memory before loading
2. **Automatic Cleanup**: Clears CUDA cache if needed
3. **Dynamic Strategy**: Chooses loading strategy based on available memory
4. **Device Distribution**: Shows how model layers are distributed across GPU/CPU
5. **Clear Error Messages**: Provides detailed error information and solutions

### Performance Considerations

- **First Load**: 7-130 seconds depending on hardware and memory
- **Subsequent Loads**: Model stays in memory, near-instant
- **Full GPU**: Fast inference (~26-30 tokens/second)
- **CPU Offload**: Slower inference (~1-2 tokens/second) but prevents crashes
- **Memory Usage**: ~7.5GB GPU memory for full GPU mode

### Troubleshooting

**Model not found**
- Ensure a Qwen safetensors file is in `models/text_encoders/` or `models/clip/`
- Recommended filename: `qwen_3_4b.safetensors` (containing "qwen", "3", and "4b")
- The node automatically searches for compatible Qwen model files

**Out of memory**
- Node will automatically use CPU offload
- If still failing, close other GPU applications
- Consider restarting ComfyUI to free memory

**Slow inference**
- Using CPU offload (expected behavior with low GPU memory)
- For faster inference, free up GPU memory

**Config download fails**
- Check internet connection
- Manually download from Hugging Face if needed

### Testing

A test script is provided: `test_qwen_node.py`

```bash
python test_qwen_node.py
```

This tests model loading, memory management, and inference capabilities.

---

## GGUFInference Node

### Overview

The **GGUFInference** node is a powerful GGUF model inference node that integrates llama-cpp-python for running quantized language models. It supports both text-only and vision-language models with automatic model detection and intelligent memory management.

### Features

- **Automatic Model Detection**: Scans `text_encoders` and `clip` folders for GGUF model files
- **Vision Model Support**: Handles VL (Vision-Language) models with mmproj files
- **Auto-Installation**: Automatically installs llama-cpp-python on Windows (CUDA 12.8 compatible)
  - **Platform**: Currently Windows only
  - **CUDA Version**: Optimized for CUDA 12.8
  - **Python Support**: 3.10, 3.11, 3.12, 3.13
- **Model Download**: Built-in suggested models with one-click download from HuggingFace
- **Template System**: Load prompt templates from `Prompt` folder or use custom prompts
- **Memory Management**: Option to keep model loaded or unload after inference
- **Seed Control**: Reproducible generation with optional seed parameter

### Requirements

- **Operating System**: Windows (for auto-installation feature)
- **CUDA Version**: 12.8 (for auto-installation)
- **Python Version**: 3.10, 3.11, 3.12, or 3.13
- **Model Files**: GGUF format models in `text_encoders` or `clip` folders
- **Python Library**: llama-cpp-python (auto-installed on Windows)

### Auto-Installation Feature

The node includes an **automatic installation system for llama-cpp-python**:

- **Trigger**: Enable the `auto_install_llama_cpp` parameter
- **Platform Support**: Windows only (manual installation required for other platforms)
- **CUDA Support**: Pre-built wheels for CUDA 12.8
- **Process**: Downloads and installs the appropriate wheel from HuggingFace
- **Restart Required**: After installation, restart ComfyUI to activate

### Parameters

**Required Inputs:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | COMBO | - | - | GGUF model file (auto-detected or download option) |
| `prompt` | STRING | "Hello, how are you?" | - | Input text prompt |
| `prompt_template` | COMBO | "Custom" | - | Template from Prompt folder or Custom |
| `system_prompt` | STRING | "" | - | System prompt (used when template is Custom) |
| `max_tokens` | INT | 4096 | 1-8192 | Maximum generation length |
| `temperature` | FLOAT | 0.7 | 0.0-2.0 | Sampling temperature |
| `top_p` | FLOAT | 0.9 | 0.0-1.0 | Nucleus sampling threshold |
| `top_k` | INT | 40 | 0-100 | Top-k sampling parameter |

**Optional Inputs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | INT | 0 | Random seed for reproducible results |
| `keep_model_loaded` | BOOLEAN | False | Keep model in memory after inference |
| `mmproj_file` | COMBO | "No mmproj files" | Vision model mmproj file |
| `image` | IMAGE | - | Input image for vision models |
| `auto_install_llama_cpp` | BOOLEAN | False | Auto-install llama-cpp-python (Windows only) |

**Outputs:**

- **text**: Generated text output
- **used_seed**: The seed value used for generation

### Supported Models

The node includes suggested models with direct download:

**Text Models:**
- Qwen3-4B (Z-Image)
- Qwen3-4B Abliterated

**Vision-Language Models:**
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-7B-Instruct Abliterated
- Huihui-Qwen3-VL-4B-Instruct Abliterated

**MMProj Files:**
- Qwen2.5-VL mmproj
- QwenVL mmproj

### Usage Examples

#### Example 1: Text Generation with Auto-Installation
```
1. Enable "auto_install_llama_cpp" checkbox
2. Run the node once (will install llama-cpp-python)
3. Restart ComfyUI
4. Select a model from dropdown or use "Download: Z-Image"
5. Enter your prompt
6. Generate!
```

#### Example 2: Vision Model Inference
```
Model: Download: Qwen (VL model)
MMProj File: Download: mmproj
Image: [Connect your IMAGE input]
Prompt: "Describe this image in detail"
Output: Detailed description of the input image
```

#### Example 3: Using Templates
```
Model: Qwen3-4B
Prompt Template: [Select from Prompt folder]
User Prompt: "A sunset over mountains"
Output: Processed text based on template instructions
```

### Auto-Installation Details

**Windows + CUDA 12.8:**
- Automatically downloads pre-compiled wheels from HuggingFace
- Supports Python 3.10, 3.11, 3.12, 3.13
- No manual compilation required
- CUDA-accelerated inference ready

**Other Platforms:**
- Manual installation required: `pip install llama-cpp-python`
- The node will display installation instructions if llama-cpp-python is not found

### Vision Model Support

The node **automatically detects VL (Vision-Language) models** by filename:
- Models with "vl" in the filename are treated as vision models
- Vision mode is enabled when both image input and mmproj file are provided
- Automatically falls back to text-only mode if vision requirements aren't met

### Performance Notes

- **GPU Acceleration**: Automatically uses GPU if CUDA is available (`n_gpu_layers=-1`)
- **Context Size**: 8192 tokens context window
- **Memory Management**: Option to keep model loaded for faster subsequent inferences
- **Thinking Tag Removal**: Automatically strips `<think>...</think>` tags from output

### Troubleshooting

**llama-cpp-python not installed:**
- Enable `auto_install_llama_cpp` on Windows
- Or manually install: `pip install llama-cpp-python`

**Model not found:**
- Place GGUF files in `models/text_encoders/` or `models/clip/`
- Or use the built-in download options

**Vision mode not working:**
- Ensure model filename contains "vl"
- Select appropriate mmproj file
- Connect IMAGE input

**Auto-installation failed:**
- Check Python version (must be 3.10-3.13)
- Check internet connection for wheel download
- Verify Windows operating system

---

## BatchToPSD Node

### Overview

The **BatchToPSD** node converts a batch of images into a multi-layer PSD (Photoshop Document) file. It automatically handles transparency, supports RGBA mode, and includes auto-installation of the required `psd-tools` package.

### Features

- **Auto-Install Dependencies**: Automatically installs `psd-tools` package on first use
- **Multi-Layer Support**: Converts batch images to individual PSD layers
- **Transparency Preservation**: Automatically detects and preserves alpha channels
- **Smart Mode Detection**: Uses RGBA mode when transparency is detected, RGB otherwise
- **Layer Control**: Optional layer order reversal
- **First Image Skip**: Skips the first image in batch (typically the merged preview)

### Requirements

- **Python Package**: psd-tools (auto-installed on first use)
- **Image Format**: Supports RGB and RGBA images
- **Input**: Batch of at least 2 images (first image is skipped)

### Auto-Installation

The node includes an **automatic installation system for psd-tools**:

- **Trigger**: Automatically triggers when psd-tools is not found
- **Process**: Installs psd-tools using pip
- **Restart Required**: After installation, restart ComfyUI to activate the package
- **First Use**: The first run will install the package and display a success message

### Parameters

**Required Inputs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | IMAGE | - | Batch of images to convert (minimum 2 images) |
| `filename_prefix` | STRING | "ComfyUI_PSD" | Prefix for output PSD filename |

**Optional Inputs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reverse_layer_order` | BOOLEAN | False | Reverse layer order (bottom to top) |

**Outputs:**

- **message**: Status message with file path or installation instructions

### Usage Examples

#### Example 1: First Time Use (Auto-Installation)
```
1. Connect batch images to BatchToPSD node
2. Run the node (will auto-install psd-tools)
3. See success message: "psd-tools package has been installed successfully. Please RESTART ComfyUI"
4. Restart ComfyUI
5. Run the node again to create PSD file
```

#### Example 2: Normal PSD Creation
```
Input: Batch of 5 images [merged, layer1, layer2, layer3, layer4]
Output: PSD file with 4 layers (first image skipped)
Layers: Layer_1, Layer_2, Layer_3, Layer_4
```

#### Example 3: With Transparency
```
Input: Batch of RGBA images with transparency
Auto-Detection: Node detects alpha channels
PSD Mode: RGBA (preserves transparency)
Output: PSD file with transparent layers
```

#### Example 4: Reverse Layer Order
```
Input: Batch of 4 images
Reverse Layer Order: True
Output: Layers arranged from bottom to top (Layer_3, Layer_2, Layer_1)
```

### How It Works

1. **Package Check**: Checks if psd-tools is installed
2. **Auto-Install**: If not found, automatically installs psd-tools
3. **Restart Prompt**: Prompts user to restart ComfyUI after installation
4. **Image Processing**:
   - Skips first image in batch
   - Converts remaining images to PIL format
   - Detects alpha channels in any layer
5. **Mode Selection**:
   - Uses RGBA mode if any layer has transparency
   - Uses RGB mode otherwise
6. **PSD Creation**:
   - Creates PSD document with appropriate mode
   - Adds each image as a separate layer
   - Names layers sequentially (Layer_1, Layer_2, etc.)
7. **File Output**: Saves PSD to ComfyUI output directory with timestamp

### Output Location

- **Directory**: ComfyUI output folder
- **Filename Format**: `{prefix}_{timestamp}.psd`
- **Example**: `ComfyUI_PSD_20250326_143022.psd`

### Technical Details

**Transparency Handling:**
- Automatically detects RGBA images
- Creates PSD in RGBA mode to preserve transparency
- Prevents transparency loss during layer creation
- Matches Photoshop's behavior for transparent layers

**Image Conversion:**
- Converts torch tensors [H, W, C] with values [0, 1]
- Converts to uint8 [0, 255] for PIL compatibility
- Preserves color accuracy and transparency

**Layer Naming:**
- Sequential naming: Layer_1, Layer_2, Layer_3, etc.
- Can be renamed in Photoshop after creation

### Troubleshooting

**psd-tools installation failed:**
- Check internet connection
- Manually install: `pip install psd-tools`
- Verify pip is up to date: `pip install --upgrade pip`

**First image is missing in PSD:**
- This is intentional - first image in batch is skipped (usually the merged preview)
- Ensure your batch has at least 2 images

**Transparency not preserved:**
- Check if input images have alpha channel (RGBA mode)
- Node automatically uses RGBA mode when alpha is detected

**Cannot open PSD in Photoshop:**
- Ensure psd-tools installed successfully
- Check output file exists in ComfyUI output directory
- Try updating Photoshop to latest version

### Compatibility

- **Photoshop**: Compatible with Adobe Photoshop CS6 and later
- **Other Software**: Works with any software that supports PSD format (GIMP, Affinity Photo, etc.)
- **Color Modes**: RGB and RGBA
- **Bit Depth**: 8-bit per channel

---

## 中文版本

### 概述

**ListHelper** 集合是 ComfyUI 的全面自定義節點集，提供強大的列表操作功能。此集合包含音頻處理、文本分割和數字生成工具，用於增強工作流程自動化。

### 包含的節點

1. [AudioListCombine 音頻列表合併](#audiolistcombine-音頻列表合併節點)
2. [NumberListGenerator 數字列表生成器](#numberlistgenerator-數字列表生成節點)
3. [PromptSplitByDelimiter 提示分割器](#promptsplitbydelimiter-提示分割節點)
4. [Qwen_TE_LLM AI照片提示詞優化器](#Qwen_TE_LLM-ai照片提示詞優化器)
5. [GGUFInference GGUF模型推理](#ggufinference-gguf模型推理節點)
6. [BatchToPSD 批次圖片轉PSD](#batchtopsd-批次圖片轉psd節點)
7. [AudioToFrameCount](#AudioToFrameCount)
8. [AudioSplitToList](#AudioSplitToList)
9. [CeilDivide](#CeilDivide)

---

## AudioListCombine 音頻列表合併節點

### 概述

**AudioListCombine** 節點是 ComfyUI 的強大自定義節點，允許您將音頻清單中的多個音頻文件合併為單一音頻輸出。支持多種合併模式和音頻處理選項。

### 功能特色

- **多種合併模式**：串接、混音或覆疊音頻文件
- **自動採樣率轉換**：統一不同採樣率至目標採樣率
- **聲道標準化**：自動處理單聲道/立體聲轉換
- **交叉淡化支持**：音頻片段間的平滑過渡
- **音頻標準化**：可選的輸出音量標準化
- **靈活輸入**：接受來自 Impact Pack 或其他清單製作節點的音頻清單

### 使用方法

#### 輸入參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `audio_list` | AUDIO | - | 音頻文件清單（來自 Impact Pack 或其他清單節點）|
| `combine_mode` | COMBO | "concatenate" | 合併方式：concatenate/mix/overlay |
| `fade_duration` | FLOAT | 0.0 | 交叉淡化持續時間（秒，0.0-5.0）|
| `normalize_output` | BOOLEAN | True | 是否標準化輸出音頻 |
| `target_sample_rate` | INT | 44100 | 目標輸出採樣率 |

#### 合併模式

1. **Concatenate（串接）**：按順序將音頻文件首尾相連
   - 支持交叉淡化過渡
   - 保持時間順序
   - 適用於：創建音頻序列、播客、音樂播放清單

2. **Mix（混音）**：將所有音頻文件平均混合
   - 較短文件用靜音填充
   - 等權重混合
   - 適用於：創建音頻混搭、平均多個錄音

3. **Overlay（覆疊）**：將所有音頻文件直接相加
   - 直接加法（可能造成削波）
   - 保持原始音量
   - 適用於：添加音效、樂器分層

#### 輸出

| 輸出 | 類型 | 說明 |
|------|------|------|
| `audio` | AUDIO | 合併後的音頻結果 |

### 使用範例

#### 範例 1：創建音樂播放清單
```
音頻文件 1 → 
音頻文件 2 → MakeAnyList → AudioListCombine (concatenate, fade=0.5s) → 保存音頻
音頻文件 3 → 
```

#### 範例 2：混合多個錄音
```
錄音 1 → 
錄音 2 → MakeAnyList → AudioListCombine (mix, normalize=True) → 保存音頻
錄音 3 → 
```

#### 範例 3：添加音效
```
背景音樂 → 
音效 1   → MakeAnyList → AudioListCombine (overlay) → 保存音頻
音效 2   → 
```

---

## NumberListGenerator 數字列表生成節點

### 概述
NumberListGenerator 節點可根據自訂參數創建數字列表，支援有序和隨機輸出。非常適合批次處理、參數掃描或任何需要受控數字序列的工作流程。

### 功能特色
- **雙重輸出格式**: 同時生成整數和浮點數列表
- **靈活範圍控制**: 設定最小值、最大值和步長
- **有序或隨機**: 可切換有序和打亂輸出
- **可重現結果**: 可選種子參數確保隨機生成的一致性
- **計數追蹤**: 返回生成數值的總數

### 參數說明

**必需輸入:**
- **min_value / 最小值** (Float): 序列的起始值 (範圍: -10,000 到 10,000，預設: 0.0)
- **max_value / 最大值** (Float): 最大值上限 (範圍: -10,000 到 10,000，預設: 10.0)
- **step / 步長** (Float): 連續數值間的增量 (範圍: 0.01 到 1,000，預設: 1.0)
- **count / 數量** (Int): 要生成的數值數量 (範圍: 1 到 10,000，預設: 10)
- **random / 隨機** (Boolean): 啟用生成列表的隨機打亂 (預設: False)

**可選輸入:**
- **seed / 種子** (Int): 隨機種子，用於可重現結果 (範圍: -1 到 1,000,000，預設: -1)

**輸出:**
- **int_list / 整數列表**: 整數值列表
- **float_list / 浮點數列表**: 浮點數值列表
- **total_count / 總計數**: 生成數值的總數

### 使用範例

**有序數字生成:**
```
最小值: 0，最大值: 20，步長: 2，數量: 10，隨機: False
輸出: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

**隨機數字生成:**
```
最小值: 1，最大值: 100，步長: 5，數量: 8，隨機: True，種子: 42
輸出: [16, 1, 31, 6, 21, 11, 26, 36] (已打亂)
```

---

## PromptSplitByDelimiter 提示分割節點

### 概述

**PromptSplitByDelimiter** 節點是一個多功能的文本處理工具，使用可自訂的分隔符分割文本內容。支援簡單字符串分隔符和高級正規表示式，並具有可選的隨機排序和分隔符保留功能。

### 功能特色

- **靈活的分隔符支援**：使用簡單字符串或正規表示式作為分隔符
- **多語言支援**：原生支援中日韓（CJK）文字
- **正規表示式模式**：高級模式匹配，用於複雜的分割規則
- **分隔符保留**：可選擇在輸出中保留分隔符
- **隨機排序**：使用可重現的種子控制打亂結果
- **高級文本處理**：處理多個換行符、跳過空白片段
- **選擇性處理**：跳過第一個分隔符出現前的內容

### 參數說明

| 參數 | 類型 | 預設值 | 範圍 | 說明 |
|------|------|--------|------|------|
| `text` | STRING | - | - | 要分割的多行文本輸入 |
| `delimiter` | STRING | "," | - | 分隔符字符串或正規表示式模式 |
| `use_regex` | BOOLEAN | False | - | 啟用正規表示式模式 |
| `keep_delimiter` | BOOLEAN | False | - | 在輸出中保留分隔符 |
| `start_index` | INT | 0 | 0-1000 | 選擇的起始索引 |
| `skip_every` | INT | 0 | 0-10 | 跳過每 N 個項目 |
| `max_count` | INT | 10 | 1-1000 | 返回的最大項目數 |
| `skip_first_index` | BOOLEAN | False | - | 跳過第一個分隔符前的內容 |
| `random_order` | BOOLEAN | False | - | 隨機輸出順序 |
| `seed` | INT | 0 | 0-2147483647 | 可重現結果的隨機種子 |

### 輸出

| 輸出 | 類型 | 說明 |
|------|------|------|
| `text_list` | STRING | 分割的文本片段列表 |
| `total_index` | INT | 找到的片段總數 |

### 使用範例

#### 範例 1：簡單逗號分割
```
輸入: "蘋果,香蕉,櫻桃,棗子"
分隔符: ","
輸出: ["蘋果", "香蕉", "櫻桃", "棗子"]
```

#### 範例 2：中文章節分割
```
輸入: "前言第一章內容第二章內容第三章結尾"
分隔符: "第.*?章" (正規表示式模式)
輸出: ["前言", "內容", "內容", "結尾"]
```

#### 範例 3：分隔符保留
```
輸入: "AAA//BBB//CCC"
分隔符: "//"
保留分隔符: True
輸出: ["AAA", "//BBB", "//CCC"]
```

#### 範例 4：隨機章節選擇
```
輸入: "第一章\n第二章\n第三章\n第四章"
分隔符: "\n"
隨機順序: True
最大數量: 2
輸出: ["第三章", "第一章"] (已隨機化)
```

### 進階功能

#### 正規表示式支援
- **模式匹配**：使用正規表示式模式進行複雜的分隔符規則
- **中日韓文字支援**：`第\d+章` 匹配 "第1章"、"第2章" 等
- **靈活模式**：`(章|節|段)` 匹配 "章"、"節" 或 "段" 中的任何一個

#### 文本處理規則
- **換行標準化**：多個連續換行符被視為單個換行符
- **空分隔符處理**：空分隔符自動使用換行符作為後備
- **空白修剪**：自動修剪前導/尾隨空白

#### 選擇和過濾
- **索引範圍**：使用 `start_index` 和 `max_count` 選擇特定範圍的結果
- **跳過模式**：使用 `skip_every` 選擇每第 N 個項目
- **第一片段跳過**：使用 `skip_first_index` 忽略第一個分隔符前的內容

### 使用案例

- **文檔處理**：將書籍分割為章節，將文章分割為段落
- **數據提取**：從格式化文本中提取結構化數據
- **內容管理**：處理支援中日韓的多語言內容
- **批次處理**：為下游處理節點生成列表
- **隨機抽樣**：創建隨機化的內容選擇

---

## Qwen_TE_LLM AI照片提示詞優化器

### 概述

**Qwen_TE_LLM** 節點是一個 AI 驅動的文本生成節點，使用 Qwen3-4B 語言模型，具備智能 GPU 記憶體管理和 CPU Offload 支援，可與 ComfyUI 無縫整合。可使用自訂模板將簡單描述轉換為詳細、專業的提示詞。

### 功能特色

- **自動模型偵測**：自動在 `text_encoders` 或 `clip` 資料夾中尋找 Qwen safetensors 檔案
- **智能記憶體管理**：自動檢測 GPU 記憶體並選擇最佳載入策略
  - 完全 GPU 模式（可用 ≥7.5GB）：~26-30 tokens/秒
  - CPU Offload 模式（可用 <7.5GB）：~1-2 tokens/秒，可與其他模型共存
- **模板系統**：從 `Prompt` 資料夾選擇預製模板或使用自訂提示詞
- **雙語支援**：處理中文和英文輸入
- **自動配置下載**：從 HuggingFace 自動下載所需的模型配置檔案
- **思考標籤移除**：自動移除 `<think>...</think>` 推理標籤

### 需求

- **GPU**：支援 CUDA 的 NVIDIA GPU（建議 12GB+，記憶體較少時使用 CPU offload）
- **模型檔案**：
  - **必需**：`qwen_3_4b.safetensors`（或類似的 Qwen3-4B safetensors 檔案）
  - **位置**：放在 `ComfyUI/models/text_encoders/` 或 `ComfyUI/models/clip/`
- **Python 套件**：transformers、safetensors、torch

### 模型設置

1. 從 [Hugging Face](https://huggingface.co/Qwen/Qwen3-4B) 下載 Qwen3-4B safetensors
   - 建議檔名：`qwen_3_4b.safetensors`
2. 將檔案放在 `ComfyUI/models/text_encoders/` 或 `ComfyUI/models/clip/`
3. 首次執行會自動下載配置檔案至 `_config` 子資料夾

### 參數說明

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `user_prompt` | STRING | "A girl in a coffee shop" | 您的輸入文字/描述 |
| `prompt_template` | COMBO | "Custom" | 從 Prompt 資料夾選擇模板或使用 "Custom" |
| `system_prompt` | STRING | "" | 自訂系統提示詞（模板為 "Custom" 時使用）|
| `max_new_tokens` | INT | 2048 | 生成文字的最大長度 |
| `temperature` | FLOAT | 0.7 | 創意程度 (0.0-2.0) |
| `do_sample` | BOOLEAN | True | 啟用採樣以產生變化輸出 |
| `top_p` | FLOAT | 0.9 | Nucleus 採樣閾值 |
| `top_k` | INT | 50 | Top-k 採樣參數 |

### 模板系統

節點支援儲存在 `Prompt` 資料夾中的自訂提示詞模板（`.md` 檔案）：
- **Custom**：直接使用 `system_prompt` 參數
- **模板檔案**：從 `Prompt` 資料夾中的 `.md` 檔案選擇
- 選擇模板時會自動取代 `system_prompt` 參數

### 載入策略

節點會根據可用 GPU 記憶體自動選擇最佳載入策略：

#### 策略 1：完全 GPU 載入
- **條件**：可用 GPU 記憶體 ≥7.5GB
- **效能**：~26-30 tokens/秒
- **適用**：獨立使用或 ComfyUI 記憶體佔用較少時

#### 策略 2：CPU Offload
- **條件**：可用 GPU 記憶體 <7.5GB
- **效能**：~1-2 tokens/秒（較慢但可靠）
- **適用**：ComfyUI 已載入大型模型時
- **優點**：無記憶體衝突，可與其他模型共存

#### 策略 3：CPU 模式
- **條件**：無 CUDA 可用
- **適用**：純 CPU 環境

### 輸出

| 輸出 | 類型 | 說明 |
|------|------|------|
| `text` | STRING | 詳細的專業攝影提示詞 |

### 使用範例

#### 範例 1：使用模板
```
使用者提示詞: "一個女孩在咖啡廳"
模板: (Prompt 資料夾中的攝影模板)
輸出: "A young woman sitting by the window in a cozy coffee shop, warm afternoon sunlight streaming through large glass windows creating soft shadows, wearing casual outfit, holding a cup of coffee, wooden table with laptop and notebook, blurred background with other customers, shallow depth of field, bokeh effect, warm color temperature, golden hour lighting..."
```

#### 範例 2：自訂系統提示詞
```
使用者提示詞: "a cat sitting on a windowsill"
模板: "Custom"
系統提示詞: "用詩意的細節描述場景"
輸出: "A calico cat sitting on a sunlit windowsill, its long fur catching golden-hour light from the right, creating soft shadows across its face and body, wearing a quiet expression of peaceful solitude, surrounded by indoor plants and a wooden bookshelf in the background..."
```

### 記憶體管理

節點包含智能記憶體管理：

1. **載入前檢查**：載入前檢查 GPU 記憶體
2. **自動清理**：需要時清理 CUDA 快取
3. **動態策略**：根據可用記憶體選擇載入策略
4. **設備分佈**：顯示模型層在 GPU/CPU 的分佈情況
5. **清楚錯誤訊息**：提供詳細的錯誤資訊和解決方案

### 效能考量

- **首次載入**：7-130 秒（取決於硬體和記憶體）
- **後續載入**：模型保留在記憶體中，幾乎即時
- **完全 GPU**：快速推理（~26-30 tokens/秒）
- **CPU Offload**：較慢推理（~1-2 tokens/秒）但防止崩潰
- **記憶體使用**：完全 GPU 模式約 7.5GB GPU 記憶體

### 疑難排解

**找不到模型**
- 確保 Qwen safetensors 檔案在 `models/text_encoders/` 或 `models/clip/`
- 建議檔名：`qwen_3_4b.safetensors`（包含 "qwen"、"3" 和 "4b"）
- 節點會自動搜尋相容的 Qwen 模型檔案

**記憶體不足**
- 節點會自動使用 CPU offload
- 若仍失敗，關閉其他 GPU 應用程式
- 考慮重啟 ComfyUI 以釋放記憶體

**推理速度慢**
- 正在使用 CPU offload（低 GPU 記憶體時的預期行為）
- 若要更快推理，釋放 GPU 記憶體

**配置下載失敗**
- 檢查網路連線
- 必要時從 Hugging Face 手動下載

### 測試

提供測試腳本：`test_qwen_node.py`

```bash
python test_qwen_node.py
```

此腳本測試模型載入、記憶體管理和推理能力。

---

## GGUFInference GGUF模型推理節點

### 概述

**GGUFInference** 節點是一個強大的 GGUF 模型推理節點,整合 llama-cpp-python 以運行量化語言模型。支援純文本和視覺語言模型,具備自動模型檢測和智能記憶體管理功能。

### 功能特色

- **自動模型檢測**: 掃描 `text_encoders` 和 `clip` 資料夾尋找 GGUF 模型檔案
- **視覺模型支援**: 處理 VL (Vision-Language) 模型與 mmproj 檔案
- **自動安裝**: 在 Windows 上自動安裝 llama-cpp-python (支援 CUDA 12.8)
  - **平台**: 目前僅支援 Windows
  - **CUDA 版本**: 針對 CUDA 12.8 最佳化
  - **Python 支援**: 3.10, 3.11, 3.12, 3.13
- **模型下載**: 內建建議模型,可從 HuggingFace 一鍵下載
- **模板系統**: 從 `Prompt` 資料夾載入提示詞模板或使用自訂提示詞
- **記憶體管理**: 可選擇保持模型載入或推理後卸載
- **種子控制**: 可選種子參數確保生成可重現

### 需求

- **作業系統**: Windows (自動安裝功能)
- **CUDA 版本**: 12.8 (自動安裝)
- **Python 版本**: 3.10, 3.11, 3.12 或 3.13
- **模型檔案**: GGUF 格式模型放在 `text_encoders` 或 `clip` 資料夾
- **Python 函式庫**: llama-cpp-python (Windows 上自動安裝)

### 自動安裝功能

節點包含 **llama-cpp-python 自動安裝系統**:

- **觸發**: 啟用 `auto_install_llama_cpp` 參數
- **平台支援**: 僅 Windows (其他平台需手動安裝)
- **CUDA 支援**: CUDA 12.8 預編譯 wheels
- **流程**: 從 HuggingFace 下載並安裝適當的 wheel
- **需要重啟**: 安裝後需重啟 ComfyUI 以啟用

### 參數說明

**必需輸入:**

| 參數 | 類型 | 預設值 | 範圍 | 說明 |
|------|------|--------|------|------|
| `model` | COMBO | - | - | GGUF 模型檔案 (自動檢測或下載選項) |
| `prompt` | STRING | "Hello, how are you?" | - | 輸入文字提示詞 |
| `prompt_template` | COMBO | "Custom" | - | Prompt 資料夾的模板或 Custom |
| `system_prompt` | STRING | "" | - | 系統提示詞 (模板為 Custom 時使用) |
| `max_tokens` | INT | 4096 | 1-8192 | 最大生成長度 |
| `temperature` | FLOAT | 0.7 | 0.0-2.0 | 取樣溫度 |
| `top_p` | FLOAT | 0.9 | 0.0-1.0 | Nucleus 取樣閾值 |
| `top_k` | INT | 40 | 0-100 | Top-k 取樣參數 |

**可選輸入:**

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `seed` | INT | 0 | 可重現結果的隨機種子 |
| `keep_model_loaded` | BOOLEAN | False | 推理後保持模型在記憶體中 |
| `mmproj_file` | COMBO | "No mmproj files" | 視覺模型 mmproj 檔案 |
| `image` | IMAGE | - | 視覺模型的輸入圖片 |
| `auto_install_llama_cpp` | BOOLEAN | False | 自動安裝 llama-cpp-python (僅 Windows) |

**輸出:**

- **text**: 生成的文字輸出
- **used_seed**: 使用的種子值

### 支援的模型

節點包含建議模型可直接下載:

**文本模型:**
- Qwen3-4B (Z-Image)
- Qwen3-4B Abliterated

**視覺語言模型:**
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-7B-Instruct Abliterated
- Huihui-Qwen3-VL-4B-Instruct Abliterated

**MMProj 檔案:**
- Qwen2.5-VL mmproj
- QwenVL mmproj

### 使用範例

#### 範例 1: 使用自動安裝進行文字生成
```
1. 啟用 "auto_install_llama_cpp" 選項
2. 執行節點一次 (將安裝 llama-cpp-python)
3. 重啟 ComfyUI
4. 從下拉選單選擇模型或使用 "Download: Z-Image"
5. 輸入您的提示詞
6. 開始生成!
```

#### 範例 2: 視覺模型推理
```
模型: Download: Qwen (VL 模型)
MMProj 檔案: Download: mmproj
圖片: [連接您的 IMAGE 輸入]
提示詞: "詳細描述這張圖片"
輸出: 輸入圖片的詳細描述
```

#### 範例 3: 使用模板
```
模型: Qwen3-4B
提示詞模板: [從 Prompt 資料夾選擇]
使用者提示詞: "山上的日落"
輸出: 根據模板指示處理的文字
```

### 自動安裝詳情

**Windows + CUDA 12.8:**
- 自動從 HuggingFace 下載預編譯 wheels
- 支援 Python 3.10, 3.11, 3.12, 3.13
- 無需手動編譯
- CUDA 加速推理已就緒

**其他平台:**
- 需要手動安裝: `pip install llama-cpp-python`
- 若未找到 llama-cpp-python,節點會顯示安裝說明

### 視覺模型支援

節點 **自動檢測 VL (Vision-Language) 模型** 透過檔名:
- 檔名中包含 "vl" 的模型被視為視覺模型
- 當提供圖片輸入和 mmproj 檔案時啟用視覺模式
- 若未滿足視覺需求會自動回退到純文本模式

### 效能說明

- **GPU 加速**: 若 CUDA 可用會自動使用 GPU (`n_gpu_layers=-1`)
- **上下文大小**: 8192 tokens 上下文視窗
- **記憶體管理**: 可選擇保持模型載入以加快後續推理
- **思考標籤移除**: 自動移除輸出中的 `<think>...</think>` 標籤

### 疑難排解

**llama-cpp-python 未安裝:**
- 在 Windows 上啟用 `auto_install_llama_cpp`
- 或手動安裝: `pip install llama-cpp-python`

**找不到模型:**
- 將 GGUF 檔案放在 `models/text_encoders/` 或 `models/clip/`
- 或使用內建下載選項

**視覺模式無法運作:**
- 確保模型檔名包含 "vl"
- 選擇適當的 mmproj 檔案
- 連接 IMAGE 輸入

**自動安裝失敗:**
- 檢查 Python 版本 (必須是 3.10-3.13)
- 檢查網路連線以下載 wheel
- 確認 Windows 作業系統

---

## BatchToPSD 批次圖片轉PSD節點

### 概述

**BatchToPSD** 節點將批次圖片轉換為多圖層 PSD (Photoshop Document) 檔案。自動處理透明度,支援 RGBA 模式,並包含自動安裝所需 `psd-tools` 套件的功能。

### 功能特色

- **自動安裝依賴套件**: 首次使用時自動安裝 `psd-tools` 套件
- **多圖層支援**: 將批次圖片轉換為獨立的 PSD 圖層
- **透明度保留**: 自動偵測並保留 alpha 通道
- **智能模式偵測**: 偵測到透明度時使用 RGBA 模式,否則使用 RGB
- **圖層控制**: 可選的圖層順序反轉
- **跳過第一張圖片**: 跳過批次中的第一張圖片 (通常是合併預覽)

### 需求

- **Python 套件**: psd-tools (首次使用時自動安裝)
- **圖片格式**: 支援 RGB 和 RGBA 圖片
- **輸入**: 至少 2 張圖片的批次 (第一張圖片會被跳過)

### 自動安裝

節點包含 **psd-tools 自動安裝系統**:

- **觸發**: 未找到 psd-tools 時自動觸發
- **流程**: 使用 pip 安裝 psd-tools
- **需要重啟**: 安裝後需重啟 ComfyUI 以啟用套件
- **首次使用**: 首次執行會安裝套件並顯示成功訊息

### 參數說明

**必需輸入:**

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `images` | IMAGE | - | 要轉換的批次圖片 (最少 2 張) |
| `filename_prefix` | STRING | "ComfyUI_PSD" | 輸出 PSD 檔案名稱前綴 |

**可選輸入:**

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `reverse_layer_order` | BOOLEAN | False | 反轉圖層順序 (由下往上) |

**輸出:**

- **message**: 狀態訊息,包含檔案路徑或安裝說明

### 使用範例

#### 範例 1: 首次使用 (自動安裝)
```
1. 將批次圖片連接到 BatchToPSD 節點
2. 執行節點 (會自動安裝 psd-tools)
3. 看到成功訊息: "psd-tools 套件已成功安裝。請重啟 ComfyUI"
4. 重啟 ComfyUI
5. 再次執行節點以建立 PSD 檔案
```

#### 範例 2: 正常 PSD 建立
```
輸入: 5 張圖片的批次 [合併圖, 圖層1, 圖層2, 圖層3, 圖層4]
輸出: 包含 4 個圖層的 PSD 檔案 (第一張圖片被跳過)
圖層: Layer_1, Layer_2, Layer_3, Layer_4
```

#### 範例 3: 帶透明度的圖片
```
輸入: 帶透明度的 RGBA 圖片批次
自動偵測: 節點偵測到 alpha 通道
PSD 模式: RGBA (保留透明度)
輸出: 包含透明圖層的 PSD 檔案
```

#### 範例 4: 反轉圖層順序
```
輸入: 4 張圖片的批次
反轉圖層順序: True
輸出: 圖層由下往上排列 (Layer_3, Layer_2, Layer_1)
```

### 運作方式

1. **套件檢查**: 檢查是否安裝 psd-tools
2. **自動安裝**: 若未找到,自動安裝 psd-tools
3. **重啟提示**: 安裝後提示使用者重啟 ComfyUI
4. **圖片處理**:
   - 跳過批次中的第一張圖片
   - 將剩餘圖片轉換為 PIL 格式
   - 偵測任何圖層中的 alpha 通道
5. **模式選擇**:
   - 若任何圖層有透明度則使用 RGBA 模式
   - 否則使用 RGB 模式
6. **PSD 建立**:
   - 使用適當模式建立 PSD 文件
   - 將每張圖片加入為獨立圖層
   - 依序命名圖層 (Layer_1, Layer_2 等)
7. **檔案輸出**: 儲存 PSD 至 ComfyUI 輸出目錄,包含時間戳記

### 輸出位置

- **目錄**: ComfyUI 輸出資料夾
- **檔案名稱格式**: `{前綴}_{時間戳記}.psd`
- **範例**: `ComfyUI_PSD_20250326_143022.psd`

### 技術細節

**透明度處理:**
- 自動偵測 RGBA 圖片
- 建立 RGBA 模式的 PSD 以保留透明度
- 防止圖層建立時透明度遺失
- 符合 Photoshop 對透明圖層的處理方式

**圖片轉換:**
- 轉換 torch 張量 [H, W, C],數值範圍 [0, 1]
- 轉換為 uint8 [0, 255] 以相容 PIL
- 保留顏色準確度和透明度

**圖層命名:**
- 依序命名: Layer_1, Layer_2, Layer_3 等
- 可在 Photoshop 中建立後重新命名

### 疑難排解

**psd-tools 安裝失敗:**
- 檢查網路連線
- 手動安裝: `pip install psd-tools`
- 確認 pip 已更新: `pip install --upgrade pip`

**PSD 中缺少第一張圖片:**
- 這是預期行為 - 批次中的第一張圖片會被跳過 (通常是合併預覽)
- 確保批次至少有 2 張圖片

**透明度未保留:**
- 檢查輸入圖片是否有 alpha 通道 (RGBA 模式)
- 節點會在偵測到 alpha 時自動使用 RGBA 模式

**無法在 Photoshop 中開啟 PSD:**
- 確保 psd-tools 已成功安裝
- 檢查輸出檔案是否存在於 ComfyUI 輸出目錄
- 嘗試更新 Photoshop 到最新版本

### 相容性

- **Photoshop**: 相容 Adobe Photoshop CS6 及更新版本
- **其他軟體**: 適用於任何支援 PSD 格式的軟體 (GIMP、Affinity Photo 等)
- **色彩模式**: RGB 和 RGBA
- **位元深度**: 每通道 8-bit

---

### Performance Considerations / 性能考慮
- **Memory Usage**: Large audio files and long text strings may require significant RAM
- **Processing Speed**: Regular expressions may be slower than simple string operations
- **File Formats**: AudioListCombine supports all formats compatible with torchaudio

### 中文技術說明
- **記憶體使用**：大型音頻文件和長文本字符串可能需要大量 RAM
- **處理速度**：正規表示式可能比簡單字符串操作慢
- **文件格式**：AudioListCombine 支援所有與 torchaudio 兼容的格式

### Common Issues / 常見問題

**Audio list is empty / 音頻列表為空**
- Ensure list creation nodes have connected inputs / 確保列表創建節點已連接輸入

**Regular expression errors / 正規表示式錯誤**
- Check pattern syntax, node will fallback to string mode / 檢查模式語法，節點將回退到字符串模式

**Memory issues with large files / 大文件記憶體問題**
- Process files in smaller batches / 以較小批次處理文件

## AudioToFrameCount

![Demo](Readme/demo3.jpg)

### 功能特色

**輸入音檔以串接圖片數目**：依照所需影片格數計算音檔長度，輸入為音檔，輸出為一固定值，可用於重複單一圖片配合音檔長度

## CeilDivide

![Demo](Readme/demo4.jpg)

### 功能特色

**無條件進位**：將AB相除結果無條件進位，以避免因尾數被捨去迴圈數目不足，導致多段影片或音檔分離時，末段音檔未被採樣

## AudioSplitToList

![Demo](Readme/demo5.jpg)

### 功能特色

**分割音檔為清單**：將聲音檔分割為清單，長篇數字人時可分段採樣，目前已經更新音檔處理的邏輯，增加FADE功能

**必需輸入:**
- **videofps / 畫格** (Float): 每秒多少格
- **samplefps / 分段採樣格** (Int): 每段採樣多少格，範例，如果是25格，分段採樣格是75，則音檔將會分隔為每三秒一個單位
- **pad_last_segment / 補足音檔** (Boolean): 將音檔長度插入空白，對齊最後一格的分段採樣 (預設: False)

**輸出:**
- **cycle / 整數**: 分割為多少段
- **audio_list / 音檔清單**: 分割完成的音檔，可直接放入audio輸入，會依序處理


## License / 授權

MIT License

## Contributing / 貢獻

歡迎提交 Issue 和 Pull Request！
Welcome to submit Issues and Pull Requests!

## Support / 支援

如有問題請在 GitHub Issues 中回報。
For questions, please report in GitHub Issues.