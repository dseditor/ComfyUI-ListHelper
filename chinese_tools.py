"""
Chinese Text Processing Tools for ComfyUI
Includes Chinese converter, translator, and auto white balance nodes
Dependencies are installed on first use to avoid import failures
"""

import sys
import subprocess
import os

# Lazy import flags
_opencc_available = None
_argostranslate_available = None

def check_and_install_opencc():
    """Check and install opencc-python-reimplemented if needed"""
    global _opencc_available
    
    if _opencc_available is not None:
        return _opencc_available
    
    try:
        import opencc
        _opencc_available = True
        print("✓ opencc is available")
        return True
    except ImportError:
        print("⚠️ opencc not found. Installing opencc-python-reimplemented...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "opencc-python-reimplemented"
            ])
            print("✓ opencc-python-reimplemented installed successfully")
            print("⚠️ Please restart ComfyUI to use Chinese Converter node")
            _opencc_available = False  # Needs restart
            return False
        except Exception as e:
            print(f"❌ Failed to install opencc: {e}")
            _opencc_available = False
            return False

def check_and_install_argostranslate():
    """Check and install argostranslate if needed"""
    global _argostranslate_available
    
    if _argostranslate_available is not None:
        return _argostranslate_available
    
    try:
        import argostranslate.package
        import argostranslate.translate
        _argostranslate_available = True
        print("✓ argostranslate is available")
        return True
    except ImportError:
        print("⚠️ argostranslate not found. Installing...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "argostranslate"
            ])
            print("✓ argostranslate installed successfully")
            print("⚠️ Please restart ComfyUI to use Chinese Translate node")
            _argostranslate_available = False  # Needs restart
            return False
        except Exception as e:
            print(f"❌ Failed to install argostranslate: {e}")
            _argostranslate_available = False
            return False


class ChineseConverterNode:
    """
    Chinese Simplified/Traditional Converter Node
    Uses opencc library for high-quality conversion
    Boolean switch: True=Simplified to Traditional, False=Traditional to Simplified
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
                    "label_on": "Simp→Trad",
                    "label_off": "Trad→Simp"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("converted_text",)
    FUNCTION = "convert_chinese"
    CATEGORY = "ListHelper/Chinese"
    
    def __init__(self):
        """Initialize converters"""
        self.s2t_converter = None
        self.t2s_converter = None
        self.opencc_ready = check_and_install_opencc()
        
        if self.opencc_ready:
            try:
                import opencc
                # Simplified to Traditional converter
                self.s2t_converter = opencc.OpenCC('s2t')
                # Traditional to Simplified converter
                self.t2s_converter = opencc.OpenCC('t2s')
            except Exception as e:
                print(f"⚠️ opencc initialization failed: {e}")
                self.s2t_converter = None
                self.t2s_converter = None
    
    def convert_chinese(self, input_text, simp_to_trad):
        """
        Convert Chinese text
        
        Args:
            input_text: Input text
            simp_to_trad: True=Simplified to Traditional, False=Traditional to Simplified
            
        Returns:
            Converted text
        """
        
        if not input_text.strip():
            return ("",)
        
        # Check if opencc is ready
        if not self.opencc_ready:
            error_msg = "ERROR: opencc is not installed.\n\nPlease restart ComfyUI to complete installation,\nor manually install: pip install opencc-python-reimplemented"
            print(error_msg)
            return (error_msg,)
        
        try:
            import opencc
            
            if simp_to_trad:
                # Simplified to Traditional
                if self.s2t_converter is None:
                    self.s2t_converter = opencc.OpenCC('s2t.json')
                converted_text = self.s2t_converter.convert(input_text)
            else:
                # Traditional to Simplified
                if self.t2s_converter is None:
                    self.t2s_converter = opencc.OpenCC('t2s.json')
                converted_text = self.t2s_converter.convert(input_text)
            
            return (converted_text,)
            
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            print(error_msg)
            return (error_msg,)


class ArgosTranslateNode:
    """
    Chinese to English Translation Node using Argos Translate
    Supports Traditional and Simplified Chinese
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
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate_text"
    CATEGORY = "ListHelper/Chinese"
    
    def __init__(self):
        """Initialize translator"""
        self.translator_available = check_and_install_argostranslate()
        
        if self.translator_available:
            try:
                import argostranslate.package
                # Update package index
                argostranslate.package.update_package_index()
            except Exception as e:
                print(f"⚠️ Failed to update language package index: {e}")
    
    def install_language_package(self):
        """Install Chinese to English language package"""
        if not self.translator_available:
            return False
        
        try:
            import argostranslate.package
            
            # Get available packages
            available_packages = argostranslate.package.get_available_packages()
            
            # Find Chinese to English package
            target_package = None
            for package in available_packages:
                # Check for Chinese to English package
                if (package.from_code == 'zh' and package.to_code == 'en') or \
                   (package.from_code == 'zh-cn' and package.to_code == 'en') or \
                   (package.from_code == 'zh-tw' and package.to_code == 'en'):
                    target_package = package
                    break
            
            if target_package is None:
                print("❌ Chinese to English language package not found")
                return False
            
            # Check if already installed
            installed_packages = argostranslate.package.get_installed_packages()
            for installed_package in installed_packages:
                if (installed_package.from_code == target_package.from_code and 
                    installed_package.to_code == target_package.to_code):
                    print(f"✓ Language package already installed: {target_package.from_code} → {target_package.to_code}")
                    return True
            
            # Download and install language package
            print(f"⬇️ Downloading language package: {target_package.from_code} → {target_package.to_code}")
            download_path = target_package.download()
            print(f"📦 Installing language package...")
            argostranslate.package.install_from_path(download_path)
            print("✓ Language package installed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Language package installation failed: {e}")
            return False
    
    def translate_text(self, input_text):
        """
        Translate text (auto-install language package)
        
        Args:
            input_text: Input Chinese text (Traditional or Simplified)
            
        Returns:
            Translated English text
        """
        
        if not input_text.strip():
            return ("",)
        
        # Check if argostranslate is available
        if not self.translator_available:
            error_msg = "ERROR: argostranslate is not installed.\n\nPlease restart ComfyUI to complete installation,\nor manually install: pip install argostranslate"
            print(error_msg)
            return (error_msg,)
        
        try:
            import argostranslate.package
            import argostranslate.translate
            
            # Auto-install language package
            if not self.install_language_package():
                error_msg = "Language package installation failed. Please check your internet connection."
                return (error_msg,)
            
            # Get installed packages
            installed_packages = argostranslate.package.get_installed_packages()
            
            # Find Chinese to English translator
            translator = None
            for package in installed_packages:
                if package.from_code in ['zh', 'zh-cn', 'zh-tw'] and package.to_code == 'en':
                    translator = package
                    break
            
            if translator is None:
                error_msg = "Chinese to English translator not found. Language package may have failed to install."
                print(error_msg)
                return (error_msg,)
            
            # Perform translation
            print(f"🌐 Using translator: {translator.from_code} → {translator.to_code}")
            translated_text = argostranslate.translate.translate(input_text, translator.from_code, translator.to_code)
            
            return (translated_text,)
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            print(error_msg)
            return (error_msg,)


# Standard imports for image processing
import torch
import numpy as np

try:
    import cv2
    cv2_available = True
except ImportError:
    print("⚠️ OpenCV (cv2) not found. Auto White Balance node will not be available.")
    print("   Install with: pip install opencv-python")
    cv2_available = False

try:
    from PIL import Image
    pil_available = True
except ImportError:
    print("⚠️ PIL not found. Auto White Balance node will not be available.")
    print("   Install with: pip install Pillow")
    pil_available = False


class AutoWhiteBalanceNode:
    """
    Auto White Balance Adjustment Node for ComfyUI
    Supports multiple white balance algorithms to correct color cast
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["gray_world", "white_patch", "simple_avg", "histogram_stretch"], {
                    "default": "gray_world"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            },
            "optional": {
                "preserve_brightness": ("BOOLEAN", {"default": True}),
                "clip_values": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_white_balance"
    CATEGORY = "ListHelper/Tools"
    
    def tensor_to_cv2(self, tensor_image):
        """Convert ComfyUI tensor format to OpenCV format"""
        # tensor_image shape: [batch, height, width, channels]
        if len(tensor_image.shape) == 4:
            tensor_image = tensor_image[0]  # Take first image
        
        # Convert to numpy array and scale to 0-255 range
        image_np = tensor_image.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # Convert to BGR format (OpenCV default)
        if image_np.shape[2] == 3:  # RGB
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:  # Assume already BGR
            image_bgr = image_np
            
        return image_bgr
    
    def cv2_to_tensor(self, cv2_image):
        """Convert OpenCV format to ComfyUI tensor format"""
        # Convert back to RGB
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
            image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2_image
        
        # Normalize to 0-1 range
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor format [1, height, width, channels]
        tensor_image = torch.from_numpy(image_normalized).unsqueeze(0)
        
        return tensor_image
    
    def gray_world_algorithm(self, image):
        """Gray World Algorithm - assumes average color should be gray"""
        # Calculate average for each channel
        avg_b = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_r = np.mean(image[:, :, 2])
        
        # Calculate overall average brightness
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        # Calculate adjustment coefficients
        scale_b = avg_gray / avg_b if avg_b > 0 else 1.0
        scale_g = avg_gray / avg_g if avg_g > 0 else 1.0
        scale_r = avg_gray / avg_r if avg_r > 0 else 1.0
        
        # Apply adjustment
        result = image.astype(np.float32)
        result[:, :, 0] *= scale_b
        result[:, :, 1] *= scale_g
        result[:, :, 2] *= scale_r
        
        return result
    
    def white_patch_algorithm(self, image):
        """White Patch Algorithm - assumes brightest point should be white"""
        # Find maximum value for each channel
        max_b = np.max(image[:, :, 0])
        max_g = np.max(image[:, :, 1])
        max_r = np.max(image[:, :, 2])
        
        # Calculate adjustment coefficients
        scale_b = 255.0 / max_b if max_b > 0 else 1.0
        scale_g = 255.0 / max_g if max_g > 0 else 1.0
        scale_r = 255.0 / max_r if max_r > 0 else 1.0
        
        # Apply adjustment
        result = image.astype(np.float32)
        result[:, :, 0] *= scale_b
        result[:, :, 1] *= scale_g
        result[:, :, 2] *= scale_r
        
        return result
    
    def simple_average_algorithm(self, image):
        """Simple Average Algorithm - equalizes RGB channel averages"""
        # Calculate average for each channel
        avg_b = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_r = np.mean(image[:, :, 2])
        
        # Use green channel as reference (human eye is most sensitive to green)
        reference = avg_g
        
        # Calculate adjustment coefficients
        scale_b = reference / avg_b if avg_b > 0 else 1.0
        scale_g = 1.0  # Green channel unchanged
        scale_r = reference / avg_r if avg_r > 0 else 1.0
        
        # Apply adjustment
        result = image.astype(np.float32)
        result[:, :, 0] *= scale_b
        result[:, :, 1] *= scale_g
        result[:, :, 2] *= scale_r
        
        return result
    
    def histogram_stretch_algorithm(self, image):
        """Histogram Stretch Algorithm"""
        result = image.astype(np.float32)
        
        for i in range(3):  # For each color channel
            channel = result[:, :, i]
            
            # Calculate 1% and 99% percentiles, ignoring extreme values
            p1 = np.percentile(channel, 1)
            p99 = np.percentile(channel, 99)
            
            # Stretch to 0-255 range
            if p99 > p1:
                channel = (channel - p1) * 255.0 / (p99 - p1)
                result[:, :, i] = np.clip(channel, 0, 255)
        
        return result
    
    def preserve_image_brightness(self, original, adjusted):
        """Preserve original image brightness"""
        # Calculate original image brightness
        original_lab = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_BGR2LAB)
        original_brightness = np.mean(original_lab[:, :, 0])
        
        # Calculate adjusted image brightness
        adjusted_lab = cv2.cvtColor(adjusted.astype(np.uint8), cv2.COLOR_BGR2LAB)
        adjusted_brightness = np.mean(adjusted_lab[:, :, 0])
        
        # Adjust brightness
        if adjusted_brightness > 0:
            brightness_ratio = original_brightness / adjusted_brightness
            adjusted_lab[:, :, 0] = np.clip(adjusted_lab[:, :, 0] * brightness_ratio, 0, 255)
            
            # Convert back to BGR
            result = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
            return result.astype(np.float32)
        
        return adjusted
    
    def adjust_white_balance(self, image, method="gray_world", strength=1.0, 
                           preserve_brightness=True, clip_values=True):
        """Main white balance adjustment function"""
        
        # Check dependencies
        if not cv2_available or not pil_available:
            error_msg = "ERROR: Required dependencies not installed.\n\nPlease install:\npip install opencv-python Pillow"
            print(error_msg)
            return (image,)  # Return original image
        
        # Convert input format
        cv2_image = self.tensor_to_cv2(image)
        original_image = cv2_image.copy()
        
        # Select algorithm
        if method == "gray_world":
            adjusted = self.gray_world_algorithm(cv2_image)
        elif method == "white_patch":
            adjusted = self.white_patch_algorithm(cv2_image)
        elif method == "simple_avg":
            adjusted = self.simple_average_algorithm(cv2_image)
        elif method == "histogram_stretch":
            adjusted = self.histogram_stretch_algorithm(cv2_image)
        else:
            adjusted = cv2_image.astype(np.float32)
        
        # Apply strength adjustment
        if strength != 1.0:
            adjusted = original_image.astype(np.float32) * (1.0 - strength) + adjusted * strength
        
        # Preserve brightness
        if preserve_brightness:
            adjusted = self.preserve_image_brightness(original_image, adjusted)
        
        # Clip value range
        if clip_values:
            adjusted = np.clip(adjusted, 0, 255)
        
        # Convert back to tensor format
        result_tensor = self.cv2_to_tensor(adjusted.astype(np.uint8))
        
        return (result_tensor,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ChineseConverter": ChineseConverterNode,
    "ChineseTranslate": ArgosTranslateNode,
    "AutoWhiteBalance": AutoWhiteBalanceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChineseConverter": "Chinese Converter (Simp⇄Trad)",
    "ChineseTranslate": "Chinese to English Translate",
    "AutoWhiteBalance": "Auto White Balance",
}
