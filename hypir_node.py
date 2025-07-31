"""
HYPIR图像修复ComfyUI节点实现
基于HYPIR源码，不重复造轮子
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List

# 导入ComfyUI folder_paths管理
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("⚠️  ComfyUI folder_paths不可用，使用手动路径管理")

# 导入HYPIR相关模块
try:
    # ComfyUI环境中的相对导入
    from .HYPIR.enhancer.sd2 import SD2Enhancer
    from .HYPIR.utils.common import SuppressLogging
except ImportError:
    # 直接运行时的绝对导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from HYPIR.enhancer.sd2 import SD2Enhancer
    from HYPIR.utils.common import SuppressLogging


def get_diffuser_models() -> List[str]:
    """
    获取ComfyUI中可用的Diffusers模型列表
    """
    models = []
    
    if COMFYUI_AVAILABLE:
        try:
            # 获取diffusers模型目录
            diffuser_path = folder_paths.get_folder_paths("diffusers")[0]
            if os.path.exists(diffuser_path):
                for item in os.listdir(diffuser_path):
                    model_path = os.path.join(diffuser_path, item)
                    if os.path.isdir(model_path):
                        # 检查是否包含stable-diffusion-2相关模型
                        if "stable-diffusion-2" in item.lower() or "sd2" in item.lower():
                            models.append(item)
                        # 也添加其他模型供选择
                        elif any(file.endswith('.json') for file in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, file))):
                            models.append(item)
        except Exception as e:
            print(f"⚠️  获取Diffusers模型列表失败: {e}")
    
    # 添加默认选项
    if not models:
        models = ["stable-diffusion-2-base", "stabilityai/stable-diffusion-2-base"]
    elif "stable-diffusion-2-base" not in models:
        models.insert(0, "stable-diffusion-2-base")
    
    return models


def get_hypir_weights() -> List[str]:
    """
    获取可用的HYPIR权重文件列表
    """
    weights = []
    
    if COMFYUI_AVAILABLE:
        try:
            # 检查是否存在HYPIR文件夹
            models_path = folder_paths.models_dir
            hypir_path = os.path.join(models_path, "HYPIR")
            
            if os.path.exists(hypir_path):
                for file in os.listdir(hypir_path):
                    if file.endswith(('.pth', '.ckpt', '.safetensors')):
                        weights.append(file)
            
            # 如果HYPIR文件夹不存在，检查其他可能的位置
            if not weights:
                possible_paths = [
                    os.path.join(models_path, "checkpoints"),
                    os.path.join(models_path, "custom"),
                    "pretrained_models"  # 相对路径
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        for file in os.listdir(path):
                            if "hypir" in file.lower() and file.endswith(('.pth', '.ckpt', '.safetensors')):
                                weights.append(os.path.join(os.path.basename(path), file))
                                
        except Exception as e:
            print(f"⚠️  获取HYPIR权重列表失败: {e}")
    
    # 添加默认选项
    if not weights:
        weights = ["HYPIR_sd2.pth"]
    
    return weights


def resolve_model_path(model_name: str) -> str:
    """
    解析模型路径，支持ComfyUI标准路径管理
    """
    if not COMFYUI_AVAILABLE:
        return model_name
    
    try:
        # 如果是完整路径，直接返回
        if os.path.isabs(model_name) and os.path.exists(model_name):
            return model_name
        
        # 检查diffusers目录
        diffuser_paths = folder_paths.get_folder_paths("diffusers")
        for base_path in diffuser_paths:
            full_path = os.path.join(base_path, model_name)
            if os.path.exists(full_path):
                return full_path
        
        # 如果找不到，返回原始名称（可能是HuggingFace模型ID）
        return model_name
        
    except Exception as e:
        print(f"⚠️  解析模型路径失败: {e}")
        return model_name


def resolve_weight_path(weight_name: str) -> str:
    """
    解析HYPIR权重文件路径
    """
    if not COMFYUI_AVAILABLE:
        return weight_name
    
    try:
        # 如果是完整路径，直接返回
        if os.path.isabs(weight_name) and os.path.exists(weight_name):
            return weight_name
        
        models_path = folder_paths.models_dir
        
        # 优先检查HYPIR专用目录
        hypir_path = os.path.join(models_path, "HYPIR", weight_name)
        if os.path.exists(hypir_path):
            return hypir_path
        
        # 检查其他可能位置
        possible_paths = [
            os.path.join(models_path, "checkpoints", weight_name),
            os.path.join(models_path, "custom", weight_name),
            os.path.join("pretrained_models", weight_name),  # 相对路径
            weight_name  # 当前目录
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果都找不到，返回原始路径
        print(f"⚠️  未找到权重文件: {weight_name}")
        return weight_name
        
    except Exception as e:
        print(f"⚠️  解析权重路径失败: {e}")
        return weight_name


def setup_models_directories():
    """
    设置模型目录结构，确保必要的文件夹存在
    """
    if not COMFYUI_AVAILABLE:
        print("⚠️  ComfyUI folder_paths不可用，跳过目录设置")
        return
    
    try:
        models_path = folder_paths.models_dir
        
        # 创建HYPIR专用目录
        hypir_dir = os.path.join(models_path, "HYPIR")
        if not os.path.exists(hypir_dir):
            os.makedirs(hypir_dir, exist_ok=True)
            print(f"✅ 创建HYPIR模型目录: {hypir_dir}")
            
            # 创建说明文件
            readme_path = os.path.join(hypir_dir, "README.txt")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("HYPIR模型权重目录\n")
                f.write("=" * 30 + "\n\n")
                f.write("请将HYPIR权重文件(.pth, .ckpt, .safetensors)放在此目录下\n\n")
                f.write("推荐的权重文件:\n")
                f.write("- HYPIR_sd2.pth (官方权重)\n")
                f.write("- 其他兼容的HYPIR权重文件\n\n")
                f.write("下载地址:\n")
                f.write("https://github.com/littlewhitesea/HYPIR\n")
        
        # 检查diffusers目录
        diffuser_paths = folder_paths.get_folder_paths("diffusers")
        if diffuser_paths:
            diffuser_dir = diffuser_paths[0]
            if not os.path.exists(diffuser_dir):
                os.makedirs(diffuser_dir, exist_ok=True)
                print(f"✅ 创建Diffusers模型目录: {diffuser_dir}")
                
                # 创建说明文件
                readme_path = os.path.join(diffuser_dir, "README.txt")
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write("Diffusers模型目录\n")
                    f.write("=" * 30 + "\n\n")
                    f.write("请将Stable Diffusion 2.0基础模型放在此目录下\n\n")
                    f.write("推荐模型:\n")
                    f.write("- stable-diffusion-2-base/\n")
                    f.write("- 其他SD2兼容模型\n\n")
                    f.write("从HuggingFace下载:\n")
                    f.write("huggingface-cli download stabilityai/stable-diffusion-2-base\n")
        
        print("📁 模型目录结构设置完成")
        
    except Exception as e:
        print(f"⚠️  设置模型目录失败: {e}")


def print_path_info():
    """
    打印路径信息，帮助用户理解新的目录结构
    """
    print("\n📂 HYPIR模型路径管理")
    print("=" * 40)
    
    if COMFYUI_AVAILABLE:
        models_path = folder_paths.models_dir
        print(f"ComfyUI模型目录: {models_path}")
        print(f"HYPIR权重目录: {os.path.join(models_path, 'HYPIR')}")
        
        diffuser_paths = folder_paths.get_folder_paths("diffusers")
        if diffuser_paths:
            print(f"Diffusers模型目录: {diffuser_paths[0]}")
    else:
        print("⚠️  ComfyUI环境未检测到，使用手动路径管理")
    
    print("\n📋 使用说明:")
    print("1. HYPIR权重文件放在: ComfyUI/models/HYPIR/")
    print("2. SD2基础模型放在: ComfyUI/models/diffusers/")
    print("3. 节点会自动检测并显示可用模型")
    print("4. 无需手动输入路径，使用下拉菜单选择")


class HYPIRImageRestoration:
    """
    HYPIR图像修复节点
    利用基于扩散模型得分先验的图像修复技术
    """
    
    def __init__(self):
        self.enhancer = None
        self.current_model_path = None
        self.current_base_model_path = None
        
        # 设置模型目录结构
        setup_models_directories()
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用的模型和权重列表
        diffuser_models = get_diffuser_models()
        hypir_weights = get_hypir_weights()
        
        return {
            "required": {
                "image": ("IMAGE",),  # 输入待修复的图像
                "base_model": (diffuser_models, {
                    "default": diffuser_models[0] if diffuser_models else "stable-diffusion-2-base",
                    "tooltip": "SD2基础模型选择，从ComfyUI/models/diffusers/目录自动检测"
                }),
                "hypir_weight": (hypir_weights, {
                    "default": hypir_weights[0] if hypir_weights else "HYPIR_sd2.pth",
                    "tooltip": "HYPIR权重文件，从ComfyUI/models/HYPIR/目录自动检测"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "high quality, detailed, sharp, photorealistic",
                    "placeholder": "描述期望的修复效果",
                    "tooltip": "修复提示词，描述期望的图像质量和特征"
                }),
                "upscale_factor": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,  # 支持1-8倍放大
                    "tooltip": "图像放大倍数 (1-8)，推荐2-4倍获得最佳平衡"
                }),
            },
            "optional": {
                "preset_config": (["自定义", "快速修复", "标准增强", "高质量修复", "人像优化", "风景增强", "最大效果"], {
                    "default": "标准增强",
                    "tooltip": "预设配置：快速选择最佳参数组合"
                }),
                "lora_rank": ("INT", {
                    "default": 256,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": "LoRA秩参数 (自动检测，通常为256)，8的倍数"
                }),
                "model_t": ("INT", {
                    "default": 200,  # 官方推荐值
                    "min": 1,
                    "max": 1000,
                    "tooltip": "模型时间步数 (1-1000)，越高效果越强，推荐150-300"
                }),
                "coeff_t": ("INT", {
                    "default": 200,  # 官方推荐值
                    "min": 1,
                    "max": 1000,
                    "tooltip": "系数时间步数 (1-1000)，与model_t通常保持一致"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore_image"
    CATEGORY = "Image Restoration/HYPIR"
    
    def auto_detect_lora_rank(self, weight_path: str) -> int:
        """
        自动检测权重文件中的LoRA rank
        """
        try:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
            
            # 查找第一个lora_A权重来确定rank
            for key, tensor in state_dict.items():
                if "lora_A" in key and "weight" in key:
                    # lora_A的形状是 (rank, input_dim)
                    rank = tensor.shape[0]
                    print(f"🔍 自动检测到LoRA rank: {rank}")
                    return rank
            
            # 如果没找到，使用默认值
            print("⚠️  无法自动检测LoRA rank，使用默认值256")
            return 256
            
        except Exception as e:
            print(f"⚠️  检测LoRA rank时出错: {e}，使用默认值256")
            return 256

    def load_model(self, base_model_path: str, weight_path: str, lora_rank: int = 256, 
                   model_t: int = 50, coeff_t: int = 50):
        """
        加载HYPIR模型
        """
        # 检查模型是否已加载且参数相同
        if (self.enhancer is not None and 
            self.current_model_path == weight_path and
            self.current_base_model_path == base_model_path):
            return
            
        print(f"加载HYPIR模型...")
        print(f"基础模型路径: {base_model_path}")
        print(f"权重文件路径: {weight_path}")
        
        # 检查权重文件是否存在
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"HYPIR权重文件不存在: {weight_path}")
        
        # 自动检测LoRA rank（如果用户没有明确指定）
        detected_rank = self.auto_detect_lora_rank(weight_path)
        if lora_rank != detected_rank:
            print(f"⚠️  用户指定的LoRA rank ({lora_rank}) 与检测到的 ({detected_rank}) 不匹配")
            print(f"🔧 使用检测到的LoRA rank: {detected_rank}")
            lora_rank = detected_rank
        
        # 获取设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LoRA模块配置（基于HYPIR官方predict.py）
        # 这是HYPIR项目中经过验证的正确配置
        lora_modules = [
            "to_k",
            "to_q", 
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2", 
            "conv_shortcut",
            "conv_out",
            "proj_in",
            "proj_out",
            "ff.net.2",
            "ff.net.0.proj",
        ]
        
        try:
            with SuppressLogging():
                # 初始化SD2增强器
                self.enhancer = SD2Enhancer(
                    base_model_path=base_model_path,
                    weight_path=weight_path,
                    lora_modules=lora_modules,
                    lora_rank=lora_rank,
                    model_t=model_t,
                    coeff_t=coeff_t,
                    device=device
                )
                
                # 初始化所有模型组件
                self.enhancer.init_models()
                
            self.current_model_path = weight_path
            self.current_base_model_path = base_model_path
            print("HYPIR模型加载完成!")
            
        except Exception as e:
            print(f"加载HYPIR模型时出错: {str(e)}")
            raise
    
    def comfyui_to_hypir_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        将ComfyUI图像张量转换为HYPIR所需格式
        ComfyUI: (B, H, W, C) [0, 1] float32
        HYPIR: (B, C, H, W) [0, 1] float32
        """
        # 转换维度顺序 (B, H, W, C) -> (B, C, H, W)
        image_tensor = image_tensor.permute(0, 3, 1, 2)
        
        # 确保数据类型和范围正确
        image_tensor = image_tensor.float()
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        return image_tensor
    
    def hypir_to_comfyui_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        将HYPIR输出张量转换为ComfyUI格式
        HYPIR: (B, C, H, W) [0, 1] float32
        ComfyUI: (B, H, W, C) [0, 1] float32
        """
        # 转换维度顺序 (B, C, H, W) -> (B, H, W, C)
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        
        # 确保数据类型和范围正确
        image_tensor = image_tensor.float()
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        return image_tensor
    
    def apply_preset_config(self, preset_config, model_t, coeff_t, prompt):
        """
        应用预设配置
        """
        PRESET_CONFIGS = {
            "快速修复": {
                "model_t": 100,
                "coeff_t": 100,
                "prompt": "high quality, clean, sharp details"
            },
            "标准增强": {
                "model_t": 200,
                "coeff_t": 200,
                "prompt": "high quality, detailed, sharp, photorealistic"
            },
            "高质量修复": {
                "model_t": 250,
                "coeff_t": 250,
                "prompt": "masterpiece, best quality, ultra high resolution, extremely detailed, sharp focus"
            },
            "人像优化": {
                "model_t": 250,
                "coeff_t": 250,
                "prompt": "high quality portrait, detailed skin texture, sharp facial features, professional photography"
            },
            "风景增强": {
                "model_t": 200,
                "coeff_t": 200,
                "prompt": "high quality landscape, sharp details, vibrant colors, natural scenery"
            },
            "最大效果": {
                "model_t": 300,
                "coeff_t": 300,
                "prompt": "masterpiece, best quality, ultra high resolution, extremely detailed, perfect quality"
            }
        }
        
        if preset_config in PRESET_CONFIGS:
            config = PRESET_CONFIGS[preset_config]
            return config["model_t"], config["coeff_t"], config["prompt"]
        else:
            # 自定义配置，使用用户输入的值
            return model_t, coeff_t, prompt
    
    def restore_image(self, image, base_model, hypir_weight, prompt, upscale_factor, 
                     preset_config="标准增强", lora_rank=256, model_t=200, coeff_t=200):
        """
        执行图像修复
        """
        try:
            # 解析路径
            base_model_path = resolve_model_path(base_model)
            weight_path = resolve_weight_path(hypir_weight)
            
            print(f"📁 解析模型路径:")
            print(f"   基础模型: {base_model} -> {base_model_path}")
            print(f"   HYPIR权重: {hypir_weight} -> {weight_path}")
            
            # 应用预设配置
            final_model_t, final_coeff_t, final_prompt = self.apply_preset_config(preset_config, model_t, coeff_t, prompt)
            
            print(f"🎛️  使用配置: {preset_config}")
            if preset_config != "自定义":
                print(f"   model_t: {final_model_t}, coeff_t: {final_coeff_t}")
                print(f"   优化提示词: {final_prompt}")
            
            # 加载模型
            self.load_model(base_model_path, weight_path, lora_rank, final_model_t, final_coeff_t)
            
            # 转换输入图像格式
            input_tensor = self.comfyui_to_hypir_tensor(image)
            
            print(f"开始图像修复...")
            print(f"输入图像尺寸: {input_tensor.shape}")
            print(f"上采样倍数: {upscale_factor}")
            print(f"修复提示: {final_prompt}")
            
            # 执行图像增强
            with torch.no_grad():
                restored_tensor = self.enhancer.enhance(
                    lq=input_tensor,
                    prompt=final_prompt,
                    upscale=upscale_factor,
                    return_type="pt"  # 返回PyTorch张量
                )
            
            # 转换输出格式
            output_tensor = self.hypir_to_comfyui_tensor(restored_tensor)
            
            print(f"图像修复完成!")
            print(f"输出图像尺寸: {output_tensor.shape}")
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"图像修复过程中出错: {str(e)}")
            # 出错时返回原始图像
            return (image,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        检查输入是否发生变化
        """
        return float("NaN")  # 总是重新计算


# 为了兼容性，添加别名
HYPIRNode = HYPIRImageRestoration