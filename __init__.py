"""
ComfyUI HYPIR 自定义节点
基于HYPIR项目实现高质量图像修复功能
"""

try:
    from .hypir_node import HYPIRImageRestoration
except ImportError:
    from hypir_node import HYPIRImageRestoration

# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "HYPIRImageRestoration": HYPIRImageRestoration
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYPIRImageRestoration": "HYPIR Image Restoration"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]