# ComfyUI-HYPIR

**English** | [中文](README.md)

---

### 📖 Introduction

**ComfyUI-HYPIR** is a ComfyUI custom node based on the [HYPIR project](https://github.com/XPixelGroup/HYPIR), implementing high-quality image restoration functionality. HYPIR (Harnessing Diffusion-Yielded Score Priors for Image Restoration) is an advanced technique that leverages diffusion model score priors for image restoration.

This node seamlessly integrates HYPIR's powerful capabilities into ComfyUI workflows, supporting:
- 🔧 **Intelligent Image Restoration** - deblurring, denoising, super-resolution
- 🎛️ **Preset Configuration System** - quick fix, high-quality restoration, portrait optimization, etc.
- 📁 **Smart Path Management** - auto-detect ComfyUI model directories, dropdown selection
- ⚡ **1-8x Upscaling** - flexible resolution enhancement
- 🎯 **LoRA Adaptation** - automatic LoRA parameter detection and configuration

### ✨ Key Features

#### 🚀 **User-Friendly**
- **Zero-Config Startup** - automatic directory structure and config file creation
- **Dropdown Selection** - no need to manually input complex paths
- **Preset Configurations** - optimized parameters for different scenarios
- **Real-time Feedback** - detailed processing status and parameter display

#### 🔧 **Technically Advanced**
- **Native ComfyUI Integration** - perfect integration into ComfyUI workflows
- **folder_paths Management** - follows ComfyUI standard path management
- **Smart Model Detection** - automatically discover and validate model files
- **Memory Optimization** - tiled processing for large images, supports high resolution

#### 🎨 **Processing Effects**
- **Multi-scenario Adaptation** - vintage photo restoration, blur sharpening, noise removal
- **Portrait Optimization** - specialized portrait restoration presets
- **Landscape Enhancement** - color and detail enhancement for landscape photos
- **Flexible Upscaling** - 1-8x resolution enhancement, balance quality and speed

### 📦 Installation

#### Method 1: Git Clone (Recommended)
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-repo/ComfyUI-HYPIR.git
cd ComfyUI-HYPIR
pip install -r requirements.txt
```

#### Method 2: Manual Download
1. Download project files to `ComfyUI/custom_nodes/ComfyUI-HYPIR/`
2. Install dependencies: `pip install -r requirements.txt`

### 📥 Model Downloads

#### Required Model Files

1. **HYPIR Weight File**
   ```bash
   # Download to ComfyUI/models/HYPIR/
   wget https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth
   ```
   Or manual download: [HuggingFace Link](https://huggingface.co/lxq007/HYPIR/tree/main)

2. **Stable Diffusion 2.0 Base Model**
   ```bash
   # Download to ComfyUI/models/diffusers/
   huggingface-cli download stabilityai/stable-diffusion-2-base --local-dir ComfyUI/models/diffusers/stable-diffusion-2-base
   ```

#### 🗂️ Standard Directory Structure
```
ComfyUI/
├── models/
│   ├── diffusers/
│   │   └── stable-diffusion-2-base/     # SD2 base model
│   └── HYPIR/
│       └── HYPIR_sd2.pth               # HYPIR weight file
└── custom_nodes/
    └── ComfyUI-HYPIR/                  # This project
```

### 🎯 Usage

#### 1. **Basic Usage**
1. Restart ComfyUI
2. Find `Image Restoration/HYPIR` → `HYPIR Image Restoration` in node menu
3. Connect image input
4. Select models (auto-detected dropdown menus)
5. Choose preset configuration or custom parameters
6. Run workflow

#### 2. **Preset Configuration Guide**
| Preset Name | Use Case | Parameter Features | Recommended Usage |
|-------------|----------|-------------------|-------------------|
| **Quick Fix** | Light issues | Low parameters, fast processing | Daily photo minor fixes |
| **Standard Enhancement** | General scenarios | Balance effect and speed | Most restoration tasks |
| **High Quality Restoration** | Severe issues | High parameters, quality priority | Important photos, severe damage |
| **Portrait Optimization** | Portrait photos | Face-optimized | Portraits, headshots |
| **Landscape Enhancement** | Landscape photos | Color and detail enhancement | Natural scenery, architecture |
| **Maximum Effect** | Extreme restoration | Highest parameter settings | Challenging restoration tasks |

#### 3. **Parameter Tuning Guide**
- **upscale_factor**: 1-8x, recommend 2-4x for quality-performance balance
- **model_t/coeff_t**: 150-300, higher values = stronger effect but slower speed
- **lora_rank**: Usually auto-detected as 256, no modification needed
- **prompt**: Describe desired effects, e.g., "high quality, sharp, detailed"

### 🚀 Advanced Features

#### 🎨 **Targeted Restoration Strategies**
```
Vintage Photo Restoration:
- Preset: Portrait Optimization
- Prompt: vintage photo restoration, enhanced colors
- Upscale: 2-3x

Severe Blur:
- Preset: Maximum Effect
- Prompt: extremely sharp, deblurred, enhanced clarity
- Upscale: 2-4x

Low Resolution Images:
- Preset: High Quality Restoration
- Prompt: high resolution upscale, detailed
- Upscale: 4-8x
```

#### ⚡ **Performance Optimization**
- **GPU Recommended**: NVIDIA RTX 4090/4080 (16GB+ VRAM)
- **Minimum Requirement**: GTX 1080Ti/RTX 3060 (8GB VRAM)
- **Large Image Processing**: Automatic tiling, supports 4K+ resolution
- **Memory Management**: Smart caching, avoid redundant loading

### 🔧 Troubleshooting

#### Common Issues

**Q: Model loading failed**
```
A: Check file paths:
   - HYPIR weights: ComfyUI/models/HYPIR/HYPIR_sd2.pth
   - SD2 model: ComfyUI/models/diffusers/stable-diffusion-2-base/
```

**Q: Processing effect not obvious**
```
A: Adjust strategy:
   - Choose appropriate preset configuration
   - Use targeted prompts
   - Increase model_t/coeff_t values
   - Increase upscale factor
```

**Q: Out of VRAM**
```
A: Optimize settings:
   - Reduce upscale factor
   - Use "Quick Fix" preset
   - Reduce input image size
   - Close other VRAM-consuming programs
```

### 📊 Effect Showcase

This node excels in the following scenarios:
- ✅ **Vintage Photo Restoration** - fading, scratches, noise removal
- ✅ **Blurry Images** - motion blur, defocus recovery
- ✅ **Low Resolution** - high-quality upscaling of small images
- ✅ **Noise Removal** - high ISO noise cleaning
- ✅ **Detail Enhancement** - texture and edge sharpening

### 🤝 Acknowledgments

This project is based on the following excellent open-source projects:
- **[HYPIR](https://github.com/XPixelGroup/HYPIR)** - Core image restoration algorithm
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** - Powerful AI workflow platform
- **[Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-base)** - Base diffusion model

Special thanks to the HYPIR team for their outstanding work:
> Xinqi Lin, Fanghua Yu, Jinfan Hu, Zhiyuan You, Wu Shi, Jimmy S. Ren, Jinjin Gu, Chao Dong

