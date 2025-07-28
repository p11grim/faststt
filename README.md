# faststt

基于 `faster-whisper` 的语音识别（ASR）系统，支持连续语音录音、语音活动检测（VAD）、多线程转录与详细转录耗时统计。

## 项目简介

本项目旨在提供一个高效实用的语音转文本（Speech-to-Text, STT）工具，结合多种语音活动检测技术，专为中文语音或中英混杂转写场景优化。  
支持 GPU 加速和多种配置参数，适合个人开发、科研实验等场景。

## 主要特性

- **语音录制和转写**：无需手动分段，自动检测语音区间并转录。
- **双重 VAD（WebRTC + Silero）**：提升语音检测准确率，减少误识别。
- **多线程异步转写**：充分利用多核/多卡系统资源。
- **详细转录耗时统计**：每次转录完成后，实时输出时长统计。
- **彩色命令行 UI**：美观直观，支持多平台。
- **高可配置**：模型、推理设备、VAD 敏感度、录音参数均可灵活设置。

## 目录结构

```
faststt/
├── faststt/
│   ├── __init__.py
│   └── audio_recorder.py        # 主要实现
├── start.py                     # 示例主程序
├── README.md
├── LICENSE
└── ...
```
## python及依赖项版本

Python 3.10.18
创建新的虚拟环境后，使用pip install -r requirements.txt安装

### requirements.txt
PyAudio==0.2.14
faster-whisper==1.1.1
pvporcupine==1.9.5
webrtcvad-wheels==2.0.14
halo==0.0.31
scipy==1.15.2
websockets==15.0.1
websocket-client==1.8.0
openwakeword>=0.4.0
numpy<2.0.0
transformers==4.46.1
tokenizers==0.20.3

### 安装 PyTorch 及相关依赖  

本项目依赖 PyTorch 2.4.0 + CUDA 11.8（或兼容版本）。 
请根据你的显卡和 CUDA 版本，安装合适的 PyTorch 包。  
推荐方法：  
1. 访问PyTorch 官网安装页面(https://pytorch.org/get-started/locally/) 生成对应命令。 
2. CUDA 11.8 用户可直接运行：pip install torch==2.4.0+cu118 torchaudio==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118 
3. 然后再安装其他依赖：pip install -r requirements.txt 
如无 NVIDIA GPU 或不需 CUDA，可将 `+cu118` 替换为 `+cpu`，或直接安装 CPU 版本：pip install torch==2.4.0 torchaudio==2.4.0
requirements.txt 说明
 `requirements.txt` 不包含 `torch`、`torchaudio` 等 PyTorch 相关包，请根据上方说明手动安装，以确保环境兼容。


## 快速开始

1. 运行 pip install git+https://github.com/p11grim/faststt.git
2. 单独下载start.py文件，或者使用git clone https://github.com/p11grim/faststt.git，拷贝整个项目到本地。
3. 运行start.py

也可以创建自己的脚本，以 `start.py` 为例：


```python
from faststt import AudioToTextRecorder

config = {
    'model': 'large-v3',
    'language': 'zh',
    'download_root': 'D:\\model',
    'device': 'cuda',
    'compute_type': 'int8',
    # ... 可选参数参见下方
}

def on_text(text):
    print("转录结果：", text)

recorder = AudioToTextRecorder(**config)
recorder.text(on_text)
```

运行后即可通过麦克风实时说话，终端会动态显示转录结果和耗时统计。

## 主要参数说明

| 参数名                      | 含义                                      | 示例/默认值      |
|:----------------------------|:------------------------------------------|:-----------------|
| model                       | Whisper 模型名或路径                      | `"large-v3"`     |
| language                    | 识别语言                                  | `"zh"`           |
| download_root               | 模型缓存目录                              | `D:\\model`      |
| device                      | 运算设备 `"cpu"`/`"cuda"`                 | `"cuda"`         |
| compute_type                | 运算类型                                  | `"int8"`         |
| batch_size                  | 推理批量大小                              | `16`             |
| beam_size                   | 解码束宽                                  | `1`              |
| silero_sensitivity          | Silero VAD 敏感度（0-1）                  | `0.6`            |
| webrtc_sensitivity          | WebRTC VAD 模式（1-3）                    | `2`              |
| post_speech_silence_duration| 语音段后判定静音的时长（秒）              | `1.0`            |
| min_length_of_recording     | 最短录音时长（秒）                        | `0.5`            |
| pre_recording_buffer_duration| 前置录音缓冲（秒）                        | `1.0`            |

更多参数请参考源码注释。

## 用法示例

1. 运行 `start.py` 直接开始语音转录
2. 自定义回调函数处理结果
3. 查看终端输出的转录内容和统计信息

## 退出方法

- 按 `Ctrl+C` 终止程序，程序会自动关闭所有资源，并输出本次会话的转录耗时统计。
