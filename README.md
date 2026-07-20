# VoxCPM TTS Server

## 依赖安装

```bash
uv sync
```

## 配置

首次运行会自动生成 `config.toml`，**必须编辑以下字段**：

```toml
[cli]
# 修改为你的 voxcpm2-cli 实际路径
path = "/path/to/voxcpm2-cli"
# 修改为你的模型文件实际路径
base_model = "/path/to/VoxCPM-0.5B-BaseLM-Q8_0.gguf"
acoustic_model = "/path/to/VoxCPM-0.5B-Acoustic-F16.gguf"
```

## 启动

```bash
uv run python main.py
```
