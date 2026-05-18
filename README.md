# YOLO-LAB

YOLO 模型训练与推理工具集，包含命令行和图形界面两种工作流。

## 项目结构

| 目录 | 说明 | 链接 |
|------|------|------|
| `cli/` | YOLO 命令行训练与推理工具 | [cli/README.md](cli/README.md) |
| `gui/` | YOLO 图形化界面 (PyQt6) | [gui/README.md](gui/README.md) |

## 快速开始

### GUI（推荐）
```bash
cd gui
bash setup.sh
python gui/main.py
```

### CLI
```bash
cd cli
pip install -r requirements.txt
python scripts/train_segment.py
```
