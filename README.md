# YOLO 图像分割训练工作室

一个基于 **Ultralytics YOLO** 的图像分割训练小项目，用于完成数据集整理、模型训练、断点续训、历史模型继续训练，以及训练日志与验证结果记录。项目已经把**配置、训练流程、日志记录**拆开，适合后续持续做实验和对比。  

---

## 项目特点

本项目不是只为了“跑一次训练”，而是尽量把训练流程做得更清楚、更规范。当前已经支持三种训练模式：**新训练、继续上次中断训练、基于历史 best.pt 再训练**；同时支持训练前确认、训练时可选数据增强，以及训练结束后自动验证并记录日志。 

---

## 项目结构

从当前目录来看，项目结构大致如下：

```text
CODE/
├── dataset_tools/
│   ├── split_images_only/
│   │   ├── split_every_5th_images_only.py
│   │   └── split_random_images_only.py
│   ├── split_train_val/
│   │   ├── split_every_5th_with_labels.py
│   │   └── split_random_with_labels.py
│   ├── split_train_val_test/
│   │   └── split_random_with_labels.py
│   └── create_empty_labels.py
│
├── result/
│
├── scripts/
│   ├── config.py
│   ├── paths.py
│   ├── train_logger.py
│   └── train_segment.py
│
├── train_logs/
│   ├── result_per_class_log.csv
│   ├── result_summary_log.csv
│   └── train_log.csv
│
├── .gitignore
├── data.yaml
```

其中，`scripts/` 是训练核心目录，`config.py` 管理训练配置，`train_segment.py` 负责训练流程，`train_logger.py` 负责日志记录；`train_logs/` 中有训练日志、整体结果日志和分类别结果日志。 

---

## 核心功能

### 1. 三种训练模式

训练主程序支持以下三种模式：

* **模式1：开启一个新的训练**
* **模式2：继续上次中断的训练**
* **模式3：基于历史实验的 `best.pt` 再次训练**

程序启动后会提示选择模式。

### 2. 训练前确认机制

正式训练前，程序会打印当前关键参数，包括：

* 当前模式
* 使用的权重文件
* 数据配置文件
* 实验名称
* 训练轮数

这样可以避免路径写错、模型选错、实验名覆盖等问题。

### 3. 支持数据增强开关

训练前可选择是否启用数据增强。如果启用，会把配置中的增强参数一起传给 `model.train()`，例如：

* `hsv_h`
* `hsv_s`
* `hsv_v`
* `translate`
* `scale`
* `fliplr`
* `mosaic`
* `mixup`
* `copy_paste`

这些参数统一写在 `config.py` 中。 

### 4. 自动验证与日志记录

训练完成后，程序会自动执行验证，并把结果写入 CSV 日志，包括：

* 训练流程日志
* 整体验证指标日志
* 分类别验证指标日志

便于后续做实验对比和分析。  

---

## 环境依赖

建议环境：

* Python 3.8+
* Ultralytics
* PyYAML

安装示例：

```bash
pip install ultralytics pyyaml
```

如果你使用 Conda，也可以先创建独立环境再安装。

---

## 配置说明

训练配置统一在 `scripts/config.py` 中管理。当前 `TrainConfig` 主要包含以下几类内容：路径配置、训练超参数、实验名称、数据增强参数，以及根据实验名自动生成的保存路径。

### 路径配置

```python
data_yaml   # data.yaml 路径
model_file  # 初始模型权重路径
results_dir # 所有实验结果保存根目录
log_dir     # 日志目录
```

### 训练参数

```python
epochs
imgsz
batch
device
experiment_name
```

### 数据增强参数

```python
use_augment
hsv_h
hsv_s
hsv_v
degrees
translate
scale
shear
perspective
flipud
fliplr
mosaic
mixup
copy_paste
```

### 自动生成的路径属性

```python
save_dir  # 当前实验结果目录
last_pt   # 当前实验中断权重
best_pt   # 当前实验最佳权重
```

这些属性是通过 `@property` 自动计算得到的，不需要手动拼接路径。

---

## 使用方法

### 1. 修改配置

先在 `scripts/config.py` 中设置好：

* 数据集配置文件路径
* 初始模型路径
* 实验结果目录
* 日志目录
* 实验名称
* 训练参数

### 2. 运行训练脚本

```bash
python scripts/train_segment.py
```

程序启动后会提示你选择训练模式。

### 3. 选择训练模式

运行后会看到：

```text
模式1 - 开启一个新的训练
模式2 - 继续上次中断的训练
模式3 - 基于历史实验的 best.pt 再次训练
```

根据提示输入 `1`、`2` 或 `3` 即可。

---

## 训练模式说明

### 模式1：开启新训练

适合第一次训练，或者希望从某个初始模型重新开始训练。程序会读取 `config.model_file` 作为起始权重，然后根据配置开始训练。训练前会询问是否启用数据增强。

### 模式2：继续上次中断训练

如果当前实验目录下存在 `last.pt`，程序会从中断处继续训练；如果不存在，则会询问是否改为开启新训练。

### 模式3：基于历史实验继续训练

程序会扫描 `results_dir` 下已有实验文件夹，让你选择一个历史实验，然后加载该实验下的 `best.pt` 作为基础模型继续训练。这个模式适合做微调和增量实验。

---

## 日志文件说明

项目当前会生成三类日志。

### `train_log.csv`

记录训练流程，包括：

* 时间
* 模式
* 状态（started / finished / failed）
* 实验名称
* 模型路径
* 数据集路径
* epochs / imgsz / batch / device
* 保存目录
* `best.pt` / `last.pt`
* 备注信息



### `result_summary_log.csv`

记录整体验证结果，包括：

* 图片数
* 实例数
* Box Precision
* Box Recall
* Box mAP50
* Box mAP50-95
* Mask Precision
* Mask Recall
* Mask mAP50
* Mask mAP50-95



### `result_per_class_log.csv`

记录每个类别的验证结果，包括：

* 类别编号
* 类别名称
* 图像数
* 实例数
* Box 指标
* Mask 指标

这对分析哪个类别效果差、是否需要补数据很有帮助。

---

## 验证结果提取逻辑

验证阶段会调用 `model.val()` 获取 Ultralytics 返回的指标；然后根据 `data.yaml` 自动定位验证集标签目录，统计每个类别在验证集中的图像数和实例数；最后将这些信息与模型评估指标一起写入日志。 

这样做的好处是：日志里不仅有模型指标，还有类别分布信息，后续分析更完整。

---

## 推荐使用流程

建议平时按照下面的思路使用这个项目：

先用 `dataset_tools/` 处理数据集，再检查 `data.yaml` 是否正确；然后在 `config.py` 中设置新的 `experiment_name` 和训练参数；接着运行 `train_segment.py` 开始训练；训练结束后查看 `result/` 下的权重文件，以及 `train_logs/` 下的 CSV 日志做实验对比。

这样做的好处是，每次实验都有独立目录和独立日志，后面无论是分析过拟合、比较增强策略，还是写博客总结，都会清楚很多。`TrainConfig` 中的 `save_dir`、`last_pt`、`best_pt` 也正是为这种实验管理方式准备的。

---

## 后续可改进方向

这个项目已经有不错的基础，后面还可以继续扩展，比如：

* 增加统一的推理脚本说明
* 增加测试集评估说明
* 增加导出 ONNX / TensorRT 的说明
* 增加实验命名规范
* 增加结果可视化分析脚本
* 增加训练参数自动保存为 txt / json

---

## 总结

这是一个已经初步模块化的 YOLO 分割训练项目。它把**训练配置、训练模式、数据增强、验证提取、日志记录**拆开了，适合做持续实验，而不只是临时跑一遍模型。对于后面做模型对比、分析各类别效果、总结训练经验，都会很方便。  
