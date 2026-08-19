# 基于 YOLO 的水果新鲜度图片判别系统

## 项目功能

本项目使用 Ultralytics YOLO 分类模型，对一张本地水果图片进行二分类：

- `fresh`：新鲜；
- `spoiled`：变质或腐烂。

支持整理图片数据集、训练 YOLO 模型、命令行单图预测和 Streamlit 图片上传页面。第一版不接入摄像头，也不识别一张图中的多个水果。

## 环境安装

建议使用 Python 3.10 或更高版本，在项目根目录运行：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果电脑有 NVIDIA GPU，PyTorch 会自动尝试使用可用 GPU；没有 GPU 也可以运行，但训练速度会较慢。

如果 Windows 在安装 PyTorch 时提示 `WinError 206`（文件名或扩展名过长），请将虚拟环境建在短路径，再用该环境运行本项目：

```bash
python -m venv C:\tmp\fruit-yolo-venv
C:\tmp\fruit-yolo-venv\Scripts\activate
pip install -r requirements.txt
```

## 数据集准备

可使用公开的 [Spoiled and fresh fruit inspection dataset](https://data.mendeley.com/datasets/6ps7gtp2wg/1)。该数据集包含不同水果在新鲜与不新鲜/腐烂状态下的图片。下载与使用时请保留原始许可、DOI 和引用信息。

将图片按标签放入以下目录：

```text
data/
  raw/
    fresh/
      image_001.jpg
    spoiled/
      image_002.jpg
```

随后运行数据整理脚本：

```bash
python scripts/prepare_dataset.py --raw-dir data/raw --output-dir data/processed
```

脚本会按照 70% / 15% / 15% 的比例将每类图片复制到：

```text
data/processed/
  train/{fresh,spoiled}/
  val/{fresh,spoiled}/
  test/{fresh,spoiled}/
```

可用 `--seed` 固定随机切分，用 `--split 70,15,15` 自定义比例。不要把同一个实物在不同角度下拍摄的图片分散到训练集和测试集，否则测试结果会偏高。

## 训练模型

```bash
python train.py --data data/processed
```

默认使用轻量 YOLO 分类预训练权重 `yolo26n-cls.pt`，训练 50 轮，输入尺寸为 224。训练完成后，最佳权重会复制到：

```text
models/best.pt
```

可以覆盖默认参数：

```bash
python train.py --data data/processed --epochs 80 --batch 8 --output-model models/tomato-freshness.pt
```

## 图片预测

使用训练完成的权重预测一张图片：

```bash
python predict.py --image examples/sample.jpg --model models/best.pt
```

`examples/sample.jpg` 只是示例路径，运行时请替换为实际图片。命令会输出中文结论、原始类别标签和置信度。

## 启动网页

```bash
streamlit run app.py
```

在浏览器中打开终端提示的本地地址，上传 JPG 或 PNG 图片。在侧边栏填写训练完成的 `.pt` 模型路径，即可看到类别和置信度。

## 评价指标

训练完成后应记录：

- Accuracy：整体正确率；
- 每一类的 Precision、Recall、F1；
- 混淆矩阵；
- 单张图片的推理时间。

报告中应重点分析哪些图片被误判，以及光照、背景、遮挡和图片来源对结果的影响。

## 项目限制

- 模型只根据图像可见的外观状态判断，不检测糖度、硬度或内部腐烂；
- 模型输出不构成食品安全结论；
- 公开图片与真实拍摄环境可能不同，后续应补充真实场景图片验证泛化能力；
- 当前版本是整图分类，不适用于一张图片中有多个水果的定位需求。
