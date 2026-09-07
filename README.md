# 基于 YOLO 的水果新鲜度图片判别系统

## Windows 一键开始

拿到仓库后，先在项目根目录执行：

```text
双击 setup_windows.bat
双击 run_web.bat
```

启动成功后，在浏览器打开：

`http://127.0.0.1:8501`

首次运行会自动创建虚拟环境、安装依赖、从 GitHub Release 下载 `models/best.pt`，并按 `model-manifest.json` 中记录的 SHA-256 做完整性校验：

The repository now includes the trained `models/best.pt` file directly. A fresh clone uses this bundled weight and does not require a separate model download.

- Included model: `models/best.pt`
- SHA-256：`f4c996c44b95f27717874d81c4bd7c7a04dbf09ac5d7cb8bacdd1fdc30eecfe6`

如果电脑没有可用 NVIDIA GPU，系统会自动回退到 CPU。网页预测和命令行预测都能继续使用，只是训练会更慢。

## 这是什么

这是一个本地运行的水果外观二分类系统，使用 Ultralytics YOLO 分类模型把整张图片判别为：

- `fresh`：新鲜
- `spoiled`：变质或腐烂

当前版本支持数据整理、模型训练、命令行单图预测和 Streamlit 网页上传测试。

## 当前版本边界

- 只支持上传外部静态图片进行识别，不接摄像头、不处理视频流、也不做实时识别；
- 只判断图片中可见的外观状态，不检测糖度、硬度、内部腐烂或农残；
- 结果仅用于课程演示和算法验证，不构成食品安全结论；
- 当前为整图分类，不适合一张图中多个水果的定位任务。

## 模型与开源许可

- 项目代码与发布流程按 AGPL-3.0 开源；
- 仓库根目录提供完整 [LICENSE](LICENSE)；
- 发布模型通过 Release 分发，本地下载后会自动做 SHA-256 校验；
- 训练数据不随仓库分发，使用者需要自行获取并遵守原始数据许可。

## 数据来源与引用

本项目训练所依据的公开数据源为 Mendeley Data 上的数据集《Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality》，作者为 Nusrat Sultana、Musfika Jahan、Mohammad Shorif Uddin。

- 数据集记录页：<https://data.mendeley.com/datasets/bdd69gyhv8/1>
- DOI：`10.17632/bdd69gyhv8.1`
- 许可：<https://creativecommons.org/licenses/by/4.0/>

本项目未重新分发原始图片，而是基于该公开数据集筛选图片，并训练了一个 `fresh` / `spoiled` 的二分类模型用于课程演示与算法验证。如需复现实验或再分发衍生成果，请保留原始作者署名、DOI、CC BY 4.0 许可链接以及上述修改说明。

## 手动使用方式

如果你不想双击批处理，也可以手动执行：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-windows.txt
streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

单图命令行预测：

```powershell
python predict.py --image <图片路径> --model models/best.pt
```

训练入口：

```powershell
python scripts/prepare_dataset.py --raw-dir data/raw --output-dir data/processed
python train.py --data data/processed
```

## 常见问题

### 1. 双击 `setup_windows.bat` 没反应

通常是没有安装 Python，或者系统没有把 `py` 启动器装进去。请安装 Python 3.10 及以上版本，并勾选 Windows 启动器后重试。

### 2. 模型下载失败

请检查网络是否能访问 GitHub Release，并确认 `model-manifest.json` 里的下载地址没有被改动。脚本会自动校验 SHA-256；如果校验失败，会拒绝使用该模型并保留安全的旧文件状态。

### 3. `8501` 端口被占用

先关闭已打开的旧网页实例或其他正在占用 8501 端口的程序，然后重新双击 `run_web.bat`。

### 4. 预测时提示图片不支持

网页仅接受常见静态图片格式，上传内容会先做文件大小、像素数量和图片完整性校验。请优先使用正常导出的 JPG、JPEG 或 PNG 图片。

## 复现说明

项目默认处理目录如下：

```text
data/
  raw/{fresh,spoiled}/
  processed/
    train/{fresh,spoiled}/
    val/{fresh,spoiled}/
    test/{fresh,spoiled}/
models/
  best.pt
```

最终第四轮训练使用 432 张去重后的原始图片，切分为训练集 301、验证集 64、测试集 67；测试集 Top-1 准确率为 100.0%，但该结果只代表当前公开图片切分上的表现，不等同于真实生产场景精度。
