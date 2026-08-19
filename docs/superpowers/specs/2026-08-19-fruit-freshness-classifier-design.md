# 水果新鲜度离线判别系统设计

## 目标

构建一个可在本地运行的图片判别系统：用户上传或指定一张水果图片，系统使用 Ultralytics YOLO 分类模型输出 `fresh`（新鲜）或 `spoiled`（变质）及置信度。第一版不接入摄像头、不依赖实物采集，也不处理一张图中的多个水果。

## 范围

### 包含

- 将已下载的公开图片按可复现规则划分为训练、验证和测试集；
- 用 YOLO 分类预训练权重训练二分类模型；
- 验证训练结果并保存测试集指标；
- 在命令行对单张图片预测；
- 用 Streamlit 提供本地图片上传页面；
- 针对数据集结构、数据划分和预测结果格式提供自动化测试；
- 提供从安装、准备数据、训练到启动界面的中文说明。

### 不包含

- 摄像头、视频流或输送带接入；
- 多目标定位、边界框标注、YOLO 检测或实例分割；
- 糖度、硬度、内部腐烂等 RGB 图片无法可靠获取的内部品质指标；
- 自动下载第三方数据集。数据集下载受网站访问与许可影响，项目提供清晰的数据放置规则和来源链接。

## 用户流程

```text
公开图片数据 → 数据整理脚本 → train/val/test 分类目录
                                  ↓
                           YOLO 分类训练
                                  ↓
                         best.pt 模型权重
                           ↙              ↘
                    命令行单图预测       Streamlit 图片上传
```

1. 用户从 README 所列公开来源下载图片，并将图片放在 `data/raw/fresh/` 与 `data/raw/spoiled/`。
2. 用户运行数据准备脚本，脚本检查图片、去除不可读文件并按固定随机种子将每类图片复制到 `data/processed/{train,val,test}/{fresh,spoiled}/`。
3. 用户运行训练脚本，加载轻量 YOLO 分类预训练权重，输出训练日志、验证图表和 `best.pt`。
4. 用户可通过命令行或网页上传图片，得到类别、置信度和中文结论。

## 技术决策

### 为什么选择 YOLO 分类

每张输入图片只包含一个主要农产品，任务只需要整图类别和置信度，不需要目标位置。YOLO 分类模型正好适合这个输入输出形式，并支持 ImageNet 预训练权重的迁移学习。使用当前 Ultralytics 发行版提供的轻量 `yolo26n-cls.pt` 作为默认权重；若后续依赖版本改名，只在配置中更换模型文件名，不改变业务代码。

### 数据与标签

默认标签固定为：

- `fresh`：外观新鲜、无明显腐烂；
- `spoiled`：不新鲜、变质或腐烂。

公开数据集只作为训练来源。系统输出代表图像中的外观状态，不等同于食品安全判定或完整质量等级。

### 目录约定

```text
data/
  raw/
    fresh/
    spoiled/
  processed/
    train/{fresh,spoiled}/
    val/{fresh,spoiled}/
    test/{fresh,spoiled}/
models/
  best.pt
runs/
  ...                 # Ultralytics 训练输出，不提交到 Git
```

原始数据、处理后的数据、模型权重和训练输出均不提交到 Git。仓库会通过 `.gitignore` 排除它们，并保留目录说明文件。

## 组件设计

| 文件 | 职责 |
|---|---|
| `src/fruit_grader/dataset.py` | 发现有效图片、检查类别目录、以可复现且按类分层的方式划分数据集。 |
| `src/fruit_grader/inference.py` | 读取 YOLO 分类结果，转换为稳定的 `Prediction` 数据结构和中文结论。 |
| `scripts/prepare_dataset.py` | 调用数据模块，将 `data/raw` 转换成 Ultralytics 分类目录。 |
| `train.py` | 验证处理后的数据集并启动 YOLO 训练与测试集验证。 |
| `predict.py` | 对指定图片进行命令行预测，并显示类别、置信度和结论。 |
| `app.py` | 提供图片上传、预览和预测结果页面。 |
| `tests/` | 验证数据准备和预测结果转换的关键行为。 |

### 关键接口

```python
@dataclass(frozen=True)
class DatasetSplit:
    train: int
    val: int
    test: int

@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    message: str

def prepare_classification_dataset(
    raw_root: Path,
    output_root: Path,
    split: DatasetSplit = DatasetSplit(70, 15, 15),
    seed: int = 42,
) -> dict[str, dict[str, int]]: ...

def prediction_from_result(result: Any) -> Prediction: ...
```

`prepare_classification_dataset` 只接受包含 `fresh` 和 `spoiled` 子目录的原始目录；每次运行先清理本次生成的输出目录，之后从已验证的源文件复制图片。它会在类别缺失、类别没有有效图片或切分比例不等于 100 时抛出带原因的异常。

`prediction_from_result` 不加载模型，只将 Ultralytics 的单张分类结果转换为项目自己的数据结构。这样数据转换可在不训练模型的情况下测试。

## 错误处理

- 训练、预测或网页应用发现模型文件不存在时，提示先完成训练或传入模型路径；
- 数据准备发现缺少 `fresh`/`spoiled` 目录、空类别或非图片文件时，给出中文错误；
- 上传非图片文件或无法解析的图片时，网页显示错误而不调用模型；
- 预测结果包含未知标签时，仍保留原始标签并给出“未知类别”的结论，避免错误地报告新鲜或变质。

## 验收标准

1. `pytest` 能验证数据准备的目录结构、可重复切分、错误输入和预测结果转换；
2. `python scripts/prepare_dataset.py --raw-dir data/raw --output-dir data/processed` 能生成 YOLO 分类目录；
3. `python train.py --data data/processed` 能启动 YOLO 分类训练；
4. `python predict.py --image <图片路径> --model <模型路径>` 能输出类别与置信度；
5. `streamlit run app.py` 能上传 JPG/PNG 图片并显示结果；
6. 训练后的测试结果明确报告准确率与每类指标，报告中不将其表述为内部品质或食品安全结论。

## 验证策略

- 使用 pytest 进行单元测试；
- 测试中临时创建有效的小图片文件，确保不依赖真实数据集；
- 不对真实 YOLO 权重进行单元测试，以避免网络、GPU 和模型下载依赖；
- 在依赖安装后运行 `pytest`、`python scripts/prepare_dataset.py --help`、`python train.py --help`、`python predict.py --help` 和 `streamlit run app.py` 的启动检查；
- 当公开数据集准备完成后，执行一次真实训练并记录测试集输出。
