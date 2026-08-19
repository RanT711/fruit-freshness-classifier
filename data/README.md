# 数据集目录

请将公开图片数据按以下结构放入本项目：

```text
data/
  raw/
    fresh/      # 新鲜水果图片
    spoiled/    # 变质或腐烂水果图片
```

随后运行：

```bash
python scripts/prepare_dataset.py --raw-dir data/raw --output-dir data/processed
```

脚本将生成 YOLO 分类训练所需的 `train`、`val`、`test` 三个子集。原始与处理后的图片不会提交到 Git。

推荐公开来源：[Spoiled and fresh fruit inspection dataset](https://data.mendeley.com/datasets/6ps7gtp2wg/1)。下载后请按图片的实际状态整理到 `fresh` 与 `spoiled` 两个目录，并保留数据集原始许可和引用信息。
