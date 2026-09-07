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

脚本将生成 YOLO 分类训练所需的 `train`、`val`、`test` 三个子集。本公开版本已包含本次实践使用的原始图片、划分后的图片和缓存文件，便于复现实验。

数据来源与引用信息请以根目录 README 为准。本项目训练所依据的公开数据源为 Mendeley Data 上的《Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality》，DOI 为 `10.17632/bdd69gyhv8.1`，许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。再分发这些图片或基于其训练的衍生成果时，请保留作者署名、DOI、许可链接和修改说明。
