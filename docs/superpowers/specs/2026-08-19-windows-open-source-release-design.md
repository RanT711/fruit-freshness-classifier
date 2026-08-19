# Windows 开源一键运行发布设计

## 目标

将“基于 YOLO 的水果新鲜度图片判别系统”发布为可公开获取的 Windows 项目。使用者无需准备数据或训练模型，只需下载仓库、执行安装脚本、启动网页，即可上传自己的 JPG/PNG 水果图片进行本地推理。

## 范围

### 包含

- Windows 一键安装与网页启动入口；
- 通过 GitHub Release 分发训练完成的 `best.pt`，而不是将其提交到 Git 仓库；
- 模型下载链接与 SHA-256 完整性校验；
- 可复现的 Windows 依赖版本；
- 模型可信路径限制和上传图片资源限制；
- 面向开源用户的 README、许可证、贡献说明和发布流程；
- 发布前测试、密钥扫描、依赖漏洞审计与 Git 内容检查。

### 不包含

- 数据集图片、训练日志、训练输出和预训练权重的重新分发；
- 摄像头、DJI Action 4、视频流或多水果检测；
- 公网部署、用户账号体系、云端推理或在线数据存储；
- macOS/Linux 的一键脚本支持。

## 使用者体验

```text
下载仓库
  ↓
双击 setup_windows.bat
  ├─ 检查 Python
  ├─ 创建 .venv
  ├─ 安装固定依赖
  └─ 下载并校验 Release 中的 best.pt
  ↓
双击 run_web.bat
  ↓
本机浏览器打开 127.0.0.1:8501
  ↓
上传 JPG/PNG 并得到新鲜/变质结论
```

GPU 是可选项：有兼容的 NVIDIA GPU 时由 PyTorch/Ultralytics 使用加速；无 GPU 时保留 CPU 推理能力。

## 发布结构

```text
repository/
  app.py
  train.py
  predict.py
  setup_windows.bat
  run_web.bat
  requirements-windows.txt
  scripts/download_model.py
  models/.gitkeep
  docs/
  tests/
  LICENSE
  CONTRIBUTING.md

GitHub Release v1.0.0/
  best.pt
  model-manifest.json
```

`model-manifest.json` 包含固定的 Release 下载地址、模型文件名、SHA-256、模型版本和训练信息。安装脚本调用 `scripts/download_model.py`：仅允许 HTTPS 下载，下载到临时文件，校验 SHA-256 成功后才原子替换 `models/best.pt`。

## 安全与隐私设计

1. `.gitignore` 继续排除 `data/`、`models/*.pt`、`runs/`、虚拟环境、缓存以及根目录下载的 `yolo26n.pt`。
2. 网页上传后先检查文件大小与像素数量，再用 Pillow 验证内容并转换 RGB；超出限制或无效文件不进入 YOLO。
3. 网页模型路径只允许解析到项目 `models/` 目录，拒绝任意外部 `.pt` 路径。命令行入口保留给受信任的本地使用者，并在文档中明确只加载可信模型。
4. Streamlit 默认绑定 `127.0.0.1`，安装和启动脚本不开放局域网或公网访问。
5. Release 资产校验失败时，安装停止并保留原有模型，不会运行未知权重。
6. 发布前必须执行密钥扫描、依赖漏洞审计、许可证核对和 `git status --ignored` 检查。

## 依赖与兼容性

- 支持 Windows 10/11 x64；
- 支持 Python 3.10–3.12；
- 使用经测试的固定版本依赖文件，避免无版本约束导致安装到不兼容的未来版本；
- 安装脚本不要求管理员权限；
- 若 Python 不存在或版本不支持，脚本明确提示安装要求后退出。

## 文档与许可证

- 代码与 Ultralytics YOLO 派生模型采用 AGPL-3.0；
- README 提供三步启动、截图、CPU/GPU 说明、模型局限、故障排查和数据/模型来源；
- `CONTRIBUTING.md` 说明不提交数据、密钥和训练输出；训练模型通过 GitHub Release 公开发布；
- 训练数据不随仓库或 Release 分发，保留公开来源链接、许可和引用说明；
- Release 页面说明模型只判断可见外观，不构成食品安全结论。

## 验收标准

1. 在一台没有本项目虚拟环境的 Windows 电脑上，运行 `setup_windows.bat` 后创建环境、安装依赖、下载并校验模型；
2. 运行 `run_web.bat` 后能访问本机网页；
3. 上传有效 JPG/PNG 可得到预测，上传无效或超限图片会得到中文错误；
4. 测试套件全部通过；
5. `pip-audit` 对发布依赖没有未处理的已知漏洞；
6. Git 仓库不包含图片数据、模型二进制、训练输出、凭据或用户路径；
7. GitHub Release 中的 `best.pt` SHA-256 与清单匹配；
8. 用户可只依照 README 完成安装和一次成功预测。

## 发布前提与命名

建议仓库名：`RanT711/fruit-freshness-classifier`。实际创建 GitHub 仓库与上传前，需要用户确认该名称或提供替代名称。项目以 AGPL-3.0 开源，完整公开代码、训练脚本、配置与 `best.pt` Release 资产。模型 Release 的发布前提是确认训练数据许可允许分发由其训练得到的权重；若无法确认，则仅开源代码与训练说明，不发布 `best.pt`。
