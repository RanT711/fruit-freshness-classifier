# Contributing

感谢你为这个项目做贡献。

在提交 Pull Request 之前，请先确认下面这些规则：

## 不要提交这些内容

- 原始数据、处理后的数据和任何数据集压缩包；
- 模型权重、临时下载文件和训练产物，例如 `models/*.pt`、`models/best.pt.part`、`runs/`；
- 凭据、密钥、Token、Cookie、账号密码；
- 带有个人机器路径、个人用户名、个人桌面目录的截图、日志或配置；
- 与当前任务无关的大文件和临时缓存。

## 提交前必须完成的检查

在项目根目录执行：

```powershell
python -m pytest -q
python -m pip_audit -r requirements-dev.txt
git diff --check
```

如果你的环境里 `pip-audit` 安装成命令行工具，也可以直接运行：

```powershell
pip-audit -r requirements-dev.txt
```

## 变更要求

- 保持 Windows 双击启动流程可用；
- 不要绕过模型下载的 SHA-256 校验逻辑；
- 不要放宽受信模型路径限制；
- 修改文档时，许可证表述必须保持为 AGPL-3.0，不要写回 MIT 或其他不一致说法；
- 如果涉及数据来源说明，必须保留 Mendeley 数据记录、CC BY 4.0 和 DOI 引用。

## Pull Request 说明建议

请在 PR 描述中简要说明：

- 你改了什么；
- 你为什么这样改；
- 你跑了哪些检查；
- 是否影响训练、预测、模型下载或网页启动流程。
