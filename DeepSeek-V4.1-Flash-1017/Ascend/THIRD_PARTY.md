# 第三方来源及改动

`recipes/`、`calib/`、`patches/` 及原版后处理脚本来自
`chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend` 的 `release/msmodelslim/`。
`make_full_model.py` 来自作者 GitHub 仓库的固定提交；逐文件 URL、版本及 SHA256 见 `sources.lock.json`。

作者 GitHub 仓库的 Apache-2.0 许可文本保留在 `licenses/chiro-LICENSE`。
原始量化补丁作用于 Ascend/msmodelslim，工具仓库保留上游 MulanPSL-2.0 许可及原文件声明。
校准文本沿用作者发布的数据，不在此宣称其底层各数据集获得新的统一许可。
两个官方配置快照来自 DeepSeek 官方模型仓库，官方 1017 模型卡声明 MIT 许可；模型权重未包含在此仓库。

本项目的改动：

- 将 `quant_dp_inner.sh` 改为 `run_stage1.py` 入口，去除写死的路径和自动递归删除输出行为。
- 新增 `setup_msmodelslim.sh`、`run_stage1.py`、使用说明、来源锁定和配置快照。
- 原始量化配方、校准文本及其他原版脚本不修改；运行时生成的工作配方仅替换 dataset 为绝对路径。

未宣称原作者或 Ascend 对 1017 产物提供背书；旧模型测试结果不构成本项目的验收结果。
