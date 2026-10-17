# msModelSlim DeepSeek-V4.1-Flash W4A8 补丁说明

> 交付对象：把 DeepSeek-V4.1 W4A8 量化适配做成可 apply 到上游
> [Ascend/msmodelslim](https://gitcode.com/Ascend/msmodelslim.git) 的补丁。

## 1. 基线 / 目标 commit

| 项 | 值 |
|---|---|
| 上游基线 commit | `92e219fa9565a5bad84d90474a27bb11524d691c`（`origin/master`，`[Bugfix] analysis linear dp bugfix`） |
| 补丁生成的实现 commit | `e05b47190392bace361a012f115d64f4abcca30d`（分支 `dsv41-w4a8`，标题 `feat(dsv41): DeepSeek-V4.1 W4A8 量化适配`） |
| 补丁形式 | `git diff`（raw diff，可被 `git apply -p1` 直接消费；文件头已注明 base commit） |
| **apply 目标** | **上游基线 `92e219f`**；适用目录为 msmodelslim 仓库根（含 `msmodelslim/`、`lab_practice/`、`config/` 的那一层） |

`e05b471` 是**打完本补丁后的结果**，因此不可在其之上再 apply（会报文件已存在）。生成的符号链接
`msmodelslim/config_repo` 不入补丁，由 apply 脚本现场创建。

## 2. 主补丁

| 项 | 值 |
|---|---|
| 文件 | `patches/msmodelslim_v41_w4a8.patch` |
| md5 | `ff375f28e6cf5d7492e71b353d34ba5e` |
| sha256 | `15f8368ea7813dbb337267d02136d330c36e23f681b892c5e866b2a9e286acfb` |
| 规模 | 1687 行 / 9 文件 |
| 内容 | `msmodelslim/model/deepseek_v41/{__init__,loader,model_adapter,model,rotation_map,convert}.py` 新建；`msmodelslim/model/deepseek_v4/model.py` 扩展（V4.1 复用其中构建块，新增开关默认值与 V4 一致）；`lab_practice/deepseek_v41/deepseek_v4_1_flash_w4a8.yaml` 配方；`config/config.ini` 注册 `deepseek_v41` 适配器 |
| 关键修复 | `V41MoE.forward` 的 DP gather/scatter：各 rank token 数不同会让 HCCL all_reduce 参数不一致，报 `EI0005/EI0006` |

适用的 recipe（主补丁内）：

```
路由专家 (40×384×w1/w2/w3)          -> W4A8_DYNAMIC (int4 per-channel SSZ + int8 per-token)
注意力 wq_a/wq_b/wkv                -> W8A8_DYNAMIC
indexer.wq_b (8 个 index source 层) -> W8A8_DYNAMIC
共享专家 w1/w2/w3                   -> W8A8_DYNAMIC
gate / norm / hc_* / wo_a / wo_b / compressor.* / indexer.wk/weights_proj / embed / head -> FLOAT
vision.* / aligner.* / mtp.*        -> 不在适配器参数内（FLOAT；vision 与 MTP 需事后另做，见 release 说明）
```

**补丁适用范围**：只覆盖主干 W4A8 量化（stage1）。补丁内的模型实现仅供量化标定阶段使用
（标定样本按整条序列送入），不影响权重的推理能力——推理由运行时 vLLM-Ascend 自身实现，
prefill / decode 均完整支持。补丁**不含** activation 量化、candidate 两级筛选、Engram、
DSpark MTP、VL 的适配代码。

## 3. apply 顺序（唯一合法顺序）

```bash
# 设 REPO = 干净的上游 msmodelslim checkout（HEAD 应为 92e219f）
REPO=/path/to/msmodelslim
REL=/path/to/release/msmodelslim        # 本发布材料所在目录

# 干跑：只 check，不改树
bash $REL/scripts/apply_msmodelslim_patch.sh --dir $REPO --check

# 正式应用（主补丁 + 现场创建 editable 安装所需的目录软链）
bash $REL/scripts/apply_msmodelslim_patch.sh --dir $REPO

# 只想手动来：
cd $REPO
git apply --check -p1 $REL/patches/msmodelslim_v41_w4a8.patch
git apply       -p1 $REL/patches/msmodelslim_v41_w4a8.patch
```

规则：

1. **先 `--check` 再 apply**；任何一步非 0 就停下，不要用 `--reject` / `--3way`。
2. 目标树必须是干净的上游基线（或包含未改动补丁涉及文件的后代提交）；HEAD 不等于
   `92e219f` 时脚本只警告，`git apply --check` 才是真正的判定。
3. 应用后必须刷新 editable 安装的包内配置：
   `config/config.ini` -> `msmodelslim/config/config.ini`，否则 CLI 看不到
   `deepseek_v41` 适配器。`apply_msmodelslim_patch.sh` 默认自动做（`--no-layout` 可关）。

## 4. 验证记录

- 主补丁去掉 7 行注释头后，与 `git diff 92e219f..e05b471` **逐字节相同**；
- 在干净的 `92e219f` 树上 `git apply --check` + `git apply` **通过**；
- apply 后与 `e05b471` 工作树比对：`deepseek_v41/model.py`、
  `deepseek_v4_1_flash_w4a8.yaml`、`config/config.ini` **完全一致**；
- 在 `e05b471`（已含补丁内容）上 apply **失败**，符合预期。

## 5. 不纳入补丁、但必须现场准备的东西

| 项 | 说明 |
|---|---|
| `msmodelslim/config_repo -> ../config` | CLI 从包内找 config_repo；apply 脚本创建 |
| `msmodelslim/lab_practice -> ../lab_practice` | 同上（`.gitignore` 已忽略） |
| `msmodelslim/lab_calib -> ../lab_calib` | 同上 |
| `msmodelslim/config/config.ini` | 从根目录 `config/config.ini` 复制，确保 `deepseek_v41` 注册生效 |
| Python 环境、官方 checkpoint、量化输出目录 | 环境与数据，不属于源码补丁 |
