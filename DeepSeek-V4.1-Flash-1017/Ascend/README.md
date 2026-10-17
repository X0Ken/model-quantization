# DeepSeek-V4.1-Flash-1017 / Ascend W4A8

基于 Chiro 已发布的旧版 V4.1 量化材料，准备官方 **1017 checkpoint** 的制作工具。
2026-10-17 核查时，1017 与旧版官方 `config.json` 内容相同；本项目尚未执行 1017 的 NPU 量化、完整模型推理或质量验收。配置相同不代表权重相同或新版已经验收。

## 文件

| 路径 | 用途 |
|---|---|
| `recipes/deepseek_v4_1_flash_w4a8.yaml` | 原版配方，专家 W4A8、选定注意力及共享专家 W8A8 |
| `calib/mix_calib.jsonl` | 作者发布的 48 条校准文本；不可同时充当独立评测集 |
| `patches/` | 原版主补丁及基线说明 |
| `scripts/setup_msmodelslim.sh` | 克隆已打补丁工具、补配置布局、安装 editable 包 |
| `scripts/run_stage1.py`、`quant_dp_inner.sh` | 参数化主干量化入口，拒绝覆盖已有工作目录 |
| `scripts/make_full_model.py` | 从官方权重装配 stage1 输入，裁去 Engram、MTP、Vision |
| 其余 `scripts/*.py` | DSpark、MTP 量化、Vision 旋转、Engram 转换和组装 |
| `sources.lock.json` | 下载地址、原文件版本及 SHA256；修改文件另记上游 SHA256 |
| `reference/` | 本次取得的两个官方配置快照 |

## 环境与工具

在匹配 Ascend 的隔离 Python/容器环境中执行。作者量化参考环境是 A3、CANN 9.1.0、Python 3.12.13、torch 2.10.0、torch_npu 2.10.0.post4、Transformers 5.14.1；这不是 910B3 上已验收的软件组合。A2/910B3 需要使用匹配其驱动和 CANN 的环境，先做局部试跑。不要将 A3 镜像直接视为 A2 可用镜像。

torch、torch_npu、Transformers、safetensors 由兼容的环境提供。脚本不自动安装或替换 CANN/驱动。
原始后处理脚本还需要 CPU torch 和 safetensors。CPU Engram 转换会分配很大的主机内存。

```bash
git clone https://github.com/X0Ken/model-quantization.git
cd model-quantization/DeepSeek-V4.1-Flash-1017/Ascend
# 在已准备好的 Ascend Python 环境内运行；工具路径应为绝对路径。
bash scripts/setup_msmodelslim.sh /workspace/msmodelslim
```

工具固定到 [X0Ken/msmodelslim@c61748f](https://github.com/X0Ken/msmodelslim/commit/c61748f295dc8778a978fbf97ee790609f9ed2ea)，分支 `deepseek-v41-w4a8`。
该提交基于上游 `92e219fa9565a5bad84d90474a27bb11524d691c` 应用作者主补丁。**已包含补丁，不要再运行 apply。**
`setup_msmodelslim.sh` 在安装环境中重建未纳入源码补丁的配置布局。

若自行从原始上游制作，请遵循 `patches/PATCHES.md`：在干净的固定基线上先 `--check` 再 apply。
本目录只收录发布版主补丁，不收录未随该版本发布的可选 hi-aux 补丁。

## Stage1 主干量化

先将 [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017) 完整下载到本地，包含索引、全部权重分片及 tokenizer。
下面的路径是示例，需要替换成实际路径。所有阶段的输出放在独立、本地存储目录，源模型只读。

```bash
# 只预览命令：不写文件、不调用 NPU。
python3 scripts/run_stage1.py \
  --model /models/DeepSeek-V4.1-Flash-1017 \
  --tool /workspace/msmodelslim \
  --work /outputs/flash-1017-dp8 --devices 0 1 2 3 4 5 6 7 --dry-run

# 确认设备空闲后，去掉 --dry-run 执行；--work 必须不存在。
# 局部排错可以增加 --layers 7；局部产物不能用于完整模型服务。
```

`bash scripts/quant_dp_inner.sh ...` 接受同一组参数。
入口校验固定工具提交、工作树、官方模型类型与分片存在性；创建 stage1 输入和工作配方，
把工作配方中的 dataset 改为本仓库数据的绝对路径，再调用量化工具。原始 YAML 保持不变。
运行参数保存到工作目录 `run.json`，控制台日志由调用方保存；该文件的 `status=started` 仅表示启动，不表示验收。
输出位于 `--work/stage1`，**只完成主干**。

校准设备数会影响产物，默认 DP8 不等于作者 DP16 参考产物。固定设备数、软件、配方及校准集后再比较。
QuaRot 矩阵由这次量化产生，后处理必须使用同一产物的 `optional/quarot.safetensors`，不要拷旧模型矩阵。

## 完整模型后处理

以下为原版脚本的调用顺序示例，尚未在 1017 上执行。每一步先核对输入与输出，再继续；不要重跑覆盖已有产物。
`OFFICIAL`、`WORK` 均需为绝对路径，且 `WORK/stage1` 应来自全 40 层量化。

```bash
OFFICIAL=/models/DeepSeek-V4.1-Flash-1017
WORK=/outputs/flash-1017-dp8

# 1. 补 DSpark，解码官方 MTP 权重并对齐主干旋转。
python3 scripts/make_dspark_ckpt.py --quant-dir "$WORK/stage1" \
  --official-dir "$OFFICIAL" --out-dir "$WORK/dspark"

# 2. 对草稿量化；保留默认敏感层精度。
python3 scripts/mtpq_quantize_draft.py --quantize --src "$WORK/dspark" --out "$WORK/mtpq"
python3 scripts/mtpq_quantize_draft.py --verify --src "$WORK/dspark" --out "$WORK/mtpq"

# 3. 补 Vision 权重，再对视觉输出边界做同基底旋转。
python3 scripts/add_vision_weights.py --official-dir "$OFFICIAL" --target-dir "$WORK/mtpq"
python3 scripts/make_qrot_vision_product.py --src "$WORK/mtpq" --dst "$WORK/vision-qrot"

# 4. 转换官方 1017 Engram 表，并组装其投影、索引及配置。
python3 scripts/engram_convert_int8.py --src "$OFFICIAL" --dst "$WORK/engram-int8"
python3 scripts/engram_dr_build.py --src "$WORK/vision-qrot" --official "$OFFICIAL" \
  --int8 "$WORK/engram-int8" --out "$WORK/final"
python3 scripts/prepare_runtime_ckpt.py --dir "$WORK/final"
```

原后处理脚本存在软链接及原地写入，相关中间目录必须保留；`final` 不是可任意单独搬走的自包含目录。
运行前请查看各脚本 `--help`；不要启用 `--force`、`--overwrite`、`--aggressive` 或额外量化开关来覆盖默认路线。
`prepare_runtime_ckpt.py` 只整理格式与检查索引，不证明模型质量或推理环境兼容。

完整推理还需要匹配 Engram、旋转与 DSpark 契约的引擎。这里不安装或覆盖 910B3 当前 QC/vLLM 服务。
新产物要单独验收文本、工具调用、长上下文和视觉，不能沿用旧版权重的分数。

## 来源

- [Chiro 模型发布及制作材料](https://modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend)
- [作者量化指南](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/quant/REPRO_W4A8_QUANT.md)
- [Ascend/msmodelslim](https://github.com/Ascend/msmodelslim)

不同日期的作者文档有阶段性差异；本目录锁定实际发布文件，不将旧验收数据记作 1017 的结果。
