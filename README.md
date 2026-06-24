# CCA-MIL

**Visual-Prototype Adaptive / Unbalanced OT MIL for Few-Shot Whole Slide Image Classification**

本仓库是在 [dddavid4real/FOCUS](https://github.com/dddavid4real/focus) 代码结构上改造的少样本 WSI 分类项目。数据集、WSI 预处理、CONCH 特征、few-shot split 和训练脚本约定基本沿用 FOCUS；当前默认模型从原始 FOCUS 改为 `CCA_MIL`。最新版 `CCA_MIL` 使用可学习 visual prototypes 从 patch features 中聚合视觉证据，并通过 adaptive / unbalanced OT 与全局 concept bank 对齐，形成 `patch -> visual prototype -> concept -> class` 的可解释证据链。

FOCUS 原论文为 **FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification, CVPR 2025**。原仓库说明中使用 TCGA-NSCLC、CAMELYON 和 UBC-OCEAN 三个数据集，WSI patch size 设为 `512`，放大倍率为 `40X`，patch feature extractor 使用 `CONCH`。如果以 Libra-MIL 论文主表作为对比基准，实验部分按其设置统一为 `1/4/16-shot`、`5-fold` mean/std；本仓库额外生成 `8-shot` split，方便调参和中间 shot 对比。

## 目录

- [项目关系](#项目关系)
- [方法概览](#方法概览)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [WSI 预处理](#wsi-预处理)
- [Few-shot Splits](#few-shot-splits)
- [Concept Bank](#concept-bank)
- [训练](#训练)
- [调参脚本与指标](#调参脚本与指标)
- [关键参数](#关键参数)
- [输出文件](#输出文件)
- [代码结构](#代码结构)
- [常见问题](#常见问题)
- [引用](#引用)

## 项目关系

本项目继承并保留了 FOCUS / ViLa-MIL / CLAM 风格的数据接口：

- 每张 WSI 对应一个 `.pt` feature file。
- `main.py` 通过 `Generic_MIL_Dataset` 读取 slide-level 标签和 split。
- `splits/` 中提供 1-shot、4-shot、8-shot、16-shot 的 5-fold few-shot split。
- `LUAD_LUSC.sh`、`camelyon.sh`、`UBC-OCEAN.sh` 是三个数据集的训练入口。

与 FOCUS 的主要区别：

- 当前默认 `--model_type CCA_MIL`。
- 原始 `models/model_FOCUS.py` 已恢复，可通过 `--model_type FOCUS` 复现实验并与 `CCA_MIL` 对比。
- CCA-MIL 不再使用整段类别文本 prompt 直接指导所有 patch，而是使用结构化 `concept_bank` 和视觉 prototype evidence。
- CCA-MIL 的 forward 中只使用高分辨率特征 `x_l`，但当前 DataLoader 仍会读取 `data_folder_s` 和 `data_folder_l`。如果没有低分辨率特征，可以把 `--data_folder_s` 和 `--data_folder_l` 指向同一个 CONCH `pt_files` 目录。

## 方法概览

当前 `CCA_MIL` 已更新为更简洁的 Visual-Prototype Adaptive / Unbalanced OT MIL。整体流程：

```text
Concept bank
  -> WSI patch features
  -> patch projector
  -> learnable visual prototypes
  -> patch-to-prototype soft assignment
  -> visual evidence tokens
  -> adaptive / unbalanced OT with global concept bank
  -> discriminative visual evidence Z_dis
  -> pooling over Z_dis
  -> classifier
```

核心模块：

- `Concept bank`：每个类别维护 `common_concepts` 和 `discriminative_concepts`。
- `Visual prototypes`：可学习视觉原型锚点，只负责从 patch features 中软聚合视觉证据。
- `Adaptive / unbalanced OT`：在视觉证据 token 与全局 concept bank 之间做 soft transport assignment，支持视觉原型数和 concept 数不一致。
- `Discriminative evidence`：分类只使用 `Z_dis`，common concepts 仅用于解释和可选 contrastive negative。
- `Training losses`：默认 CE，可选 class-aware discriminative contrastive loss 和 GT-class diversity loss。

## 环境准备

建议使用独立 conda 环境。具体版本可以按你的 CUDA 环境调整。

```bash
conda create -n cca_mil python=3.10
conda activate cca_mil

pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn scipy h5py openslide-python pillow tqdm
pip install tensorboardX ml-collections
```

还需要安装并配置这些外部项目：

- [CLAM](https://github.com/mahmoodlab/CLAM)：用于 WSI tissue segmentation、patch coordinate extraction 和 feature extraction。
- [CONCH](https://github.com/mahmoodlab/CONCH)：用于提取 512 维 patch embedding，并在 CCA-MIL 中编码 concept text。
- [ViLa-MIL](https://github.com/Jiangbo-Shi/ViLa-MIL)：用于理解 few-shot split 和两尺度特征接口。

训练前需要准备 CONCH checkpoint：

```text
ckg/pytorch_model.bin
```

默认路径可通过 `--conch_ckpt_path` 覆盖。如果你的 CONCH 安装使用其他 checkpoint 格式或路径，请在训练和调参脚本中传入对应路径。

## 数据准备

FOCUS 使用三个数据集，本仓库沿用这些任务名和类别定义：

| 数据集 | 下载来源 | `--task` | 类别 |
| --- | --- | --- | --- |
| TCGA-NSCLC | [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/) | `task_tcga_lung_subtyping` | `LUAD`, `LUSC` |
| CAMELYON16/17 | [CAMELYON16](https://camelyon16.grand-challenge.org/Data/), [CAMELYON17](https://camelyon17.grand-challenge.org/Data/) | `task_camelyon_subtyping` | `normal`, `tumor` |
| UBC-OCEAN | [Kaggle UBC-OCEAN](https://www.kaggle.com/competitions/UBC-OCEAN) | `task_UBC-OCEAN_subtyping` | `CC`, `HGSC`, `LGSC`, `EC`, `MC` |

每个任务需要一个 slide-level CSV，至少包含：

```csv
case_id,slide_id,label
case_001,slide_001,LUAD
case_002,slide_002,LUSC
```

注意：

- `slide_id` 不带文件扩展名。
- `slide_id` 必须和特征文件名一致，例如 `slide_001` 对应 `slide_001.pt`。
- `label` 必须与 `main.py` 中对应任务的 `label_dict` 一致。
- `main.py` 支持通过 `--csv_path` 指定本机数据 CSV；不传时才会回退到原作者机器上的默认路径。
- CAMELYON split 中的 `patient_XXX_node_Y` 是 CAMELYON17 官方 ID，`slide_N` 是 FOCUS 风格的 CAMELYON16 重命名 ID。映射规则和生成脚本见 [splits/README_camelyon_mapping.md](splits/README_camelyon_mapping.md)。

建议的数据目录：

```text
DATA_ROOT/
  raw_wsi/
    tcga_nsclc/
      slide_001.svs
      slide_002.svs
    camelyon/
      slide_a.tif
    ubc_ocean/
      image_001.tif
  features/
    tcga_nsclc_conch_40x_512/
      h5_files/
      pt_files/
    camelyon_conch_40x_512/
      h5_files/
      pt_files/
    ubc_ocean_conch_40x_512/
      h5_files/
      pt_files/
  dataset_csv/
    LUAD_LUSC.csv
    camelyon.csv
    UBC-OCEAN.csv
```

## WSI 预处理

FOCUS 的 README 只简要写到参考 CLAM、patch size 为 `512`、magnification 为 `40X`、feature extractor 为 CONCH。下面把完整流程展开成可执行步骤。本仓库额外提供两个加速脚本：

```text
fast_create_patches_fp.py     # 多进程 WSI segmentation / patch coordinate extraction
fast_extract_features_fp.py   # 多 GPU patch feature extraction
```


### 1. 准备 CLAM

如果使用原版 CLAM 脚本，建议单独准备 CLAM：

```bash
git clone https://github.com/mahmoodlab/CLAM.git
cd CLAM
conda env create -f env.yml
conda activate clam_latest
```


如果你使用本项目自己的环境，也要确保 `openslide-python`、OpenSlide system library、`h5py`、`torch` 和 CONCH 依赖可用。

### 2. Tissue segmentation 和 patch coordinate extraction

以 TCGA-NSCLC 为例。FOCUS 设置为 `512` pixel patch、`40X` magnification。对于 40X 扫描的 TCGA `.svs`，通常使用 `patch_level 0`。

推荐使用本仓库的并行版本：

```bash
python fast_create_patches_fp.py \
  --source /data2/yuhaowang/WSIFew/TCGA-NSCLC \
  --save_dir /data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC \
  --patch_size 512 \
  --step_size 512 \
  --patch_level 0 \
  --preset tcga.csv \
  --seg \
  --patch \
  --stitch \
  --num_workers 24 \
  --contour_workers 1 \
  --slide_exts .svs,.tif,.tiff
```


并行脚本说明：

- `--num_workers`：同时处理的 WSI 数量。一般设置为 CPU 核数的一半到 CPU 核数之间；如果磁盘 I/O 压力很大，适当调低。
- `--contour_workers`：每个 WSI worker 内部用于坐标过滤的进程数。很多 WSI 同时跑时建议设为 `1`；只有少量超大 WSI 时可设为 `2` 到 `4`。
- `--resume_process_list`：默认开启，会优先读取 `save_dir/process_list_autogen.csv` 继续上次未完成的 slide。
- `--no_auto_skip`：关闭自动跳过。默认会检查 `patches/{slide_id}.h5`、`masks/{slide_id}.jpg`、`stitches/{slide_id}.jpg`。
- `--process_list process_list_edited.csv`：使用手动修改过的参数表继续处理。
- `--slide_ext .svs`：当 process list 中的 `slide_id` 不带后缀时自动补后缀。

输出结构：

```text
clam_processed/tcga_nsclc_40x_512/
  masks/
    slide_001.png
  patches/
    slide_001.h5
  stitches/
    slide_001.png
  process_list_autogen.csv
```

其中：

- `masks/`：组织区域分割结果。
- `patches/*.h5`：patch 坐标，不保存实际图像 patch。
- `stitches/`：patch 覆盖区域的可视化检查图。
- `process_list_autogen.csv`：每张 slide 的处理参数记录。

CAMELYON 可以使用同样流程。如果没有合适的 preset，可以先去掉 `--preset tcga.csv`，再根据 `masks/` 和 `stitches/` 检查结果调整分割参数。

```bash
python create_patches_fp.py \
  --source /data2/yuhaowang/WSIFew/camelyon \
  --save_dir /data2/yuhaowang/WSIFew/processd_wsi/camelyon_40x_512 \
  --patch_size 512 \
  --patch_level 0 \
  --seg \
  --patch \
  --stitch
```


### 2.1 UBC-OCEAN PNG 的特殊处理

UBC-OCEAN 下载后是超大的单层 `.png` 图像。不要直接对原始 PNG 跑下面这种命令：

```bash
python fast_create_patches_fp.py \
  --source /data2/yuhaowang/WSIFew/UBC-OCEAN \
  --save_dir /data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN \
  --patch_size 512 \
  --step_size 512 \
  --patch_level 0 \
  --seg \
  --patch \
  --stitch \
  --slide_exts .png
```

原因是 OpenSlide 不直接支持普通 PNG，会 fallback 到 PIL `ImageSlide`；UBC-OCEAN 中部分图片超过 20 亿像素，容易触发 PIL 的 `DecompressionBombError`。即使取消该限制，PNG 也只有 level 0，没有低分辨率 pyramid，CLAM/FOCUS 的 tissue segmentation 会试图在超大的 level 0 上做分割，内存和速度都不合适。

推荐流程是先用 `vips` 把 PNG 转为 tiled pyramidal TIFF，再走标准 OpenSlide / CLAM patch 流程。当前机器已经有 `vips` 命令；若你的环境没有，需要先安装 `libvips-tools`。

先转换：

```bash
python tools/convert_ubc_png_to_pyramid_tiff.py \
  --source /data2/yuhaowang/WSIFew/UBC-OCEAN \
  --dest /data2/yuhaowang/WSIFew/UBC-OCEAN-pyramid-tiff \
  --workers 4 \
  --compression jpeg \
  --quality 90 \
  --tile-size 512
```



转换完成后，对 `.tif` 目录做 segmentation 和 patch coordinate extraction：

```bash
python fast_create_patches_fp.py \
  --source /data2/yuhaowang/WSIFew/UBC-OCEAN-pyramid-tiff \
  --save_dir /data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN \
  --patch_size 512 \
  --step_size 512 \
  --patch_level 0 \
  --seg \
  --patch \
  --stitch \
  --num_workers 12 \
  --contour_workers 1 \
  --slide_exts .tif,.tiff \
  --slide_ext .tif
```

建议：

- `--num_workers` 不要一开始设太大。UBC 图像很大，建议先用 `--limit_slides 5` 检查 `masks/` 和 `stitches/`，确认分割合理后再全量跑。
- 如果上次直接跑 PNG 已经生成了 `process_list_autogen.csv` 且里面有失败状态，可以直接复用同一个 `save_dir`，改成 TIFF source 后加 `--refresh_pending_params` 继续跑 pending slides。
- `tools/convert_ubc_png_to_pyramid_tiff.py` 默认会跳过已经存在的 `.tif`，中断后可直接重跑。

UBC-OCEAN 的 CONCH feature extraction 也要指向转换后的 TIFF 目录：

```bash
python fast_extract_features_fp.py \
  --data_h5_dir /data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN \
  --data_slide_dir /data2/yuhaowang/WSIFew/UBC-OCEAN-pyramid-tiff \
  --csv_path /data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN/process_list_autogen.csv \
  --feat_dir /data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN/feature \
  --slide_ext .tif \
  --model_name conch_v1 \
  --conch_ckpt_path /home/yuhaowang/project/WSIFew/cca_mil/ckg/pytorch_model.bin \
  --target_patch_size 448 \
  --batch_size 1024 \
  --gpus 4,5,6,7 \
  --num_workers 12
```

生成的训练特征目录是：

```text
/data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN/feature/pt_files
```

训练或调参时把 UBC feature 路径设置为这个 `pt_files` 目录。


关于倍率：

- FOCUS 报告的设置是 `40X`。如果 level 0 本身是 40X，使用 `--patch_level 0` 即可。
- 如果某个数据集 level 0 是 20X，则 `--patch_level 0` 得到的是 20X patch，不是 40X。此时需要在实验记录中说明，或使用与你复现实验一致的扫描倍率和 downsample 设置。
- `patch_size 512` 指从 WSI 中读取的原始 patch 边长。后续 CONCH encoder 可能会在输入 transform 中 resize 或 crop，这不改变 patch coordinate extraction 的设置。

### 3. 使用 CONCH 提取 patch features

CLAM 新版推荐 `extract_features_fp.py`。如果使用 CONCH，需要先按 CLAM / CONCH 的说明下载 checkpoint，并设置环境变量：

```bash
export CONCH_CKPT_PATH=/home/yuhaowang/project/WSIFew/cca_mil/ckg/pytorch_model.bin
```

准备一个 feature extraction CSV。CLAM 的 `extract_features_fp.py` 需要 CSV 中的 slide 名称与 WSI 文件名匹配，通常使用 `process_list_autogen.csv`，并确保 slide id 不带扩展名。

然后提取 CONCH 特征。推荐使用本仓库的多 GPU 版本：

```bash
python fast_extract_features_fp.py \
  --data_h5_dir /data2/yuhaowang/WSIFew/processd_wsi/TCGA-RCC \
  --data_slide_dir /data2/yuhaowang/WSIFew/TCGA-RCC \
  --csv_path /data2/yuhaowang/WSIFew/processd_wsi/TCGA-RCC/process_list_autogen.csv \
  --feat_dir /data2/yuhaowang/WSIFew/processd_wsi/TCGA-RCC/feature \
  --slide_ext .svs \
  --model_name conch_v1 \
  --conch_ckpt_path /home/yuhaowang/project/WSIFew/cca_mil/ckg/pytorch_model.bin \
  --target_patch_size 448 \
  --batch_size 512 \
  --gpus 0,1,2,3,4,5,6,7 \
  --num_workers 6
```

也可以使用 CLAM 原版单 GPU 脚本：

```bash
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
  --data_h5_dir /data2/yuhaowang/WSIFew/processd_wsi/tcga_nsclc_40x_512 \
  --data_slide_dir /data2/yuhaowang/WSIFew/tcga_nsclc \
  --csv_path /data2/yuhaowang/WSIFew/processd_wsi/tcga_nsclc_40x_512/process_list_autogen.csv \
  --feat_dir /path/to/DATA_ROOT/features/tcga_nsclc_conch_40x_512 \
  --batch_size 256 \
  --slide_ext .svs \
  --model_name conch_v1 \
  --target_patch_size 448
```

说明：

- `--data_h5_dir` 指向上一步生成的目录，里面应有 `patches/*.h5`。
- `--feat_dir` 是输出目录。
- CONCH v1 输出 512 维 patch feature，本项目 `CCA_MIL` 默认输入维度也是 `512`。
- 如果 GPU 显存不足，降低 `--batch_size`。
- `--target_patch_size` 应与你的 CONCH / CLAM 版本保持一致。CONCH 常用输入尺寸为 `448`；如果你使用已经由 FOCUS 或其他 pipeline 提供的特征，保持其原始设置即可。
- `fast_extract_features_fp.py` 默认以 `pt_files/{slide_id}.pt` 作为完成标志，检测到已有 `.pt` 会自动跳过。
- `--gpus auto` 会使用所有可见 GPU；也可以显式指定 `--gpus 0,1`。
- `--num_workers` 是每张 GPU 对应的 DataLoader worker 数，不是总 worker 数。总 CPU worker 约为 `GPU数量 * num_workers`。
- `--overwrite` 会强制重算已有特征；`--verify_outputs` 会读取已有 `.pt` 做更严格检查。
- `--amp` 可以开启自动混合精度以进一步加速，但为了保持特征与原始 CONCH fp32 流程一致，默认关闭。

输出结构：

```text
features/tcga_nsclc_conch_40x_512/
  h5_files/
    slide_001.h5
  pt_files/
    slide_001.pt
```

本项目训练时只读取 `.pt`：

```text
/path/to/DATA_ROOT/features/tcga_nsclc_conch_40x_512/pt_files/{slide_id}.pt
```

每个 `.pt` 文件通常是：

```text
[num_patches, 512]
```

### 4. 本项目中的路径连接

运行 `main.py` 时：

```bash
--data_folder_l /path/to/DATA_ROOT/features/tcga_nsclc_conch_40x_512/pt_files
--data_folder_s /path/to/DATA_ROOT/features/tcga_nsclc_conch_40x_512/pt_files
```

当前 CCA-MIL forward 会忽略 `x_s`，但 DataLoader 仍会加载它，所以 `data_folder_s` 也必须是一个存在且包含同名 `.pt` 文件的目录。没有低分辨率特征时，最简单的做法是让 `data_folder_s` 和 `data_folder_l` 指向同一个 `pt_files` 目录。

## Few-shot Splits

本仓库已经提供三个数据集的 1-shot、4-shot、8-shot、16-shot、5-fold split。Libra-MIL 主表使用 `1/4/16-shot`，这里额外保留 `8-shot` 作为中间 shot 调参和补充实验：

```text
splits/
  LUAD_LUSC_1shots_5folds/
  LUAD_LUSC_4shots_5folds/
  LUAD_LUSC_8shots_5folds/
  LUAD_LUSC_16shots_5folds/
  camelyon_1shots_5folds/
  camelyon_4shots_5folds/
  camelyon_8shots_5folds/
  camelyon_16shots_5folds/
  UBC-OCEAN_1shots_5folds/
  UBC-OCEAN_4shots_5folds/
  UBC-OCEAN_8shots_5folds/
  UBC-OCEAN_16shots_5folds/
```

每个 split 文件格式：

```text
splits_0.csv
splits_1.csv
...
splits_4.csv
```

CSV 列为：

```csv
train,val,test
slide_001,slide_021,slide_101
slide_002,slide_022,slide_102
```

注意：

- `--k 5` 表示跑 5 个 fold。
- `--split_dir LUAD_LUSC_16shots_5folds` 会自动解析为 `splits/LUAD_LUSC_16shots_5folds`。
- split 中的 slide id 必须能在 dataset CSV 和 feature `pt_files` 中找到。
- 如果重新生成 split，请保持列名 `train,val,test` 和 slide id 命名一致。

重新生成所有 5-fold few-shot split：

```bash
python tools/create_libra_fewshot_splits.py \
  --datasets all \
  --shots 1,4,8,16 \
  --folds 5 \
  --seed 1 \
  --overwrite
```

## Concept Bank

CCA-MIL 使用 JSON concept bank。默认提供：

```text
text_prompt/concept_bank/tcga_nsclc.json
text_prompt/concept_bank/camelyon.json
text_prompt/concept_bank/ubc_ocean.json
```

格式示例：

```json
{
  "LUAD": {
    "common_concepts": [
      "tumor cell atypia",
      "inflammatory infiltrates",
      "irregular tumor margins"
    ],
    "discriminative_concepts": [
      "irregular glandular formation",
      "acinar growth pattern",
      "lepidic growth along alveolar walls",
      "mucin producing tumor cells"
    ]
  },
  "LUSC": {
    "common_concepts": [
      "tumor cell atypia",
      "necrotic tumor area",
      "solid tumor nests"
    ],
    "discriminative_concepts": [
      "keratin pearl formation",
      "intercellular bridges",
      "squamous cell differentiation",
      "eosinophilic cytoplasm with distinct cell borders"
    ]
  }
}
```

要求：

- 顶层 key 必须与 `main.py` 中的 `args.class_names` 一致。
- `common_concepts` 表示多个类别都可能出现的通用病理现象。
- `discriminative_concepts` 表示更有类别区分度的形态学证据。
- 建议 concept text 使用纯病理形态学描述，避免直接写类别名或类别同义词，降低 label leakage 风险。

`utils/concept_loader.py` 会把 common concepts 的权重设置为 `--common_concept_weight`，discriminative concepts 的权重设置为 `1.0`。

## 训练

训练前确认：

```bash
cd cca_mil
mkdir -p logs ckpts
```

确认这些文件和目录存在：

```text
ckg/pytorch_model.bin
text_prompt/concept_bank/*.json
splits/<split_dir>/splits_0.csv ... splits_9.csv
/path/to/features/<dataset>_conch_40x_512/pt_files/{slide_id}.pt
```

### 直接运行脚本

先编辑脚本中的数据路径：

```bash
--data_folder_s '/data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files'
--data_folder_l '/data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files'
```

如果只有一套 CONCH 特征，两个参数都指向同一个 `pt_files`：

```bash
--data_folder_s '/path/to/DATA_ROOT/features/tcga_nsclc_conch_40x_512/pt_files'
--data_folder_l '/path/to/DATA_ROOT/features/tcga_nsclc_conch_40x_512/pt_files'
```

运行：

```bash
bash LUAD_LUSC.sh
bash run_focus_tcga_nsclc.sh
bash camelyon.sh
bash UBC-OCEAN.sh
```

### 手动运行 TCGA-NSCLC

```bash
CUDA_VISIBLE_DEVICES=7 python main.py \
  --seed 1 \
  --drop_out \
  --early_stopping \
  --early_stopping_patience 15 \
  --early_stopping_stop_epoch 0 \
  --max_epochs 80 \
  --lr 1e-4 \
  --k 5 \
  --label_frac 1 \
  --bag_loss ce \
  --task task_tcga_lung_subtyping \
  --results_dir /data2/yuhaowang/cca-mil-result/results/CCA_MIL/conch/ \
  --exp_code LUAD_LUSC_16shots_5folds \
  --model_type CCA_MIL \
  --mode transformer \
  --log_data \
  --data_folder_s /data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files/ \
  --data_folder_l /data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files/ \
  --split_dir LUAD_LUSC_16shots_5folds \
  --concept_bank_path text_prompt/concept_bank/tcga_nsclc.json \
  --num_visual_prototypes 6 \
  --proto_tau 0.1 \
  --ot_epsilon 0.05 \
  --sinkhorn_iter 20 \
  --uot_rho_a 0.5 \
  --uot_rho_b 0.5 \
  --concept_pooling attention \
  --lambda_contrast 0.1 \
  --lambda_div 0.01
```

### 运行原始 FOCUS：TCGA-NSCLC

在本机已经补齐的 TCGA-NSCLC CONCH 特征上，可以直接测试原始 FOCUS 模型：

```bash
cd /home/yuhaowang/project/WSIFew/cca_mil
conda activate hest
bash run_focus_tcga_nsclc.sh
```

默认使用：

```text
dataset_csv = /home/yuhaowang/project/WSIFew/cca_mil/dataset_csv/LUAD_LUSC.csv
features    = /data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files
split_dir   = splits/LUAD_LUSC_16shots_5folds
prompt      = text_prompt/TCGA_Lung_two_scale_text_prompt.csv
checkpoint  = /home/yuhaowang/project/WSIFew/cca_mil/ckg/pytorch_model.bin
```

只跑第 0 个 fold 做快速检查：

```bash
DEVICE=0 K_START=0 K_END=1 SHOTS=16 bash run_focus_tcga_nsclc.sh
```

完整命令等价于：

```bash
python main.py \
  --seed 1 \
  --drop_out \
  --early_stopping \
  --early_stopping_patience 15 \
  --early_stopping_stop_epoch 0 \
  --max_epochs 80 \
  --lr 1e-4 \
  --k 5 \
  --label_frac 1 \
  --bag_loss ce \
  --task task_tcga_lung_subtyping \
  --csv_path /home/yuhaowang/project/WSIFew/cca_mil/dataset_csv/LUAD_LUSC.csv \
  --results_dir /data2/yuhaowang/cca-mil-result/results/FOCUS/conch/ \
  --exp_code LUAD_LUSC_16shots_5folds \
  --model_type FOCUS \
  --mode transformer \
  --log_data \
  --data_root_dir /data2/yuhaowang/WSIFew \
  --data_folder_s /data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files \
  --data_folder_l /data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files \
  --split_dir LUAD_LUSC_16shots_5folds \
  --text_prompt_path text_prompt/TCGA_Lung_two_scale_text_prompt.csv \
  --conch_ckpt_path /home/yuhaowang/project/WSIFew/cca_mil/ckg/pytorch_model.bin \
  --max_context_length 8192 \
  --window_size 8 \
  --sim_threshold 0.8 \
  --prototype_number 16
```

### FOCUS 论文设置复验

如果要检查 FOCUS 是否能复现其原论文设置下的效果，使用独立调度脚本：

```bash
cd /home/yuhaowang/project/WSIFew/cca_mil

python tools/run_focus_paper_eval.py \
  --datasets all \
  --shots 4,8,16 \
  --folds 10 \
  --max-epochs 200 \
  --gpus 4,5,6,7 \
  --max-jobs-per-gpu 4 \
  --run-name focus_paper_10fold_4_8_16
```

该脚本固定使用 `--model_type FOCUS`、10-fold split、`text_prompt/*_two_scale_text_prompt.csv`、`prototype_number=16`、`window_size=8`、`sim_threshold=0.8`，默认严格跑满 `--max_epochs 200`，不启用 early stopping。结果保存到：

```text
/data2/yuhaowang/cca-mil-result/results/FOCUS_paper_eval/<run_name>/
/data2/yuhaowang/cca-mil-result/logs/FOCUS_paper_eval/<run_name>/
```

每个任务的日志在 `<dataset>/<shot>shots/seed<seed>.log` 下，例如：

```bash
tail -f /data2/yuhaowang/cca-mil-result/logs/FOCUS_paper_eval/focus_paper_10fold_4_8_16/tcga/4shots/seed1.log
```

新版调度器会在终端每隔 `--poll-interval` 秒打印每个任务的 PID、最新 epoch/val 指标、log 更新时间和 result 状态。如果已经有相同 `exp_code` 的主进程在跑，会默认输出 `[skip-running]` 并跳过，避免同一实验重复写同一个结果目录。训练代码也会在每个 fold 的每个 epoch 后写：

```text
fold_<fold_id>_progress.csv
```

因此即使最终 `result.csv` 要等 10 个 fold 全部结束后才生成，也可以通过 `progress.csv` 和调度器状态确认训练在推进。

如果已有旧调度器在跑，可以另开一个终端只监控当前 run，不启动新任务：

```bash
python tools/run_focus_paper_eval.py \
  --datasets all \
  --shots 4,8,16 \
  --run-name focus_paper_10fold_4_8_16 \
  --monitor-only
```

先只检查路径和命令是否正确，不启动训练：

```bash
python tools/run_focus_paper_eval.py \
  --datasets all \
  --shots 4,8,16 \
  --validate-only
```

训练前可用下面命令确认 split 中所有 slide 都有 `.pt` 特征：

```bash
python tools/audit_preprocessing.py \
  --raw_dir /data2/yuhaowang/WSIFew/TCGA-NSCLC \
  --processed_dir /data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC \
  --feat_dir /data2/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature \
  --dataset_csv dataset_csv/LUAD_LUSC.csv \
  --split_dir splits/LUAD_LUSC_16shots_5folds \
  --slide_exts .svs \
  --strict
```

### 手动运行 CAMELYON

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 \
  --drop_out \
  --early_stopping \
  --early_stopping_patience 15 \
  --early_stopping_stop_epoch 0 \
  --max_epochs 80 \
  --lr 1e-4 \
  --k 5 \
  --label_frac 1 \
  --bag_loss ce \
  --task task_camelyon_subtyping \
  --csv_path /home/yuhaowang/project/WSIFew/cca_mil/dataset_csv/camelyon.csv \
  --results_dir /data2/yuhaowang/cca-mil-result/results/CCA_MIL/conch/ \
  --exp_code camelyon_16shots_5folds \
  --model_type CCA_MIL \
  --mode transformer \
  --log_data \
  --data_folder_s /data2/yuhaowang/WSIFew/processd_wsi/CAMELYON/feature/pt_files/ \
  --data_folder_l /data2/yuhaowang/WSIFew/processd_wsi/CAMELYON/feature/pt_files/ \
  --split_dir camelyon_16shots_5folds \
  --concept_bank_path text_prompt/concept_bank/camelyon.json \
  --num_visual_prototypes 10 \
  --proto_tau 0.1 \
  --ot_epsilon 0.05 \
  --sinkhorn_iter 20 \
  --uot_rho_a 0.5 \
  --uot_rho_b 0.5 \
  --concept_pooling attention \
  --lambda_contrast 0.1 \
  --lambda_div 0.01
```

### 手动运行 UBC-OCEAN

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 \
  --drop_out \
  --early_stopping \
  --early_stopping_patience 15 \
  --early_stopping_stop_epoch 0 \
  --max_epochs 80 \
  --lr 1e-4 \
  --k 5 \
  --label_frac 1 \
  --bag_loss ce \
  --task task_UBC-OCEAN_subtyping \
  --results_dir /data2/yuhaowang/cca-mil-result/results/CCA_MIL/conch/ \
  --exp_code UBC-OCEAN_16shots_5folds \
  --model_type CCA_MIL \
  --mode transformer \
  --log_data \
  --csv_path /home/yuhaowang/project/WSIFew/cca_mil/dataset_csv/UBC-OCEAN.csv \
  --data_folder_s /data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN/feature/pt_files \
  --data_folder_l /data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN/feature/pt_files \
  --split_dir UBC-OCEAN_16shots_5folds \
  --concept_bank_path text_prompt/concept_bank/ubc_ocean.json \
  --num_visual_prototypes 10 \
  --proto_tau 0.1 \
  --ot_epsilon 0.05 \
  --sinkhorn_iter 20 \
  --uot_rho_a 0.5 \
  --uot_rho_b 0.5 \
  --concept_pooling attention \
  --lambda_contrast 0.1 \
  --lambda_div 0.01
```

## 调参脚本与指标

本仓库提供统一调参调度脚本：

```text
tools/run_aot_sweep.py
```

它会为每个数据集、shot、seed、参数组合创建独立结果目录，并按指定 GPU 列表和每张卡最大任务数排队运行。默认会跳过已经完成的实验，适合中断后继续跑。这里的“完成”不是只看文件是否存在，而是检查 `result.csv` 是否可读、包含 `metric=mean` 行，并且至少有一个 validation/test 指标。

如果同一个 `--run-name` 下已经有旧版 hash 目录完成了等价参数，脚本也会默认通过 `existing_result_csv` 识别并跳过，避免仅因为新旧目录命名不同而重复训练。需要强制重跑时加 `--no-skip-existing`；只想关闭旧目录等价检测时加 `--no-skip-equivalent-existing`。

### 路径配置

TCGA-NSCLC 和 CAMELYON 默认使用当前脚本中的本机路径。UBC-OCEAN 如果不在默认位置，需要显式指定：

```bash
export CCA_MIL_UBC_CSV=/path/to/UBC-OCEAN.csv
export CCA_MIL_UBC_FEATURE_DIR=/path/to/UBC-OCEAN/feature/pt_files
```

也可以在命令行中传入：

```bash
--ubc-csv-path /path/to/UBC-OCEAN.csv \
--ubc-feature-dir /path/to/UBC-OCEAN/feature/pt_files
```

当前默认 UBC feature 路径是：

```text
/data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN/feature/pt_files
```

UBC 还需要一个带标签的训练 CSV。若你手里是 Kaggle 原始 `train.csv`，先生成仓库格式：

```bash
python tools/prepare_ubc_ocean_csv.py \
  --metadata /path/to/UBC-OCEAN/train.csv \
  --feature-dir /data2/yuhaowang/WSIFew/processd_wsi/UBC-OCEAN/feature/pt_files \
  --output dataset_csv/UBC-OCEAN.csv
```

输出格式为 `dir,case_id,slide_id,label`，并默认只保留已经提取出 `.pt` 特征的 slide。

### 快速检查

先只生成任务，不启动训练：

### 启动调参

在 4 张 GPU 上运行，每张卡最多 8 个任务：

```bash
python tools/run_aot_sweep.py \
  --datasets all \
  --shots 1,4,8,16 \
  --preset balanced \
  --gpus 4,5,6,7 \
  --max-jobs-per-gpu 4 \
  --python "$CCA_MIL_PYTHON" \
  --run-name aot_balanced_allshots
```

如果只想跑 Libra-MIL 主表对应的 shot，使用 `--shots 1,4,16`；如果要加上 8-shot 中间点，使用 `--shots 1,4,8,16`。

```bash
python tools/run_aot_sweep.py \
  --datasets all \
  --shots 1,4,8,16 \
  --preset balanced \
  --gpus 4,5,6,7 \
  --max-jobs-per-gpu 8 \
  --python "$CCA_MIL_PYTHON" \
  --run-name aot_balanced_allshots
```

后台运行示例：

```bash
mkdir -p /data2/yuhaowang/cca-mil-result/logs/AOT_MIL_sweeps
nohup python tools/run_aot_sweep.py \
  --datasets all \
  --shots 1,4,8,16 \
  --preset balanced \
  --gpus 4,5,6,7 \
  --max-jobs-per-gpu 8 \
  --python "$CCA_MIL_PYTHON" \
  --run-name aot_balanced_allshots \
  > /data2/yuhaowang/cca-mil-result/logs/AOT_MIL_sweeps/aot_balanced_allshots.launcher.log 2>&1 &
```

### 默认调参网格

`--preset balanced` 使用关键参数优先的 one-factor sweep：以默认配置为中心，每次只改变一类真正进入当前 CCA_MIL 计算图的超参，避免全因子组合爆炸和 no-op 重复实验。调度脚本会用 canonical key 做语义去重；例如当前实现中 `common_concept_weight` 只影响未使用的 concept weight 缓存，因此不会再进入默认消融。

| 维度 | 默认值 | sweep 候选 |
| --- | --- | --- |
| `lr` | `1e-4` | `2e-5`, `5e-5`, `2e-4`, `5e-4` |
| `num_visual_prototypes` | TCGA-NSCLC: `6`; CAMELYON/UBC: `10` | `8`, `32` |
| `proto_tau` | `0.1` | `0.05`, `0.2` |
| `ot_epsilon` | `0.05` | `0.03`, `0.1` |
| `sinkhorn_iter` | `20` | extended preset: `30`, `75` |
| `(uot_rho_a, uot_rho_b)` | `(0.5, 0.5)` | `(0.3,0.3)`, `(1.0,1.0)` |
| loss | `(lambda_contrast=0.1, lambda_div=0.01)` | `ce_only`, `contrast_only`, `weak_contrast`, `strong_contrast`, `strong_div` |

`--preset smoke` 只跑少量组合，用于检查数据路径和训练是否能正常启动。`--preset extended` 在 `balanced` 基础上追加 `sinkhorn_iter`、`concept_pooling`、`contrast_tau` 和非对称 `uot_rho` 等低优先级参数。`--preset wide` 在 `extended` 基础上继续扩大搜索范围，默认的 `run_cca_mil_ablation_all.sh` 当前使用该 preset。`--preset custom` 可以用 `--combo` 手动指定组合，例如：

`--preset wide` 额外增加：

| 维度 | 额外候选 |
| --- | --- |
| `lr` | `7e-4`, `1e-3` |
| `num_visual_prototypes` | `4`, `12`, `16`, `24`, `48`, `64` |
| `proto_tau` | `0.025`, `0.075`, `0.15`, `0.3` |
| `ot_epsilon` | `0.01`, `0.02`, `0.075`, `0.15`, `0.2` |
| `(uot_rho_a, uot_rho_b)` | `(0.1,0.1)`, `(0.2,0.2)`, `(2.0,2.0)`, `(0.2,1.0)`, `(1.0,0.2)` |
| loss | `tiny_contrast`, `no_contrast_keep_div`, `very_strong_contrast`, `very_strong_div`, `regularized` |
| prompt | `concept_prompt_n_ctx=1/16`, `concept_prompt_template_count=16/22` |
| patch budget | `1024`, `12288`, `16384` |
| concept logits | `concept_logit_weight=0.25/2.0`, `concept_logit_tau=0.5/2.0` |
| targeted combos | `lr1e-3_tau0.05`, `lr5e-4_tau0.05`, `ce_tau0.05`, `ctx8_tau0.05`, `proto8_tau0.05`, `concept_strong_tau0.05`, `concept_strong_lr2e-4` |

```bash
python tools/run_aot_sweep.py \
  --datasets tcga \
  --shots 16 \
  --preset custom \
  --combo num_visual_prototypes=16,proto_tau=0.1,ot_epsilon=0.05,lambda_contrast=0.1,lambda_div=0.01 \
  --gpus 4 \
  --run-name aot_custom_tcga
```

### 输出目录

每个参数组合都会保存到可读的层级目录，`axis` 表示消融维度，`setting` 表示该维度的取值：

```text
/data2/yuhaowang/cca-mil-result/results/AOT_MIL_sweeps/<run_name>/<dataset>/<shots>shots/<axis>/<setting>/<exp_code>_s<seed>/
  experiment_<exp_code>.txt
  summary.csv
  result.csv
  s_<fold>_checkpoint.pt
```

示例：

```text
.../tcga/16shots/lr/5e-5/LUAD_LUSC_16shots_balanced_lr_5e-5_s1/
.../tcga/16shots/loss/ce_only/LUAD_LUSC_16shots_balanced_loss_ce_only_s1/
```

调度脚本自身会写：

```text
/data2/yuhaowang/cca-mil-result/logs/AOT_MIL_sweeps/<run_name>/
  jobs.csv              # 所有任务和参数
  commands.sh           # 可复现命令
  sweep_summary.csv     # 汇总排序表
  <dataset>/<shots>shots/<axis>/<setting>_seed<seed>.log
```

`jobs.csv` 中会额外保存 `axis`、`setting`、`param_id`、`param_summary` 和 `canonical_key`，用于确认没有语义重复的实验。

### 调参指标

调参主指标使用 `val_auc_mean`，也就是 5-fold validation AUC 的平均值。不要用 test 指标选择超参；test 只用于最终报告。

推荐排序规则：

| 优先级 | 指标 | 说明 |
| --- | --- | --- |
| 1 | `val_auc_mean` | 主调参指标，越高越好。 |
| 2 | `val_f1_mean` | AUC 接近时看 macro-F1，尤其适合类别不均衡数据。 |
| 3 | `val_auc_std` / `val_f1_std` | 均值接近时优先选择跨 fold 更稳定的配置。 |
| 4 | `test_auc_mean`, `test_f1_mean`, `test_acc_mean` | 只在确定超参后用于最终报告。 |

`main.py` 会在每个实验目录中保存：

- `summary.csv`：每个 fold 的 `val_auc`、`val_acc`、`val_f1`、`test_auc`、`test_acc`、`test_f1`。
- `result.csv`：上述指标的 mean / std。
- `sweep_summary.csv`：调度脚本汇总所有 `result.csv`，默认按 `val_auc_mean` 降序排列。

如果训练已经完成，只重新汇总结果：

```bash
python tools/run_aot_sweep.py \
  --datasets all \
  --shots 1,4,8,16 \
  --preset balanced \
  --run-name aot_balanced_allshots \
  --collect-only
```

把所有 run、所有 dataset、所有 shot 的 `result.csv` 汇总为一个消融总表：

```bash
python tools/collect_ablation_results.py \
  --root /data2/yuhaowang/cca-mil-result/results \
  --output /data2/yuhaowang/cca-mil-result/results/ablation_results_all.csv \
  --print-top 20
```

该 CSV 会包含 `dataset`、`shots`、`axis`、`setting`、`param_id`、所有关键超参，以及 `val_auc_mean/std`、`val_f1_mean/std`、`test_auc_mean/std` 等指标。

## 关键参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--model_type` | `CCA_MIL` | 当前默认模型；可设为 `FOCUS` 跑原始 FOCUS。 |
| `--text_prompt_path` | `None` | `FOCUS` / `ViLa_MIL` 必需的文本 prompt CSV。 |
| `--conch_ckpt_path` | `ckg/pytorch_model.bin` | `FOCUS` / `CCA_MIL` 使用的 CONCH checkpoint。 |
| `--max_context_length` | `8192` | FOCUS token compression 后的最大 token 数。 |
| `--window_size` | `8` | FOCUS 局部相似度压缩窗口大小。 |
| `--sim_threshold` | `0.8` | FOCUS spatial token compression 相似度阈值。 |
| `--concept_bank_path` | 自动按 task 选择 | concept bank JSON 路径。 |
| `--max_epochs` | `80` | Libra-MIL 风格实验默认最大训练 epoch。 |
| `--k` | `5` | 默认 5-fold cross-validation。 |
| `--early_stopping_patience` | `15` | early stopping patience。 |
| `--num_visual_prototypes` | TCGA-NSCLC: `6`; CAMELYON/UBC 脚本: `10` | 可学习视觉 prototype 数量；与 concept 数量不需要一致。 |
| `--proto_tau` | `0.1` | patch-to-prototype soft assignment temperature。 |
| `--ot_epsilon` | `0.05` | entropy-regularized OT 的平滑系数。 |
| `--sinkhorn_iter` | `20` | unbalanced Sinkhorn 迭代次数。 |
| `--uot_rho_a` | `0.5` | visual evidence side 的 unbalanced penalty。 |
| `--uot_rho_b` | `0.5` | concept side 的 unbalanced penalty。 |
| `--concept_pooling` | `attention` | `Z_dis` pooling 方式：`attention` / `mean` / `learnable`。 |
| `--common_concept_weight` | `0.3` | common concepts 的聚合权重。 |
| `--lambda_contrast` | `0.1` | class-aware discriminative contrastive loss 权重。 |
| `--lambda_div` | `0.01` | GT-class discriminative evidence diversity loss 权重。 |
| `--contrast_tau` | `0.07` | contrastive loss temperature。 |
| `--lambda_con` / `--tau` | legacy | 分别作为 `--lambda_contrast` / `--contrast_tau` 的兼容别名。 |
| `--train_concept_prompt` | off | 是否训练 concept prompt context。 |
| `--concept_prompt_n_ctx` | `0` | CCA-MIL concept prompt 的可训练 context token 数；默认使用 pathology template ensemble，不插入随机 context。 |
| `--store_explanations` | off | 是否保存最近一次 forward 的解释信息。 |

## 输出文件

训练结果会保存到：

```text
/data2/yuhaowang/cca-mil-result/results/CCA_MIL/conch/<exp_code>_s<seed>/
  experiment_<exp_code>.txt
  splits_0.csv
  s_0_checkpoint.pt
  split_0_results.pkl
  ...
  summary.csv
  result.csv
```

`main.py` 会把相对 `--results_dir` 自动重定向到 `/data2/yuhaowang/cca-mil-result/results` 下。例如旧命令中的 `--results_dir results/CCA_MIL/conch/` 会实际写入 `/data2/yuhaowang/cca-mil-result/results/CCA_MIL/conch/`，避免 checkpoint 写满项目所在的 `/` 分区。

其中：

- `summary.csv`：每个 fold 的 validation/test AUC、ACC、F1。
- `result.csv`：所有 fold 的 mean / std；调参时优先看 `val_auc` 的 mean。
- `s_<fold>_checkpoint.pt`：对应 fold 的模型 checkpoint。
- `experiment_<exp_code>.txt`：本次实验参数记录。

## 解释性输出

开启 `--store_explanations` 后，模型会在 `model.last_explanations` 中保留最近一次 forward 的解释信息：

- `patch_proto_assign`：patch 到视觉 prototype 的 soft assignment。
- `transport` / `P_dis` / `P_com`：视觉 evidence token 到 concept bank 的 OT transport。
- `patch_dis_score` / `patch_com_score`：由 prototype-level 分数回传到 patch 维度的解释分数。
- `concept_dis_score` / `concept_com_score`：concept-level evidence score。
- `Z_dis` / `Z_com`：OT 聚合后的 discriminative/common visual evidence。

这些信息可以用于构建 `patch -> visual prototype -> concept -> class` 的可解释证据链。
