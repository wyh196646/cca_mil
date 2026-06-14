# CCA-MIL

**Concept-Cluster Alignment for Few-Shot Whole Slide Image Classification**

本仓库是在 [dddavid4real/FOCUS](https://github.com/dddavid4real/focus) 代码结构上改造的少样本 WSI 分类项目。数据集、WSI 预处理、CONCH 特征、few-shot split 和训练脚本约定基本沿用 FOCUS；当前默认模型从原始 FOCUS 改为 `CCA_MIL`，核心思想是把病理概念与单张 WSI 内部的局部组织聚类进行对齐，形成 `concept -> cluster -> patch -> class` 的可解释证据链。

FOCUS 原论文为 **FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification, CVPR 2025**。原仓库说明中使用 TCGA-NSCLC、CAMELYON 和 UBC-OCEAN 三个数据集，WSI patch size 设为 `512`，放大倍率为 `40X`，patch feature extractor 使用 `CONCH`。

## 目录

- [项目关系](#项目关系)
- [方法概览](#方法概览)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [WSI 预处理](#wsi-预处理)
- [Few-shot Splits](#few-shot-splits)
- [Concept Bank](#concept-bank)
- [训练](#训练)
- [关键参数](#关键参数)
- [输出文件](#输出文件)
- [代码结构](#代码结构)
- [常见问题](#常见问题)
- [引用](#引用)

## 项目关系

本项目继承并保留了 FOCUS / ViLa-MIL / CLAM 风格的数据接口：

- 每张 WSI 对应一个 `.pt` feature file。
- `main.py` 通过 `Generic_MIL_Dataset` 读取 slide-level 标签和 split。
- `splits/` 中提供 4-shot、8-shot、16-shot 的 10-fold few-shot split。
- `LUAD_LUSC.sh`、`camelyon.sh`、`UBC-OCEAN.sh` 是三个数据集的训练入口。

与 FOCUS 的主要区别：

- 当前默认 `--model_type CCA_MIL`。
- 原始 `models/model_FOCUS.py` 已恢复，可通过 `--model_type FOCUS` 复现实验并与 `CCA_MIL` 对比。
- CCA-MIL 不再使用整段类别文本 prompt 直接指导所有 patch，而是使用结构化 `concept_bank`。
- CCA-MIL 的 forward 中只使用高分辨率特征 `x_l`，但当前 DataLoader 仍会读取 `data_folder_s` 和 `data_folder_l`。如果没有低分辨率特征，可以把 `--data_folder_s` 和 `--data_folder_l` 指向同一个 CONCH `pt_files` 目录。

## 方法概览

整体流程：

```text
Concept bank
  -> WSI patch features
  -> slide-level KMeans
  -> concept-cluster alignment
  -> concept-aware patch selection
  -> concept-guided aggregation
  -> slide-level classification
```

核心模块：

- `Concept bank`：每个类别维护 `common_concepts` 和 `discriminative_concepts`。
- `WSI-level clustering`：对单张 WSI 的 patch features 做 KMeans，得到局部组织 clusters。
- `Concept-cluster alignment`：用 cluster center 与 concept embedding 的余弦相似度建立语义对应。
- `Concept-aware patch selection`：在每个 cluster 内选择同时接近 assigned concept 和 cluster center 的 patches。
- `Concept-guided aggregation`：以 concepts 为 query，对 class-specific selected patches 做 cross-attention 聚合。
- `Training losses`：包含 slide-level CE、concept-class contrastive loss、concept evidence diversity loss。

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
ckpts/conch.pth
```

当前 `models/cca_mil.py` 中的 text encoder 会从这个硬编码路径读取 checkpoint。可以从 FOCUS 提供的 HuggingFace 资源下载 `conch.pth`，放到 `ckpts/` 下。如果你的 CONCH 安装使用其他 checkpoint 格式或路径，需要同步修改 `models/cca_mil.py` 中的 `conch_checkpoint_path`。

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
  --source /data/yuhaowang/WSIFew/TCGA-NSCLC \
  --save_dir /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC \
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

CAMELYON 和 UBC-OCEAN 可以使用同样流程。如果没有合适的 preset，可以先去掉 `--preset tcga.csv`，再根据 `masks/` 和 `stitches/` 检查结果调整分割参数。

```bash
python create_patches_fp.py \
  --source /data/yuhaowang/WSIFew/camelyon \
  --save_dir /data/yuhaowang/WSIFew/processd_wsi/camelyon_40x_512 \
  --patch_size 512 \
  --patch_level 0 \
  --seg \
  --patch \
  --stitch
```



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
  --data_h5_dir /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC \
  --data_slide_dir /data/yuhaowang/WSIFew/TCGA-NSCLC \
  --csv_path /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/process_list_autogen.csv \
  --feat_dir /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature \
  --slide_ext .svs \
  --model_name conch_v1 \
  --conch_ckpt_path /home/yuhaowang/project/WSIFew/cca_mil/ckg/pytorch_model.bin \
  --target_patch_size 448 \
  --batch_size 1024 \
  --gpus 2,3,4,5,6,7 \
  --num_workers 12
```

也可以使用 CLAM 原版单 GPU 脚本：

```bash
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
  --data_h5_dir /data/yuhaowang/WSIFew/processd_wsi/tcga_nsclc_40x_512 \
  --data_slide_dir /data/yuhaowang/WSIFew/tcga_nsclc \
  --csv_path /data/yuhaowang/WSIFew/processd_wsi/tcga_nsclc_40x_512/process_list_autogen.csv \
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

本仓库已经提供三个数据集的 4-shot、8-shot、16-shot、10-fold split：

```text
splits/
  LUAD_LUSC_4shots_10folds/
  LUAD_LUSC_8shots_10folds/
  LUAD_LUSC_16shots_10folds/
  camelyon_4shots_10folds/
  camelyon_8shots_10folds/
  camelyon_16shots_10folds/
  UBC-OCEAN_4shots_10folds/
  UBC-OCEAN_8shots_10folds/
  UBC-OCEAN_16shots_10folds/
```

每个 split 文件格式：

```text
splits_0.csv
splits_1.csv
...
splits_9.csv
```

CSV 列为：

```csv
train,val,test
slide_001,slide_021,slide_101
slide_002,slide_022,slide_102
```

注意：

- `--k 10` 表示跑 10 个 fold。
- `--split_dir LUAD_LUSC_16shots_10folds` 会自动解析为 `splits/LUAD_LUSC_16shots_10folds`。
- split 中的 slide id 必须能在 dataset CSV 和 feature `pt_files` 中找到。
- 如果重新生成 split，请保持列名 `train,val,test` 和 slide id 命名一致。

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
ckpts/conch.pth
text_prompt/concept_bank/*.json
splits/<split_dir>/splits_0.csv ... splits_9.csv
/path/to/features/<dataset>_conch_40x_512/pt_files/{slide_id}.pt
```

### 直接运行脚本

先编辑脚本中的数据路径：

```bash
--data_folder_s 'path/to/your/low-resolution/feature'
--data_folder_l 'path/to/your/high-resolution/feature'
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
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 \
  --drop_out \
  --early_stopping \
  --lr 1e-4 \
  --k 10 \
  --label_frac 1 \
  --bag_loss ce \
  --task task_tcga_lung_subtyping \
  --results_dir results/CCA_MIL/conch/ \
  --exp_code LUAD_LUSC_16shots_10folds \
  --model_type CCA_MIL \
  --mode transformer \
  --log_data \
  --data_folder_s /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files/ \
  --data_folder_l /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files/ \
  --split_dir LUAD_LUSC_16shots_10folds \
  --concept_bank_path text_prompt/concept_bank/tcga_nsclc.json \
  --cluster_k 8 \
  --selection_top_r 3 \
  --concept_alpha 0.5 \
  --lambda_con 0.1 \
  --lambda_div 0.01 \
  --prototype_number 16
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
features    = /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files
split_dir   = splits/LUAD_LUSC_16shots_10folds
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
  --lr 1e-4 \
  --k 10 \
  --label_frac 1 \
  --bag_loss ce \
  --task task_tcga_lung_subtyping \
  --csv_path /home/yuhaowang/project/WSIFew/cca_mil/dataset_csv/LUAD_LUSC.csv \
  --results_dir results/FOCUS/conch/ \
  --exp_code LUAD_LUSC_16shots_10folds \
  --model_type FOCUS \
  --mode transformer \
  --log_data \
  --data_root_dir /data/yuhaowang/WSIFew \
  --data_folder_s /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files \
  --data_folder_l /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature/pt_files \
  --split_dir LUAD_LUSC_16shots_10folds \
  --text_prompt_path text_prompt/TCGA_Lung_two_scale_text_prompt.csv \
  --conch_ckpt_path /home/yuhaowang/project/WSIFew/cca_mil/ckg/pytorch_model.bin \
  --max_context_length 8192 \
  --window_size 8 \
  --sim_threshold 0.8 \
  --prototype_number 16
```

训练前可用下面命令确认 split 中所有 slide 都有 `.pt` 特征：

```bash
python tools/audit_preprocessing.py \
  --raw_dir /data/yuhaowang/WSIFew/TCGA-NSCLC \
  --processed_dir /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC \
  --feat_dir /data/yuhaowang/WSIFew/processd_wsi/TCGA-NSCLC/feature \
  --dataset_csv dataset_csv/LUAD_LUSC.csv \
  --split_dir splits/LUAD_LUSC_16shots_10folds \
  --slide_exts .svs \
  --strict
```

### 手动运行 CAMELYON

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 \
  --drop_out \
  --early_stopping \
  --lr 1e-4 \
  --k 10 \
  --label_frac 1 \
  --bag_loss ce \
  --task task_camelyon_subtyping \
  --csv_path /home/yuhaowang/project/WSIFew/cca_mil/dataset_csv/camelyon.csv \
  --results_dir results/CCA_MIL/conch/ \
  --exp_code camelyon_16shots_10folds \
  --model_type CCA_MIL \
  --mode transformer \
  --log_data \
  --data_folder_s /data/yuhaowang/WSIFew/processd_wsi/CAMELYON/feature/pt_files/ \
  --data_folder_l /data/yuhaowang/WSIFew/processd_wsi/CAMELYON/feature/pt_files/ \
  --split_dir camelyon_16shots_10folds \
  --concept_bank_path text_prompt/concept_bank/camelyon.json \
  --cluster_k 8 \
  --selection_top_r 3 \
  --concept_alpha 0.5 \
  --lambda_con 0.1 \
  --lambda_div 0.01 \
  --prototype_number 16
```

### 手动运行 UBC-OCEAN

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 1 \
  --drop_out \
  --early_stopping \
  --lr 1e-4 \
  --k 10 \
  --label_frac 1 \
  --bag_loss ce \
  --task task_UBC-OCEAN_subtyping \
  --results_dir results/CCA_MIL/conch/ \
  --exp_code UBC-OCEAN_16shots_10folds \
  --model_type CCA_MIL \
  --mode transformer \
  --log_data \
  --data_folder_s /path/to/DATA_ROOT/features/ubc_ocean_conch_40x_512/pt_files \
  --data_folder_l /path/to/DATA_ROOT/features/ubc_ocean_conch_40x_512/pt_files \
  --split_dir UBC-OCEAN_16shots_10folds \
  --concept_bank_path text_prompt/concept_bank/ubc_ocean.json \
  --cluster_k 8 \
  --selection_top_r 3 \
  --concept_alpha 0.5 \
  --lambda_con 0.1 \
  --lambda_div 0.01 \
  --prototype_number 16
```

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
| `--cluster_k` | `8` | 每张 WSI 内部 KMeans 聚类数。 |
| `--kmeans_iters` | `10` | KMeans 迭代次数。 |
| `--min_cluster_size` | `5` | 小 cluster 合并阈值。 |
| `--selection_top_r` | `3` | 每个 cluster 保留的 patch 数。 |
| `--concept_alpha` | `0.5` | concept relevance 与 cluster representativeness 的融合权重。 |
| `--common_concept_weight` | `0.3` | common concepts 的聚合权重。 |
| `--lambda_con` | `0.1` | concept-class contrastive loss 权重。 |
| `--lambda_div` | `0.01` | concept evidence diversity loss 权重。 |
| `--tau` | `0.07` | contrastive loss temperature。 |
| `--no_normalize_kmeans` | off | 关闭 KMeans 前的 L2 normalization。 |
| `--train_concept_prompt` | off | 是否训练 concept prompt context。 |
| `--store_explanations` | off | 是否保存最近一次 forward 的解释信息。 |

## 输出文件

训练结果会保存到：

```text
results/CCA_MIL/conch/<exp_code>_s<seed>/
  experiment_<exp_code>.txt
  splits_0.csv
  s_0_checkpoint.pt
  split_0_results.pkl
  ...
  summary.csv
  result.csv
```

其中：

- `summary.csv`：每个 fold 的 test AUC、ACC、F1。
- `result.csv`：所有 fold 的 mean / std。
- `s_<fold>_checkpoint.pt`：对应 fold 的模型 checkpoint。
- `experiment_<exp_code>.txt`：本次实验参数记录。

## 解释性输出

开启 `--store_explanations` 后，模型会在 `model.last_explanations` 中保留最近一次 forward 的解释信息：

- `alignment`：当前候选类别的 concept-cluster alignment matrix。
- `concept_evidence`：每个 concept 的最大 cluster evidence。
- `selected_indices`：被选择的 patch index。
- `selection_scores`：patch selection score。
- `assigned_concepts`：每个 cluster 对齐到的 concept index。

这些信息可以用于构建 `concept -> cluster -> patch -> class` 的可解释证据链。

## 代码结构

```text
main.py                                   # 训练入口
fast_create_patches_fp.py                 # 多进程 WSI segmentation / patch extraction
fast_extract_features_fp.py               # 多 GPU feature extraction
LUAD_LUSC.sh                             # TCGA-NSCLC 训练脚本
run_focus_tcga_nsclc.sh                  # TCGA-NSCLC 原始 FOCUS 训练脚本
camelyon.sh                              # CAMELYON 训练脚本
UBC-OCEAN.sh                             # UBC-OCEAN 训练脚本
datasets/dataset_generic.py              # WSI feature dataset 和 split 读取
models/cca_mil.py                        # CCA-MIL 主模型
models/model_FOCUS.py                    # 原始 FOCUS 模型
models/concept_guided_aggregator.py      # concept-guided cross-attention 聚合器
models/model_ViLa_MIL.py                 # 保留的 ViLa-MIL 模型
utils/core_utils.py                      # 训练、验证、测试流程
utils/concept_loader.py                  # concept bank 读取工具
text_prompt/concept_bank/*.json          # 数据集对应的 concept bank
splits/*/splits_*.csv                    # few-shot 10-fold splits
wsi_core/                                # 从 CLAM 继承的 WSI 工具代码
```

# cca_mil
