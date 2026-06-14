# CAMELYON split mapping

FOCUS-style CAMELYON splits mix two naming schemes:

- `patient_XXX_node_Y`: CAMELYON17 official slide ids.
- `slide_N`: CAMELYON16 slides after FOCUS-style renaming.

For the local CAMELYON16 copy in `/data/yuhaowang/WSIFew/CAMELYON16`, the
`slide_N` ids are ordered as:

1. `normal_*.tif` in numeric order
2. `tumor_*.tif` in numeric order
3. `test_*.tif` in numeric order

The generated mapping is:

```bash
python tools/build_camelyon16_focus_mapping.py --query slide_247 slide_222
```

Examples:

- `slide_222 -> tumor_063.tif`
- `slide_247 -> tumor_088.tif`

The script writes:

```text
splits/camelyon16_focus_slide_mapping.csv
```

This CSV has `slide_id` for FOCUS-compatible output names and `source_path` for
the original CAMELYON16 WSI path. It can be used directly as a process list.

Patch extraction example:

```bash
python fast_create_patches_fp.py \
  --source /data/yuhaowang/WSIFew/CAMELYON16 \
  --save_dir /data/yuhaowang/WSIFew/processd_wsi/CAMELYON16_FOCUS \
  --process_list /home/yuhaowang/project/WSIFew/cca_mil/splits/camelyon16_focus_slide_mapping.csv \
  --slide_ext .tif \
  --patch --seg --stitch \
  --patch_size 512 \
  --step_size 512 \
  --num_workers 8
```

Feature extraction example:

```bash
python fast_extract_features_fp.py \
  --data_h5_dir /data/yuhaowang/WSIFew/processd_wsi/CAMELYON16_FOCUS \
  --data_slide_dir /data/yuhaowang/WSIFew/CAMELYON16 \
  --csv_path /home/yuhaowang/project/WSIFew/cca_mil/splits/camelyon16_focus_slide_mapping.csv \
  --feat_dir /data/yuhaowang/WSIFew/processd_wsi/CAMELYON16_FOCUS/feature \
  --slide_ext .tif \
  --model_name conch_v1 \
  --conch_ckpt_path /home/yuhaowang/project/WSIFew/cca_mil/ckg/pytorch_model.bin \
  --target_patch_size 448 \
  --batch_size 256 \
  --gpus 2,3,4,5,6,7 \
  --num_workers 6
```

For training, pass the local dataset table explicitly:

```bash
python main.py \
  --task task_camelyon_subtyping \
  --csv_path /home/yuhaowang/project/PathARK/data/dataset_csv/camelyon.csv \
  --split_dir camelyon_4shots_10folds \
  ...
```

If the original CAMELYON16 files are renamed or restored to a different count,
regenerate the mapping and check that all labels match the FOCUS table.
