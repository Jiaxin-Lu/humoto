# HUMOTO Dataset Code Release

[[Paper](https://arxiv.org/abs/2504.10414)] [[Project Page](https://jiaxin-lu.github.io/humoto/)] [[Public Dataset Page](https://adobe-research.github.io/humoto/)] [[Public Dataset Release](https://github.com/adobe-research/humoto)]

![HUMOTO_Overview](./docs/humoto_overview.png)

## Overview

**HUMOTO** is a large-scale 4D mocap dataset capturing high-quality human-object interactions with synchronized motion capture of both humans and objects. The dataset provides:

- **Rich motion sequences** with diverse human-object interaction scenarios
- **High-quality skeletal animations** using standard Adobe Mixamo character models
- **Object meshes and metadata** for all interacted objects
- **Flexible character and bone layout options** for various use cases
- **PyTorch-compatible tools** for differentiable forward kinematics and batched operations

This repository provides comprehensive tools for:
- **Processing and converting** dataset files into user-friendly formats
- **Loading and manipulating** motion sequences efficiently
- **Rendering and visualizing** human-object interactions
- **Differentiable human modeling** for optimization and learning tasks

The code is compatible with both the [publicly released version](https://adobe-research.github.io/humoto/) and the full version of the dataset. To access the full version, please contact Yi Zhou following the [instructions](https://github.com/adobe-research/humoto?tab=readme-ov-file#accessing-the-full-dataset).

## Dataset Structure

The full dataset is organized as follows:
```
humoto
├── humoto_0805
│   ├── <sequence_name>
│   │   ├── <sequence_name>.fbx
│   │   │── <sequence_name>.glb
│   │   └── <sequence_name>.yaml
│   └── ...
└── humoto_objects_0805
│   ├── <object_name>
│   │   ├── <object_name>.fbx
│   │   ├── <object_name>.glb
│   │   ├── <object_name>.obj
│   │   └── <object_name>.yaml
├── └── ...
```
The `humoto_0805` folders contain the motion sequences, while the `humoto_objects_0805` folder contains object meshes and metadata.

We recommend using the following steps to convert the dataset to a format that is easier to use.

## Installation

Python 3.10 is required for bpy version 4.0.0. If you need to use a different Python version, refer to the official bpy documentation at https://download.blender.org/pypi/bpy/.

```bash
conda create -n humoto python=3.10
conda activate humoto
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Follow the instructions in the [PyTorch3D documentation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to install PyTorch3D.

## Human Model

The human model is a standard Adobe Mixamo character model located in the `human_model` folder.

We provide four character variations:
- `with_texture`: Human model with texture maps.
- `without_texture`: Human model without texture maps.
- `Remy`: A male Mixamo character adapted to match the base model's dimensions.
- `Sophie`: A female Mixamo character adapted to match the base model's dimensions.

For each character, we provide two bone layout variants:
- `mixamo_bone`: Standard Mixamo bone layout where each bone points toward its child bone.
- `up_bone`: Same bone hierarchy as Mixamo, but with all bones oriented upward instead of pointing toward children.

We also provide a PyTorch model `HumanModelDifferentiable` in `human_model.py` for differentiable forward kinematics and batched operations. This model should be used with the corresponding JSON files in the `human_model` folder. Select the appropriate JSON file based on your coordinate system (y-up or z-up). The model is compatible with Blender's pose parameters and mimics the behavior of Blender's armature system. 

**Input:** Either a dictionary of `{bone_name: tensor [batch_size, 4, 4]}` (4x4 transformation matrices) or a single tensor `[batch_size, num_joints, 4, 4]` (transformation matrices in predefined bone order).

**Output:** Deformed vertices tensor `[batch_size, num_vertices, 3]` and joint positions `{bone_name: tensor [batch_size, 3]}`.

## Data Conversion Tools

We provide tools to convert the dataset to a more user-friendly format in `script` folder. Follow these steps in order:

### Step 1: Clear Human Scale

Removes the root scale from the armature and its child meshes to normalize the model.

```bash
bash clear_human_scale.sh -d <path_to_humoto_0805> -o <path_to_a_new_scale_cleared_humoto_folder>
```

### Step 2: Transfer Human Model

Converts the human model to a different character or bone layout. The `up_bone` version is recommended if you want to use the dataset for machine learning tasks.

```bash
bash transfer_human_model.sh \
  -d <path_to_scale_cleared_humoto_0805> \
  -m ../human_model/human_model_without_texture_up_bone.fbx \
  -o <path_to_a_new_up_bone_humoto_folder>
```

**Note:** This script also converts between different character models (e.g., `without_texture` to `Remy`) and bone layouts (e.g., `mixamo_bone` to `up_bone`). It only works between models with identical bone lengths and rest positions; using it with incompatible models will corrupt the animation data.

### Step 3: Extract Pickle Data

Extracts skeleton and object pose data from FBX files into pickle format for easier loading.

```bash
bash extract_pk_data.sh -d <path_to_up_bone_humoto_0805> ( -o <output_folder> ) ( -m )
```

- If `-o` is not specified, pickle files will be saved in the same folder as the source FBX files.
- The `-m` flag extracts object meshes into pickle files. This is not recommended if you have access to the `humoto_objects_0805` folder, as it significantly increases file size and object meshes can be loaded more efficiently from the centralized object folder.

### Step 4: Copy Text Metadata

Copies YAML metadata files from the original dataset to the converted dataset folder for better organization.

```bash
python copy_text.py -d <original_dataset_path> -o <target_dataset_path>
```

**Note:** The bash scripts in Steps 1-3 automatically process all sequences in the dataset directory. You can also run the corresponding Python scripts directly to process individual sequences.

We recommend create new intermediate folders to store the converted dataset for easier management.

## Using the Dataset

We provide utilities to load the dataset in a format suitable for batched operations.

### Environment Variables

Before proceeding, we recommend setting the following environment variables so the programs can automatically locate the dataset:

```bash
export HUMOTO_DATASET_DIR=/path/to/humoto_0805_up_bone
export HUMOTO_OBJECT_DIR=/path/to/humoto_objects_0805
```

- `HUMOTO_DATASET_DIR`: Used by default dataset directory
- `HUMOTO_OBJECT_DIR`: Used by loading utilities as the default path to the object models folder

These environment variables are optional; you can always specify paths explicitly using command-line arguments.

### Example: Rendering Sequences

As a usage example, we include `render_humoto_pytorch3d.py`, which demonstrates how to render sequences using PyTorch3D and the provided helper functions.

**Usage:**
```bash
python render_humoto_pytorch3d.py -d <path_to_sequence> -o <output_folder> -m <path_to_object_model> -y -t -u
```

**Flags:**
- `-d`: Path to the sequence folder (or the name of the sequence if HUMOTO_DATASET_DIR is set) containing the pickle (and yaml if include_text is True) file (required)
- `-o`: Output folder for the rendered video (defaults to sequence folder if not specified)
- `-m`: Path to the object model directory (defaults to `HUMOTO_OBJECT_DIR` environment variable)
- `-y`: Render in y-up coordinate system (omit for z-up)
- `-t`: Include text metadata overlay in the rendered video
- `-u`: Use the `up_bone` version of the human model (omit for `mixamo_bone`). This must match the human model you used for extracting the pickle files.
- `-b`: Batch size for rendering (default: 50)

This example demonstrates how to handle different coordinate systems and data formats. You can use the same loading functions for other operations on the dataset.

## Citation

If you use this code or dataset in your research, please cite:
```
@InProceedings{Lu_2025_HUMOTO,
    author    = {Lu, Jiaxin and Huang, Chun-Hao Paul and Bhattacharya, Uttaran and Huang, Qixing and Zhou, Yi},
    title     = {HUMOTO: A 4D Dataset of Mocap Human Object Interactions},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {10886-10897}
}
```
