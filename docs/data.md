## Prepare data

#### Download

Please download the annotation of the final mixture our data [Stage1.json](https://huggingface.co/datasets/starriver030515/FUSION-Pretrain-10M/tree/main/Stage1), [Stage1.5.json](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main/Stage1.5-json), [Stage2.json](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main/Stage2-json), and download the images from [FUSION-Pretrain-10M](https://huggingface.co/datasets/starriver030515/FUSION-Pretrain-10M/tree/main), [FUSION-Finetune-12M](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main).

After downloading all of them, organize the data in `playground/train/jsons` and `playground/train/images` as follows:

```
├── images
│   ├── LLaVA-Pretrain
│   ├── PixelProse
│   ├── ShareCaptioner
│   ├── SynthCount
│   ├── SynthScene
│   ├── SynthSpatial
│   ├── SynthTextVQA
│   └── URSA_Alignment
└── jsons
    ├── LLaVA-Pretrain-558k.json
    ├── PixelProse-5500k.json
    ├── ShareCaptioner-1246k.json
    ├── SynthColor-244k.json
    ├── SynthCount-300k.json
    ├── SynthScene-268k.json
    ├── SynthSpatial-300k.json
    ├── SynthTextVQA-400k.json
    └── URSA_Alignment-860k.json
```

#### Format

Inspired by LLaVA-OneVision, we use YAML files to configure our training datasets. A sample YAML configuration is shown below:

```
datasets:
  - json_path: ../../playground/train/jsons/LLaVA-Pretrain-558k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/ShareCaptioner-1246k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/URSA_Alignment-860k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthColor-244k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthCount-300k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthScene-268k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthSpatial-300k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthTextVQA-400k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/PixelProse-5500k.json
    sampling_strategy: "all"
```

You can adjust the sampling_strategy for each dataset to control the size and proportion of the data used during training. For details on how sampling works, please refer to the implementation in [fusion/train/train.py](https://github.com/starriver030515/FUSION/blob/main/fusion/train/train.py).

Our official training YAML files for FUSION are located in [scripts/train/yaml](https://github.com/starriver030515/FUSION/tree/main/scripts/train/yaml). You can modify these files to suit your training needs.