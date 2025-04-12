<div align="center">


#  <img src="images/ai.png" alt="图标" width="30" height="auto">*FUSION*:<br> Fully Integration of Vision-Language Representations for Deep Cross-Modal Understanding

</div>

<div align="center">
<a href="https://arxiv.org/" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-FUSION-red?logo=arxiv" height="25" />
</a>
<a href="https://github.com/starriver030515/FUSION/tree/main" target="_blank">
    <img alt="Github Star" src="https://img.shields.io/github/stars/starriver030515/FUSION?style=social" height="25" />
</a>
<a href="https://huggingface.co/collections/starriver030515/fusion-model-67f7cd1b9ff2f360fcfe00f1" target="_blank">
    <img alt="HF Model: FUSION" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-FUSION-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/collections/starriver030515/fusion-data-67f7ccd0b087b0aa5995cbbe" target="_blank">
    <img alt="HF Dataset: FUSION-12M" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-FUSION--12M-ffc107?color=ffc107&logoColor=white" height="25" />
</a>


[Zheng Liu]()<sup>1,2</sup>, 
[Mengjie Liu]()<sup>1,2</sup>,
[Jingzhou Chen]()<sup>1,2</sup>,  <br>
[Bin Cui](https://cuibinpku.github.io/index.html)<sup>1</sup>, 
[Conghui He](https://conghui.github.io/)<sup>2†</sup>, 
[Wentao Zhang](https://zwt233.github.io)<sup>1†</sup>

<sup>1</sup>Peking University, <sup>2</sup>Shanghai Artificial Intelligence Laboratory<br>


<p>
    <img src="images/model_comparision.jpg" alt="FUSION" width="500" height="auto">
</p>


<br>

> *With only 630 vision tokens, FUSION-X outperforms Cambrian-1 and Florence-VL, matching LLaVA-OneVision and nearly reaching the performance of top models like InternVL2 and Qwen2VL. Even with 300 vision tokens, FUSION-L retains 95% of its original performance, staying on par with Florence-VL.*
</div>

## Release

- [04/12/25] 🤗 We released all three stages of the FUSION and FUSION-X models in 3B and 8B sizes. Detailed information can be found in the [Model Zoo](https://github.com/starriver030515/FUSION/blob/main/docs/model_zoo.md).
- [04/11/25] 🚀  We have released the training and evaluation code for FUSION. For training, we provide scripts for FUSION and FUSION-X in 3B and 8B sizes, available in the [train/](https://github.com/starriver030515/FUSION/tree/main/scripts/train) folder. For evaluation, we have adapted lmms-eval and provided the corresponding model files, available in the [lmms-eval/](https://github.com/starriver030515/FUSION/tree/main/scripts/lmms-eval) folder. Additionally, we have implemented benchmark evaluation code in LLaVA format for quick evaluation, found in the [eval/](https://github.com/starriver030515/FUSION/tree/main/scripts/eval) folder.
- [04/07/25] 🤗 We released the FUSION-Pretrain-10M and FUSION-Finetune-12M datasets. We summarized existing benchmarks and used our dataengine to synthesize additional data. Please see [here](https://huggingface.co/collections/starriver030515/fusion-data-67f7ccd0b087b0aa5995cbbe) for more information.

## Contents

- [Installation](#installation)
- [FUSION Weights](#fusion-weights)
- [Demo](#demo)
- [FUSION Dataset](#fusion-dataset)
- [Train](#train)
- [Evaluation](#evaluation)

## Installation

1. Clone this repository and navigate to FUSION folder

```bash
git clone https://github.com/starriver030515/FUSION.git
cd FUSION
```

2. Install Package

```Shell
conda create -n fusion python=3.10 -y
conda activate fusion
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases

```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

## FUSION Weights

Please check out our [Model Zoo](https://github.com/starriver030515/FUSION/blob/main/docs/model_zoo.md) for all public FUSION checkpoints, and the instructions of how to use the weights.

### Model Performance Comparison

| Model                     | # Vis Tok. | MMB_EN | MMB_CN | VizWiz | POPE | MM-Vet | MME$^P$ | MME$^C$ | Seed-Image | HallB | LLaVA_W | MMStar | MME-RW | RWQA | CV-Bench | MMVP | AI2D | MathVista | MMMU | SQA  | TextVQA | OCRBench | ChartQA | DocVQA |
| ------------------------- | :--------- | :----- | :----- | :----- | :--- | :----- | :------ | :------ | :--------- | :---- | :------ | :----- | :----- | :--- | :------- | :--- | :--- | :-------- | :--- | :--- | :------ | :------- | :------ | :----- |
| **<=4B Model Comparison** |            |        |        |        |      |        |         |         |            |       |         |        |        |      |          |      |      |           |      |      |         |          |         |        |
| **Qwen2.5VL 3B**          | -          | 79.1   | 78.1   | -      | 85.9 | 61.4   | 1592.4  | 607.5   | 74.0       | 46.6  | -       | 56.3   | 53.1   | 65.4 | -        | -    | 81.4 | 61.2      | 51.2 | 79.3 | -       | 82.8     | 84.0    | 93.93  |
| **InternVL2 4B**          | -          | 78.5   | 73.9   | -      | 84.6 | 50.5   | 1532.8  | 531.8   | 73.2       | 42.4  | -       | 53.9   | 52.1   | 60.5 | -        | -    | 79.0 | 58.5      | 48.3 | 96.0 | 74.7    | 78.4     | 81.5    | 89.2   |
| **DeepSeek-VL2-Tiny**     | -          | 74.6   | 72.1   | -      | -    | 52.5   | 1548.3  | 357.1   | 72.3       | 39.6  | -       | 45.9   | -      | 64.2 | -        | -    | 71.6 | 53.6      | 40.7 | -    | 80.7    | 80.5     | 81.0    | 86.9   |
| **MM1.5 3B**              | -          | -      | -      | -      | 88.1 | 41.0   | 1478.4  | 319.6   | 72.4       | -     | 73.0    | -      | -      | 56.9 | -        | -    | 65.7 | 44.4      | 37.1 | 85.8 | 76.5    | 65.7     | 74.2    | 87.5   |
| **Phi 3.5-Vision**        | -          | 75.5   | 64.2   | 58.2   | 82.2 | 46.5   | 1473.4  | 412.1   | 69.9       | 53.3  | 68.8    | 49.0   | -      | 53.5 | 69.3     | 67.7 | 77.4 | -         | 43.3 | 89.0 | 61.1    | 59.8     | 72.0    | 75.9   |
| **Florence-VL 3B**        | 576        | 71.6   | 60.8   | 59.1   | 88.3 | 51.0   | 1498.7  | 403.9   | 70.6       | 58.1  | 71.1    | 44.9   | -      | 60.4 | 70.2     | 64.7 | 73.8 | 52.2      | 41.8 | 84.6 | 69.1    | 63.0     | 70.7    | -      |
| **FUSION 3B (ours)**      | 780        | 79.5   | 71.7   | 64.6   | 88.9 | 57.2   | 1595.9  | 416.5   | 74.6       | 51.4  | 84.7    | 52.4   | 41.5   | 65.1 | 76.4     | 76.0 | 78.9 | 54.3      | 44.7 | 87.1 | 71.8    | 60.0     | 75.7    | 70.9   |
| **FUSION-X 3B (ours)**    | 620        | 80.3   | 74.8   | 66.1   | 88.7 | 60.3   | 1582.1  | 440.0   | 75.3       | 51.9  | 85.2    | 50.9   | 41.7   | 63.7 | 78.3     | 78.1 | 79.2 | 54.9      | 44.2 | 87.3 | 73.9    | 63.7     | 75.8    | 71.1   |
| **FUSION-L 3B (ours)**    | 308        | 77.6   | 70.8   | 65.3   | 88.3 | 56.7   | 1573.7  | 406.8   | 74.1       | 48.7  | 77.6    | 44.7   | 39.5   | 61.8 | 76.2     | 77.0 | 77.3 | 48.6      | 43.4 | 85.6 | 71.4    | 56.9     | 67.7    | 63.5   |
| **>=7B Model Comparison** |            |        |        |        |      |        |         |         |            |       |         |        |        |      |          |      |      |           |      |      |         |          |         |        |
| **Qwen2VL 7B**            | -          | 83.0   | 80.5   | -      | 88.4 | 62.0   | 1639.2  | 637.1   | 76.0       | 50.6  | -       | 60.7   | 57.4   | 70.1 | -        | -    | 83.0 | 58.2      | 54.1 | 85.5 | 84.3    | 86.6     | 83.0    | 94.5   |
| **InternVL2 8B**          | -          | 81.7   | 81.2   | -      | 86.9 | 54.2   | 1639.7  | 575.3   | 75.4       | 45.2  | -       | 61.5   | 53.5   | 64.4 | -        | -    | 83.6 | 58.3      | 52.6 | 96.3 | 77.4    | 79.4     | 83.3    | 91.6   |
| **LLaVA-OneVision 8B**    | -          | 81.7   | 78.0   | -      | 87.2 | 58.8   | 1626.0  | 483.0   | 74.8       | 47.5  | 86.9    | 60.9   | 57.5   | 65.5 | -        | -    | 81.6 | 56.1      | 47.7 | 96.6 | 78.5    | 69.7     | 78.8    | 87.5   |
| **MM1.5 7B**              | -          | -      | -      | -      | 88.6 | 42.2   | 1514.9  | 346.4   | 73.4       | -     | 74.2    | -      | -      | 62.5 | -        | -    | 72.2 | 47.6      | 41.8 | 89.6 | 76.5    | 63.5     | 88.1    | 78.2   |
| **Cambrian 8B**           | 576        | 75.9   | 67.9   | -      | 87.4 | 48.0   | 1547.1  | -       | 74.7       | 48.7  | 71.0    | 50.0   | -      | 64.2 | 72.2     | 51.3 | 73.0 | 49.0      | 42.7 | 80.4 | 71.7    | 62.4     | 73.3    | 77.8   |
| **Florence-VL 8B**        | 576        | 76.2   | 69.5   | 59.1   | 89.9 | 56.3   | 1560.0  | 381.1   | 74.9       | 57.3  | 74.2    | 50.0   | -      | 64.2 | 73.4     | 73.3 | 74.2 | 55.5      | 43.7 | 85.9 | 74.2    | 63.4     | 74.7    | -      |
| **Eagle 8B**              | 1024       | 75.9   | -      | -      | -    | -      | 1559.0  | -       | 76.3       | -     | -       | -      | -      | 66.5 | -        | 71.6 | 76.1 | 52.7      | 43.8 | 84.3 | 77.1    | 62.6     | 80.1    | 86.6   |
| **FUSION 8B (ours)**      | 780        | 80.5   | 74.9   | 59.5   | 89.3 | 60.0   | 1592.3  | 396.1   | 77.2       | 52.6  | 86.9    | 52.4   | 46.0   | 65.2 | 78.7     | 78.7 | 80.4 | 56.6      | 43.1 | 89.2 | 77.3    | 63.8     | 80.3    | 78.6   |
| **FUSION-X 8B (ours)**    | 620        | 82.0   | 76.2   | 62.9   | 88.8 | 60.0   | 1607.5  | 337.2   | 78.2       | 51.4  | 88.0    | 52.7   | 44.7   | 66.1 | 79.2     | 79.9 | 81.4 | 59.4      | 42.2 | 90.3 | 74.7    | 66.6     | 79.8    | 77.8   |
| **FUSION-L 8B (ours)**    | 308        | 80.0   | 73.6   | 59.9   | 88.5 | 57.3   | 1601.7  | 338.9   | 75.9       | 46.7  | 82.1    | 49.3   | 42.3   | 65.1 | 78.2     | 76.7 | 79.2 | 55.2      | 41.8 | 88.3 | 72.8    | 59.5     | 73.0    | 66.0   |

*For the full table, please refer to our [FUSION paper](https://arxiv.org/).*

### Using FUSION

<details>
<summary>Example Code</summary>


```Python
from fusion.model.builder import load_pretrained_model
from fusion.mm_utils import get_model_name_from_path
from fusion.eval.run_fusion import eval_model

model_path = "starriver030515/FUSION-X-Phi3.5-3B"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

Check out the details wth the `load_pretrained_model` function in `fusion/model/builder.py`.

You can also use the `eval_model` function in `fusion/eval/run_fusion.py` to get the output easily. 

``` python
model_path = "starriver030515/FUSION-X-Phi3.5-3B"
prompt = "What statue is shown in the image? What name is the person?"
image_file = "https://raw.githubusercontent.com/starriver030515/FUSION/main/images/example.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```

</details>

## Demo

### Gradio Web UI

To launch a Gradio demo locally, please run the following commands one by one. If you plan to launch multiple model workers to compare between different checkpoints, you only need to launch the controller and the web server *ONCE*.

```mermaid
flowchart BT
    %% Declare Nodes
    gws("Gradio (UI Server)")
    c("Controller (API Server):<br/>PORT: 10000")
    mw7b("Model Worker:<br/>llava-v1.5-7b<br/>PORT: 40000")
    mw13b("Model Worker:<br/>llava-v1.5-13b<br/>PORT: 40001")
    sglw13b("SGLang Backend:<br/>llava-v1.6-34b<br/>http://localhost:30000")
    lsglw13b("SGLang Worker:<br/>llava-v1.6-34b<br/>PORT: 40002")

    %% Declare Styles
    classDef data fill:#3af,stroke:#48a,stroke-width:2px,color:#444
    classDef success fill:#8f8,stroke:#0a0,stroke-width:2px,color:#444
    classDef failure fill:#f88,stroke:#f00,stroke-width:2px,color:#444

    %% Assign Styles
    class id,od data;
    class cimg,cs_s,scsim_s success;
    class ncimg,cs_f,scsim_f failure;

    subgraph Demo Connections
        direction BT
        c<-->gws
        
        mw7b<-->c
        mw13b<-->c
        lsglw13b<-->c
        sglw13b<-->lsglw13b
    end
```

#### Launch a controller

```Shell
python -m fusion.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a gradio web server.

```Shell
python -m fusion.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

You just launched the Gradio web interface. Now, you can open the web interface with the URL printed on the screen. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker.

#### Launch a model worker

This is the actual *worker* that performs the inference on the GPU.  Each worker is responsible for a single model specified in `--model-path`.

```Shell
python -m fusion.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path starriver030515/FUSION-X-Phi3.5-3B
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...".  Now, refresh your Gradio web UI, and you will see the model you just launched in the model list.

You can launch as many workers as you want, and compare between different model checkpoints in the same Gradio interface. Please keep the `--controller` the same, and modify the `--port` and `--worker` to a different port number for each worker.

```Shell
python -m fusion.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port <different from 40000, say 40001> --worker http://localhost:<change accordingly, i.e. 40001> --model-path <ckpt2>
```

### CLI Inference

Chat about images using FUSION without the need of Gradio interface.

```Shell
python -m fusion.serve.cli \
    --model-path starriver030515/FUSION-X-Phi3.5-3B \
    --image-file "https://raw.githubusercontent.com/starriver030515/FUSION/main/images/example.png" \
```

<img src="images/demo_cli.gif" width="70%">



## FUSION Dataset

<p align="center">
    <img src="images/fusion_data.jpg" alt="FUSION-Data" width="900" height="auto">
</p>


The FUSION Dataset consists of large-scale, diverse multimodal data, designed for pretraining and fine-tuning in various tasks involving both vision and language understanding. The dataset includes two main versions:

   •   **FUSION-Pretrain-10M**

   •   **FUSION-Finetune-12M**

These datasets are built upon the Cambrian-1 7M dataset by significantly expanding both the quantity and variety of data. They are designed to improve model performance in tasks such as image captioning, visual reasoning, and logical deduction.

Both FUSION-Pretrain-10M and FUSION-Finetune-12M datasets are available on [FUSION-Data](https://huggingface.co/collections/starriver030515/fusion-data-67f7ccd0b087b0aa5995cbbe).

### Data Collection

FUSION-Pretrain-10M is primarily built upon high-quality image-caption datasets, including LLaVA-558K, ShareCaptioner-1.2M, and PixelProse-9M. 

FUSION-Finetune-12M leverages a diverse range of benchmark datasets spanning categories such as OCR, Science, and General QA. It also introduces additional datasets to improve model performance in domains such as math and visual reasoning including MMathCot and MulBerry.

The datasets also includes 4 million synthetic samples generated using our **Language-Driven QA Synthesis pipeline**, with the goal of improving instruction alignment and visual understanding. These 4 million synthetic samples are divided into:

   •   2M used for pretraining (PT)

   •   2M used for supervised fine-tuning (SFT)

### Synthesized Language-Driven QA Dataset

<p align="center">
    <img src="images/synth_method.jpg" alt="FUSION-Data" width="900" height="auto">
</p>



To increase diversity and instruction alignment, **Language-Driven QA Synthesis pipeline** is used to generate synthesized data. The process includes the following steps:

1. **Caption Pool Collection**: A large pool of image captions is assembled from diverse datasets.

2.  **Description Expansion**: Captions are expanded into detailed descriptions using LLaMA3.1-70B.

3.  **Image Generation**: The expanded descriptions are used as prompts for FLUX.1 Dev to synthesize corresponding images.

4.  **QA Generation**: Descriptions and images are passed through LLaMA3.1-70B to generate high-quality Q&A pairs.

#### **Types of Synthetic Alignment Datasets**

The synthesized alignment data covers five primary categories:

   •   **SynthColor**: Describing and differentiating colors in the image.

   •   **SynthCount**: Counting and describing the number of objects in the image.

   •   **SynthSpatial**: Spatial relations between objects (e.g., left/right, above/below).

   •   **SynthScene**: General scene or landscape descriptions.

   •   **SynthText**: Identifying and describing visible text in the image.

#### Types of Synthetic Instruction Datasets

The synthesized instruction data covers six primary categories:

   •   **SynthMultiChoice QA**: Multi-turn dialogues with multiple-choice questions to teach the model to distinguish closely related options.

   •   **SynthConvShort QA**: Multi-turn dialogues with short answers focusing on fast key information extraction.

   •   **SynthConvLong QA**: Multi-turn dialogues with long-form answers to encourage detailed explanations.

   •   **SynthContrastShort QA & SynthContrastLong QA**: Dialogues comparing two similar images to train the model to observe subtle visual differences.

   •   **SynthReasoning QA**: Single-turn visual reasoning questions that require inference from visual inputs.
   
   •   **SynthText QA**: Multi-turn dialogues that identify and describe visible text in the image.

#### Data Filtering and Diversity Generation

To ensure the diversity of the generated data types, we generated several categories of data for pretraining and finetuning. We also designed a detailed and rigorous approach to filter the generated data, with specific details available in our [FUSION Paper](https://arxiv.org/). However, we found that even after filtering, there is still a considerable amount of noise and low-quality data. Additionally, the number of QA types remains limited, and many other data types have yet to be explored.

Thus, we provide an additional Hugging Face repository: [FUSION-Synth-4M](https://huggingface.co/datasets/starriver030515/FUSION-Synth-4M), which contains all the generated data along with detailed descriptions. Some data also includes generated Q&A pairs. On one hand, we hope the community will explore more efficient data filtering methods; on the other hand, using the descriptions, we hope users will explore a variety of new QA data types to further enrich the capabilities of MLLMs. 

Have Fun!

### FUSION-Pretrain-10M

FUSION-10M includes all of LLaVA-558K, ShareCaptioner-1.2M, and URSA-Alignment-860K, and we filtered 5.5M high-quality image-caption pairs from PixelProse. Additionally, we synthesized 2M specific types of image-caption pairs.

**Data Composition for Fusion-Pretrain-10M**

| **Category**   | **Percentage** |
| -------------- | -------------- |
| Short Captions | 21.3%          |
| Long Captions  | 69.8%          |
| Math Captions  | 8.9%           |

### FUSION-5M-Stage1.5

The Fusion-5M-Stage1.5 subset consists of 5 million samples used in the second phase of FUSION model training (Stage 1.5). This subset focuses on increasing the diversity of question types and conversational interactions. If using this part of the data alone, please download the corresponding [json1.5](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main/Stage1.5-json) and [image data](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main).

**Data Composition for Fusion-Stage1.5**

| **Category**  | **Percentage** |
| ------------- | -------------- |
| Language      | 4.3%           |
| General       | 20.1%          |
| OCR           | 14.1%          |
| SynthQA       | 21.5%          |
| Science       | 10.0%          |
| Long Captions | 29.7%          |

### FUSION-7M-Stage2

The Fusion-7M-Stage2 subset includes 7 million samples used in the third phase of training (Stage 2). The focus here is on vision-centric instruction tuning. If using this part of the data alone, please download the corresponding [json2](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main/Stage2-json) and [image data](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main).

**Data Composition for Fusion-Stage2**

| **Category**  | **Percentage** |
| ------------- | -------------- |
| Language      | 2.9%           |
| General       | 27.4%          |
| OCR           | 28.9%          |
| Counting      | 3.6%           |
| SynthQA       | 12.3%          |
| Code          | 0.87%          |
| Science       | 19.2%          |
| Long Captions | 5.8%           |

### Data Usage Recommendation

For Pretrain, we recommend using the entire Fusion-Pretrain-10M dataset. 

For Finetune, if sft-training in two stages, we recommend first using Fusion-5M-Stage1.5 and then using Fusion-7M-Stage2 for the training process. For one-stage sft-training, to achieve better performance, we recommend merging the Fusion-5M-Stage1.5 and Fusion-7M-Stage2 datasets by using their respective JSON files for training.

## Train

Below is the latest training configuration for FUSION.

We introduce a three-stage training framework, distinct from traditional two-stage paradigms, ensuring comprehensive alignment and integration between visual and linguistic modalities. In each stage, we unfreeze all components to ensure comprehensive optimization and seamless integration.

1. **Foundational Semantic Alignment**: We use FUSION-Pretrain-10M data to pretrain the vision encoder to establish precise semantic alignment between visual and textual representations.
2. **Contextual Multimodal Fusion**: We use FUSION-5M-Stage1.5 to enhance the model’s adaptability in aligning vision and language representations across a broad spectrum of scenarios.
3. **Visual Instruction Tuning**: We use FUSION-7M-Stage2 to expose the model to various visual tasks, enabling it to answer downstream vision-related questions effectively.

### Hyperparameters

Both hyperparameters used in pretraining and finetuning are provided below.

#### 1. Foundational Semantic Alignment

| Model Name   |   Base LLM   |      Base Vision Encoder      | Global Batch Size | Learning rate | Vision Tower Learning Rate | Epochs | Max length |
| ------------ | :----------: | :---------------------------: | :---------------: | :------------------------: | :------------------------: | :----: | ---------- |
| FUSION 3B    |  Phi3.5 3B   |   siglip-so400m-patch14-384   |        256        |            2e-5            |            2e-5            |   1    | 2048       |
| FUSION- X 3B |  Phi3.5 3B   | siglip2-giant-opt-patch16-384 |        256        |            2e-5            |            2e-5            |   1    | 2048       |
| FUSION 8B    | LLaMA-3.1 8B |   siglip-so400m-patch14-384   |        256        |            2e-5            |            2e-5            |   1    | 2048       |
| FUSION-X 8B  | LLaMA-3.1 8B | siglip2-giant-opt-patch16-384 |        256        |            2e-5            |            2e-5            |   1    | 2048       |

#### 2. Contextual Multimodal Fusion

| Model Name   |   Base LLM   |      Base Vision Encoder      | Global Batch Size | Learning rate | Vision Tower Learning Rate | Epochs | Max length |
| ------------ | :----------: | :---------------------------: | :---------------: | :------------------------: | :------------------------: | :----: | :--------: |
| FUSION 3B    |  Phi3.5 3B   |   siglip-so400m-patch14-384   |        128        |            2e-5            |            2e-6            |   1    |    4096    |
| FUSION- X 3B |  Phi3.5 3B   | siglip2-giant-opt-patch16-384 |        128        |            2e-5            |            2e-6            |   1    |    4096    |
| FUSION 8B    | LLaMA-3.1 8B |   siglip-so400m-patch14-384   |        128        |            2e-5            |            2e-6            |   1    |    4096    |
| FUSION-X 8B  | LLaMA-3.1 8B | siglip2-giant-opt-patch16-384 |        128        |            2e-5            |            2e-6            |   1    |    4096    |

#### 3. Visual Instruction Tuning

| Model Name   |   Base LLM   |      Base Vision Encoder      | Global Batch Size | Learning rate | Vision Tower Learning Rate | Epochs | Max length |
| ------------ | :----------: | :---------------------------: | :---------------: | :------------------------: | :------------------------: | :----: | :--------: |
| FUSION 3B    |  Phi3.5 3B   |   siglip-so400m-patch14-384   |        128        |            1e-5            |            1e-6            |   1    |    4096    |
| FUSION- X 3B |  Phi3.5 3B   | siglip2-giant-opt-patch16-384 |        128        |            1e-5            |            1e-6            |   1    |    4096    |
| FUSION 8B    | LLaMA-3.1 8B |   siglip-so400m-patch14-384   |        128        |            1e-5            |            1e-6            |   1    |    4096    |
| FUSION-X 8B  | LLaMA-3.1 8B | siglip2-giant-opt-patch16-384 |        128        |            1e-5            |            1e-6            |   1    |    3072    |

### Foundational Semantic Alignment

We use a combination of  LLaVA, ShareCaptioner, URSA-Alignment, PixelProse and our synthetic data to pretrain the vision encoder  to establish precise semantic alignment between visual and textual representations. We conduct extensive studies to demonstrate the necessity of unfreezing all parameters and validating the effectiveness of the generated data.

To begin, please visit our [FUSION-Pretrain-10M](https://huggingface.co/datasets/starriver030515/FUSION-Pretrain-10M) for more details. You can download the alignment data from the following links:

- [Pretrain Data (JSON file)](https://huggingface.co/datasets/starriver030515/FUSION-Pretrain-10M/tree/main/Stage1)
- [Corresponding Images](https://huggingface.co/datasets/starriver030515/FUSION-Pretrain-10M/tree/main)

We provide sample training scripts in:

- [fusion_3b_stage1](scripts/train/fusion_3b_stage1.sh)
- [fusion-x_3b_stage1](scripts/train/fusion-x_3b_stage1.sh)
- [fusion_8b_stage1](scripts/train/fusion_8b_stage1.sh)
- [fusion-x_8b_stage1](scripts/train/fusion-x_8b_stage1.sh)


### Contextual Multimodal Fusion

Similar to Foundational Semantic Alignment, please visit our [FUSION-Finetune-12M data](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M) for more details on the instruction tuning data. 

- [FUSION-5M-Stage1.5 (JSON file)](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main/Stage1.5-json)
- [Corresponding Images](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main)

We provide sample training scripts in:

- [fusion_3b_stage1.5](scripts/train/fusion_3b_stage1.5.sh)
- [fusion-x_3b_stage1.5](scripts/train/fusion-x_3b_stage1.5.sh)
- [fusion_8b_stage1.5](scripts/train/fusion_8b_stage1.5.sh)
- [fusion-x_8b_stage1.5](scripts/train/fusion-x_8b_stage1.5.sh)

### Visual Instruction Tuning

Similar to Contextual Multimodal Fusion, please visit our [FUSION-Finetune-12M data](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M) for more details on the instruction tuning data. 

- [FUSION-7M-Stage2 (JSON file)](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main/Stage2-json)
- [Corresponding Images](https://huggingface.co/datasets/starriver030515/FUSION-Finetune-12M/tree/main)

We provide sample training scripts in:

- [fusion_3b_stage2](scripts/train/fusion_3b_stage2.sh)
- [fusion-x_3b_stage2](scripts/train/fusion-x_3b_stage2.sh)
- [fusion_8b_stage2](scripts/train/fusion_8b_stage2.sh)
- [fusion-x_8b_stage2](scripts/train/fusion-x_8b_stage2.sh)

### Options to note:

- `--image_aspect_ratio`: To use our fusion model, set this value to `static_hd`. 
- `--window_size`: The window size in our interaction layers’ localized window size, which is set to a default value of 3.
- `--query_len`: the values for context-aware latent token lengths, where each value must be a perfect square. It is usually set to a single length or a series of candidate values (e.g. “4, 16, 36, 64, 144”).
- `--num_of_vision_sampler_layers`: The total number of interaction layers inserted inside the LLM.
- `--start_of_vision_sampler_layers`: The LLM layer index after which the insertion of interaction layers begins.
- `--stride_of_vision_sampler_layers`: The stride of the interaction layers module insertion inside the LLM.

## Evaluation

We have released our evaluation code in the [`eval/`](https://github.com/starriver030515/FUSION/tree/main/scripts/eval) subfolder. Please see [evaluation.md](https://github.com/starriver030515/FUSION/blob/main/docs/evaluation.md) for more details.


## Citation

If you find Cambrian useful for your research and applications, please cite using this BibTeX:

```bibtex
TBD~
```

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): We start from codebase from the amazing LLaVA
- [Cambrian-1](https://github.com/cambrian-mllm/cambrian/tree/539ffc3254bba004e5d012b65c0ad6cb308897c5): We thank Cambrian for its codebase and the awesome Cambrian-1 7M data.
- [LLaMA](https://www.bing.com/search?q=LLama3.1+meta&form=APMCS1&PC=APMC): We thank LLaMA for continuing contribution to the open-source community and providing LLaMA-3.1 checkpoints.
- [Phi](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/discover-the-new-multi-lingual-high-quality-phi-3-5-slms/4225280): We thank Phi for continuing contribution to the open-source community and providing Phi3.5 checkpoints.

## License

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/cambrian-mllm/cambrian/blob/main/LICENSE)<br>
**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-3.1, and Phi3.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.
