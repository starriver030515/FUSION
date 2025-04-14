# Evaluation

In FUSION, we evaluate models on a diverse set of 21 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

we have adapted lmms-eval and provided the corresponding model files, available in the [lmms-eval/](https://github.com/starriver030515/FUSION/tree/main/scripts/lmms-eval) folder. Additionally, we have implemented benchmark evaluation code in LLaVA format for quick evaluation, found in the [eval/](https://github.com/starriver030515/FUSION/tree/main/scripts/eval) folder.

## Evaluate with lmms-eval

This is the recommended method for evaluating the FUSION model.

Simply follow the instructions in the lmms-eval README to set up both the lmms-eval and FUSION environments. After configuration, copy the files from the [lmms-eval/](https://github.com/starriver030515/FUSION/tree/main/scripts/lmms-eval) folder into lmms-eval/models. Then, update lmms-eval/models/\_\_init\_\_.py by adding the following entries to AVAILABLE_MODELS:

```
"fusion_phi": "Fusion_Phi",
"fuson": "Fusion_LLaMA",
```

To run evaluations, simply specify --model Fusion_Phi or --model Fusion_LLaMA.

**Note:**

lmms-eval does not support benchmarks such as CV-Bench and MMVP. To evaluate on these benchmarks, please use the scripts provided directly in the [eval/](https://github.com/starriver030515/FUSION/tree/main/scripts/eval) folder.

## Evaluate on Custom Datasets

We followed the evaluation of LLaVA. You can evaluate FUSION on your custom datasets by converting your dataset to LLaVA's jsonl format, and evaluate using [`model_vqa.py`](https://github.com/starriver030515/FUSION/blob/main/fusion/eval/model_vqa.py).

Below we provide a general guideline for evaluating datasets with some common formats.

1. Short-answer (e.g. MME).

```
<question>
Answer the question using a single word or phrase.
```

2. Option-only for multiple-choice (e.g. MMBench, SEED-Bench).

```
<question>
A. <option_1>
B. <option_2>
C. <option_3>
D. <option_4>
Answer with the option's letter from the given choices directly.
```

3. Natural QA (e.g. LLaVA-Bench, MM-Vet).

No postprocessing is needed.

## Scripts

Before preparing task-specific data, **you MUST first download [eval.zip](https://drive.google.com/file/d/1ZpTFLRrhULzijadyrUsSyvnTBd1Boc4u/view?usp=share_link)**. It contains custom annotations, scripts. Extract to `playground/eval`. This also provides a general structure for all datasets.

### VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `playground/eval/vizwiz`.
2. Multiple-GPU inference.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash vizwiz.sh
```

3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2185/my-submission): `playground/eval/vizwiz/answers_upload`.

### ScienceQA

1. Under `playground/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Multiple-GPU inference and evaluate.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `playground/eval/textvqa`.
2. Multiple-GPU inference and evaluate.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash textvqa.sh
```

### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `playground/eval/pope`.
2. Multiple-GPU inference and evaluate.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash pope.sh
```

### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. put the official `eval_tool` and `MME_Benchmark_release_version` under `playground/eval/MME`.
4. Multiple-GPU inference and evaluate.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash mme.sh
```

### MMBench

1. Download [`MMBench_TEST_EN_legacy.tsv`](http://opencompass.openxlab.space/utils/MMBench/MMBench_TEST_EN_legacy.tsv), rename it to `MMBench_TEST_EN_legacy.tsv`and put under `playground/eval/mmbench`.
2. Multiple-GPU inference.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash mmbench.sh
```

3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `playground/eval/mmbench/answers_upload/`.

### MMBench-CN

1. Download [`MMBench_TEST_CN_legacy.tsv`](http://opencompass.openxlab.space/utils/MMBench/MMBench_TEST_CN_legacy.tsv) , rename it to `MMBench_TEST_CN_legacy.tsv` and put under `playground/eval/mmbench_cn`.
2. Multiple-GPU inference.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash mmbench_cn.sh
```

3. Submit the results to the evaluation server: `playground/eval/mmbench/answers_upload/`.

### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `playground/eval/mmvet`.
2. Multiple-GPU inference.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash mmvet.sh
```

3. Evaluate the predictions in `playground/eval/mmvet/results` using the official jupyter notebook.

### CVBench

1. Download [`CVBench_img`](https://huggingface.co/datasets/nyu-visionx/CV-Bench), rename it to `img`and put under `playground/eval/CVBench`.
2. Multiple-GPU inference.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash cvbench.sh
```

### MMVP

1. Download [`MMVP_Images`](https://huggingface.co/datasets/MMVP/MMVP), rename it to `MMVP_Images `and put under `playground/eval/MMVP`.
2. Multiple-GPU inference.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash mmvp.sh
```

### MMMU

1. Download [`MMMU`](https://huggingface.co/datasets/MMMU/MMMU) ,rename it to `images `   and put under `playground/eval/MMMU`.
2. Multiple-GPU inference.

```Shell
cd scripts/eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash mmmu_val.sh
```

### Notes:

We also provide a separate eval repository that organizes the above benchmark evaluations for convenience. You can directly download this from our Hugging Face repository:

👉 [FUSION Evaluation Benchmarks](https://huggingface.co/datasets/starriver030515/FUSION-Eval)

This repository includes structured resources for the above benchmarks, making it easier to set up and run evaluations.