# Model Zoo

If you are interested in including any other details in Model Zoo, please open an issue.

The usage of FUSION checkpoints should comply with the base LLM's model license.

## FUSION

| Version  | LLM           | Vision Encoder | # Vis Tok. | Schedule   | Checkpoint                                                   | MMB_EN | MMB_CN | VizWiz | POPE | MM-Vet | MME$^P$ | MME$^C$ | Seed-Image | HallB | LLaVA_W | MMStar | MME-RW | RWQA | CV-Bench | MMVP | AI2D | MathVista | MMMU | SQA  | TextVQA | OCRBench | ChartQA | DocVQA |
| -------- | ------------- |  ------------- | ------------- | ---------- | ------------------------------------------------------------ | :----- | :----- | :----- | :--- | :----- | :------ | :------ | :--------- | :---- | :------ | :----- | :----- | :--- | :------- | :--- | :--- | :-------- | :--- | :--- | :------ | :------- | :------ | :----- |
| FUSION   | LLaMA3.1-8B     | siglip-so400m-patch14-384 | 780 | full_ft | [starriver030515/FUSION-LLaMA3.1-8B](https://huggingface.co/starriver030515/FUSION-LLaMA3.1-8B) | 80.5   | 74.9   | 59.5   | 89.3 | 60.0   | 1592.3  | 396.1   | 77.2       | 52.6  | 86.9    | 52.4   | 46.0   | 65.2 | 78.7     | 78.7 | 80.4 | 56.6      | 43.1 | 89.2 | 77.3    | 63.8     | 80.3    | 78.6   |
| FUSION   | Phi3.5-3B    | siglip-so400m-patch14-384 | 780 | full_ft | [starriver030515/FUSION-Phi3.5-3B](https://huggingface.co/starriver030515/FUSION-Phi3.5-3B) | 79.5   | 71.7   | 64.6   | 88.9 | 57.2   | 1595.9  | 416.5   | 74.6       | 51.4  | 84.7    | 52.4   | 41.5   | 65.1 | 76.4     | 76.0 | 78.9 | 54.3      | 44.7 | 87.1 | 71.8    | 60.0     | 75.7    | 70.9   |
| FUSION-X | LLaMA3.1-8B    | siglip2-giant-opt-patch16-384 | 620 |full_ft | [starriver030515/FUSION-X-LLaMA3.1-8B](https://huggingface.co/starriver030515/FUSION-X-LLaMA3.1-8B) | 82.0   | 76.2   | 62.9   | 88.8 | 60.0   | 1607.5  | 337.2   | 78.2       | 51.4  | 88.0    | 52.7   | 44.7   | 66.1 | 79.2     | 79.9 | 81.4 | 59.4      | 42.2 | 90.3 | 74.7    | 66.6     | 79.8    | 77.8   |
| FUSION-X | Phi3.5-3B | siglip2-giant-opt-patch16-384 | 620 | full_ft | [starriver030515/FUSION-X-Phi3.5-3B](https://huggingface.co/starriver030515/FUSION-X-Phi3.5-3B) | 80.3   | 74.8   | 66.1   | 88.7 | 60.3   | 1582.1  | 440.0   | 75.3       | 51.9  | 85.2    | 50.9   | 41.7   | 63.7 | 78.3     | 78.1 | 79.2 | 54.9      | 44.2 | 87.3 | 73.9    | 63.7     | 75.8    | 71.1   |

> *Despite using only 630 vision tokens, FUSION demonstrates performance comparable to current state-of-the-art models across most metrics. Notably, FUSION-X 3B achieved the highest score on MMBench among models under 4B in size, even surpassing Qwen2.5VL 3B!*



## Stage1 && Stage1.5 weights
The following weights are the Stage1 and Stage1.5 weights of our FUSION models. They are not recommended for use in any downstream tasks, but can be used for visual instruction tuning.

NOTE: When you use our pretrained weights for visual instruction tuning, it is very important to use the same base LLM and vision encoder as the one we used for pretraining the projector. Otherwise, the performance will be very poor.

| Version | Stage | Base LLM         | Vision Encoder  | training Data | schedule | Download                                                     |
| ---------------- | ---------------- | -------------- | ---------- | ------------- | -------------------- | ------------------------------------------------------------ |
FUSION | Stage1 | LLaMA3.1-8B  | siglip-so400m-patch14-384   | FUSION-Pretrain-10M     | full_ft                   | [weight](https://huggingface.co/starriver030515/FUSION-LLaMA3.1-8B-Stage1) |
FUSION | Stage1 | Phi3.5-3B | siglip-so400m-patch14-384   | FUSION-Pretrain-10M     | full_ft              | [weight](https://huggingface.co/starriver030515/FUSION-Phi3.5-3B-Stage1) |
FUSION-X | Stage1 | LLaMA3.1-8B  | siglip2-giant-opt-patch16-384   | FUSION-Pretrain-10M     | full_ft                 | [weight](https://huggingface.co/starriver030515/FUSION-X-LLaMA3.1-8B-Stage1) |
FUSION-X | Stage1 | Phi3.5-3B | siglip2-giant-opt-patch16-384         | FUSION-Pretrain-10M     | full_ft               | [weight](https://huggingface.co/starriver030515/FUSION-X-Phi3.5-3B-Stage1) |
FUSION | Stage1.5 | LLaMA3.1-8B  | siglip-so400m-patch14-384         | FUSION-Pretrain-10M + FUSION-5M-Stage1.5     | full_ft                    | [weight](https://huggingface.co/starriver030515/FUSION-LLaMA3.1-8B-Stage1.5) |
FUSION | Stage1.5 | Phi3.5-3B  | siglip-so400m-patch14-384   | FUSION-Pretrain-10M + FUSION-5M-Stage1.5      | full_ft                   | [weight](https://huggingface.co/starriver030515/FUSION-Phi3.5-3B-Stage1.5) |
FUSION-X | Stage1.5 | LLaMA3.1-8B   | siglip2-giant-opt-patch16-384   | FUSION-Pretrain-10M + FUSION-5M-Stage1.5      | full_ft                   | [weight](https://huggingface.co/starriver030515/FUSION-X-LLaMA3.1-8B-Stage1.5) |
FUSION-X | Stage1.5 | Phi3.5-3B  | siglip2-giant-opt-patch16-384         | FUSION-Pretrain-10M + FUSION-5M-Stage1.5      | full_ft               | [weight](https://huggingface.co/starriver030515/FUSION-Phi3.5-3B-Stage1.5) |
