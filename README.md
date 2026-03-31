# Awesome-efficiency

Sparsity is an important paradigm to reduce the parameters and accelerate the inference. To mitigate LLM inference costs, there are many approaches are proposed, including quantization, pruning, weight sparsification and recent popular mixture of experts. Most recently, researh work have observed that activations in the MLP blocks of LLMs are sparse, which means only a few columns or rows are required in the foward pass. 


## Activation Sparsity

Recent work about the sparsity has framed the rows of weight matrices in MLP layers as experts. 
| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [Prompt-prompted Mixture of Experts for Efficient LLM Generation](https://arxiv.org/abs/2404.01365v1) | 2024 | Arxiv | no Run|
| [Scalable LLM Math Reasoning Acceleration with Low-rank Distillation](https://arxiv.org/abs/2505.07861) | 2024 | Arxiv | no Run|
| [CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models](https://arxiv.org/pdf/2404.08763) | 2024 | Arxiv | no Run|
| [MoEfication: Transformer Feed-forward Layers are Mixtures of Experts](https://arxiv.org/abs/2110.01786) | 2022 | EMNLP | no Run|

## Mixture of Experts (MoE)

### Mixture of LoRA adapters

| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [Mixture of lora experts](https://arxiv.org/abs/2404.01365v1) | 2024 | ICLR | not available|
| [Mixlora: Enhancing large language models fine-tuning with lora-based mixture of experts](https://arxiv.org/abs/2404.15159) | 2024 | ICLR | not available|
| [When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications](https://dl.acm.org/doi/pdf/10.1145/3626772.3657722) | 2024 | SIGIR |[code](https://github.com/Applied-Machine-Learning-Lab/MOELoRA-peft)|
| [Towards Modular LLMs by Building and Reusing a Library of LoRAs](https://arxiv.org/abs/2405.11157) | 2024 | ICML | no Run|
| [LoRA-Mixer: Coordinate Modular LoRA ExpertsThrough Serial Attention Routing](https://arxiv.org/pdf/2507.00029) | 2026 | ICLR | not available|
| [UNITE: Universal kNowledge Integration from Task-specific Experts](https://openreview.net/forum?id=WnW0zndglL) | 2026 | ICLR | not available|




## KV cache compression
| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference](https://arxiv.org/abs/2402.09398) | 2024 | Arxiv | no Run|
| [Towards Modular LLMs by Building and Reusing a Library of LoRAs](https://arxiv.org/abs/2405.11157) | 2024 | ICML | no Run|

## Routing Methods

## Model Merging
| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [Learning to Route Among Specialized Experts for Zero-Shot Generalization](https://arxiv.org/pdf/2402.05859) | 2024 | ICLR | no Run|


## Evaluation of  Model Merging
| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [MergeBench: A Benchmark for Merging Domain-Specialized LLMs](https://arxiv.org/pdf/2505.10833) | 2025 | NeurIPS | no Run|

## methods for model merging

| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/2306.12621) | 2018 | UAI | no Run|
| [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/pdf/2203.05482) | 2022 | ICML2022 | no Run|
| [Merging models with fisher-weighted averaging](https://arxiv.org/abs/2111.09832) | 2022 | NeurIPS | no Run|
| [Dataless Knowledge Fusion by Merging Weights of Language Models](https://arxiv.org/abs/2212.09849) | 2023 | ICLR | no Run|
| [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) | 2023 | ICLR | no Run|
| [Ties-merging: Resolving interference when merging models.](https://arxiv.org/abs/2306.01708) | 2023 | NeurIPS | no Run|
| [Language Models are Super Mario:Absorbing Abilities from Homologous Models as a Free Lunch](https://openreview.net/pdf?id=fq0NaiU8Ex) | 2024 | ICML | no Run|
| [Localizing Task Information for Improved Model Merging and Compression](https://arxiv.org/pdf/2405.07813) | 2024 | ICML | no Run|
| [Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging](https://proceedings.neurips.cc/paper_files/paper/2024/file/8fcd17eb91bae20d9826786d7d6be799-Paper-Conference.pdf) | 2024 | NeurIPS | no Run|
| [Multi-task model merging via adaptive weight disentanglement](https://arxiv.org/pdf/2411.18729) | 2024 | Arxiv | no Run|
| [MODEL MERGING WITH SVD TO TIE THE KNOTS](https://arxiv.org/pdf/2410.19735) | 2025 | ICLR | no Run|
| [Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors](https://arxiv.org/pdf/2503.08099) | 2025 | ICML | no Run|
| [Modeling Multi-Task Model Merging as Adaptive Projective Gradient Descent](https://arxiv.org/abs/2501.01230) | 2025 | ICML | no Run|
| [DC-Merge: Improving Model Merging with Directional Consistency](https://arxiv.org/abs/2603.06242) | 2026 | CVPR | no Run|


### multimodal model merging
| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [An Empirical Study of Multimodal Model Merging](https://arxiv.org/pdf/2304.14933) | 2023 | EMNLP | no Run|

### model merging survey



