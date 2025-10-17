# Awesome-efficiency

Sparsity is an important paradigm to reduce the parameters and accelerate the inference. 
To mitigate LLM inference costs, there are many approaches are proposed, including quantization, pruning, weight sparsification and recent popular mixture of experts. Most recently, researh work have observed that activations in the MLP blocks of LLMs are sparse, which means only a few columns or rows are required in the foward pass. 

| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [Prompt-prompted Mixture of Experts for Efficient LLM Generation](https://arxiv.org/abs/2404.01365v1) | 2024 | Arxiv | no Run|
| [Scalable LLM Math Reasoning Acceleration with Low-rank Distillation](https://arxiv.org/abs/2505.07861) | 2024 | Arxiv | no Run|
| [CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models](https://arxiv.org/pdf/2404.08763) | 2024 | Arxiv | no Run|

## Mixture of Experts (MoE)

### Mixture of LoRA adapters

| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [Mixture of lora experts](https://arxiv.org/abs/2404.01365v1) | 2024 | ICLR | not available|
| [Mixlora: Enhancing large language models fine-tuning with lora-based mixture of experts](https://arxiv.org/abs/2404.15159) | 2024 | ICLR | not available|
| [When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications](https://dl.acm.org/doi/pdf/10.1145/3626772.3657722) | 2024 | SIGIR |[code](https://github.com/Applied-Machine-Learning-Lab/MOELoRA-peft)|









## KV cache compression
| **Paper Title** | **Year** | **Conference/Journal** | **Code** |
| --------------- | :----: | :----: | :----: |
| [Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference](https://arxiv.org/abs/2402.09398) | 2024 | Arxiv | no Run|

