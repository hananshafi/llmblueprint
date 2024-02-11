# LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts [ICLR 2024]
(code will be available soon!)

[Hanan Gani<sup>1</sup>](https://hananshafi.github.io/), [Shariq Farooq Bhat<sup>2</sup>](https://shariqfarooq123.github.io/), [Muzammal Naseer<sup>1</sup>](https://muzammal-naseer.com/), [Salman Khan<sup>1</sup>](https://salman-h-khan.github.io/), [Peter Wonka<sup>2</sup>](https://peterwonka.net/)

<sup>1</sup>MBZUAI      <sup>2</sup>KAUST

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.10640)

Official implementation of the paper "LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts".

<hr>

## Contents

1. [Updates](#News)
2. [Highlights](#Highlights)
3. [Main Contributions](#Main-Contributions)
4. [Installation](#Installation)
5. [Run PromptAlign](#Run-PromptAlign)
6. [Results](#Results)
7. [Citation](#Citation)
8. [Contact](#Contact)
9. [Acknowledgements](#Acknowledgements)

<hr>

## Updates

* Code for PrompAlign is released. [November 3, 2023]
* Our paper is accepted at NeurIPS 2023 [September 22, 2023]

## Highlights
![concept-diagram](https://jameelhassan.github.io/promptalign/static/images/conceptdiagram.png)

> **Abstract:** *The promising zero-shot generalization of vision-language models such as CLIP
has led to their adoption using prompt learning for numerous downstream tasks.
Previous works have shown test-time prompt tuning using entropy minimization
to adapt text prompts for unseen domains. While effective, this overlooks the key
cause for performance degradation to unseen domains – distribution shift. In this
work, we explicitly handle this problem by aligning the out-of-distribution (OOD)
test sample statistics to those of the source data using prompt tuning. We use a
single test sample to adapt multi-modal prompts at test time by minimizing the
feature distribution shift to bridge the gap in the test domain. Evaluating against the
domain generalization benchmark, our method improves zero-shot top-1 accuracy
beyond existing prompt-learning techniques, with a 3.08% improvement over the
baseline MaPLe. In cross-dataset generalization with unseen categories across 10
datasets, our method improves by 1.82% compared to the existing state-of-the-art.*
>
<hr>

![intro-diagram](https://github.com/hananshafi/llmblueprint/blob/main/docs/intro_image_arxiv.png)


<hr>

## Methodology
![main-figure](https://github.com/hananshafi/llmblueprint/blob/main/docs/iclr_main_figure_arxiv.png)


<hr>

## Code
The full code for the paper will be available soon! Check back after a while.


## Contact
Should you have any questions, please contact at hanan.ghani@mbzuai.ac.ae

## Citation
If you use our work, please consider citing:
```bibtex 
@misc{gani2023llm,
      title={LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts}, 
      author={Hanan Gani and Shariq Farooq Bhat and Muzammal Naseer and Salman Khan and Peter Wonka},
      year={2023},
      eprint={2310.10640},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgements
Our code is built on the repositories of  [LLM Grounded Diffusion](https://github.com/TonyLianLong/LLM-groundedDiffusion) and [Paint by Example](https://github.com/Fantasy-Studio/Paint-by-Example). We thank them for their open-source implementation and instructions.
