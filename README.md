# LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts [ICLR 2024]
(code will be available soon!)

[Hanan Gani<sup>1</sup>](https://hananshafi.github.io/), [Shariq Farooq Bhat<sup>2</sup>](https://shariqfarooq123.github.io/), [Muzammal Naseer<sup>1</sup>](https://muzammal-naseer.com/), [Salman Khan<sup>1,3</sup>](https://salman-h-khan.github.io/), [Peter Wonka<sup>2</sup>](https://peterwonka.net/)

<sup>1</sup>MBZUAI      <sup>2</sup>KAUST      <sup>3</sup>Australian National University

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.10640)

Official implementation of the paper "LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts".

<hr>

## Contents

1. [Updates](#News)
2. [Highlights](#Highlights)
3. [Main Contributions](#Main-Contributions)
4. [Installation](#Installation)
5. [Run LLMBlueprint](#Run-LLMBlueprint)
6. [Results](#Results)
7. [Citation](#Citation)
8. [Contact](#Contact)
9. [Acknowledgements](#Acknowledgements)

<hr>

## Updates

* Code is released. [Feb 12, 2024]
* Our paper is accepted at **ICLR 2024** [Jan 15, 2024]

## Highlights
![intro-diagram](https://github.com/hananshafi/llmblueprint/blob/main/docs/intro_image_arxiv.png)

> **Abstract:** *Diffusion-based generative models have significantly advanced text-to-image generation but encounter challenges when processing lengthy and intricate text prompts describing complex scenes with multiple objects. While excelling in generating images from short, single-object descriptions, these models often struggle to faithfully capture all the nuanced details within longer and more elaborate textual inputs. In response, we present a novel approach leveraging Large Language Models (LLMs) to extract critical components from text prompts, including bounding box coordinates for foreground objects, detailed textual descriptions for individual objects, and a succinct background context. These components form the foundation of our layout-to-image generation model, which operates in two phases. The initial Global Scene Generation utilizes object layouts and background context to create an initial scene but often falls short in faithfully representing object characteristics as specified in the prompts. To address this limitation, we introduce an Iterative Refinement Scheme that iteratively evaluates and refines box-level content to align them with their textual descriptions, recomposing objects as needed to ensure consistency. Our evaluation on complex prompts featuring multiple objects demonstrates a substantial improvement in recall compared to baseline diffusion models. This is further validated by a user study, underscoring the efficacy of our approach in generating coherent and detailed scenes from intricate textual inputs.*
>
<hr>

## Main Contributions
* **Scene Blueprints:** we present a novel approach leveraging Large Language Models (LLMs)
to extract critical components from text prompts, including bounding box coordinates for foreground objects, detailed textual descriptions for individual objects,
and a succinct background context. Utilizing bounding 
* **Global Scene Generation:** Utilzing the bounding box layout and genralized background prompt, we generate an initial image using Layout-to-Image generator.
* **Iterative Refinement Scheme :** Given the initial image, our proposed refinement mechanism iteratively evaluates and refines the box-level content of each object to align
them with their textual descriptions, recomposing objects as needed to ensure consistency.


## Methodology
![main-figure](https://github.com/hananshafi/llmblueprint/blob/main/docs/iclr_main_figure_revised.png)



## Installation
This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

```bash
# Create a conda environment
conda create -n llmblueprint python==3.8

# Activate the environment
conda activate llmblueprint

# Install requirements
pip install -r requirements.txt
```

## Run LLMBlueprint

Download the pretrained weights of composition model from [here](https://github.com/Fantasy-Studio/Paint-by-Example) and provide its path in yaml place inside configs folder.

#### Generate
```bash
python main.py --config configs/livingroom_1.yaml
```


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
