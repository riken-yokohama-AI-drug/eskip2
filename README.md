# eSkip2

## Overview

**eSkip2** is a machine learning–based tool for predicting effective antisense oligonucleotides (ASOs) that induce exon skipping.  
It is designed to support exon-skipping ASO development by prioritizing target regions within both **exonic** and **intronic** sequences.  

Unlike previous approaches limited to specific sequence contexts, eSkip2 can handle diverse scenarios, including **dual-targeting ASO designs**.  
The model builds on the genome-pretrained **HyenaDNA** architecture and has been fine-tuned using curated datasets of ASO activity across multiple genes and splicing contexts.  

---

## Key Features

- **Complex ASO design**: Helps prioritize effective target sites for dual-targeting ASOs.
- **Broad applicability**: Works with both exonic and intronic regions.

---

## Installation

eSkip2 is distributed as a patch and additional scripts on top of the [HyenaDNA](https://github.com/HazyResearch/hyena-dna) repository.  

### Step 1. Clone HyenaDNA
```bash
git clone https://github.com/HazyResearch/hyena-dna
cd hyena-dna
git checkout d553021   # specific commit used for eSkip2
```

### Step 2. Apply the eSkip2 patch

Clone this repository and apply the patch file:

```bash
# in a separate location
git clone https://github.com/riken-yokohama-AI-drug/eskip2
cd eskip2

# apply patch to hyena-dna
cd ../hyena-dna
patch -p1 < ../eskip2/diff.patch
```

### Step 3. Install dependencies
HyenaDNA requirements must be installed beforehand (see the HyenaDNA repository for details).

## Usage
### Inference
This repository additionally provides scripts for prediction and preprocessing:

- **`predict.py`**  
  Run inference using eSkip2 with pretrained weights.  
    ```bash
    python ../eskip2/predict.py \
    --in_csv Input CSV, must have 'sequence' and 'label' columns \
    --out_csv Output CSV with predictions \
    --model_dir The checkpoint directory \
    --model_name Model name, e.g., hyenadna-medium-160k-seqlen-FTed \
    --max_length Max_length of input, this should be consisitent with what was set when training, default: 500.
    ```


### Generation of an input CSV for walking and dual-targeting designs
- **`make-Nmasked-sequence.py`**  
  Utility script to generate sequence inputs in the correct format for inference.  
    ```bash
    python ../eskip2/make-Nmasked-sequence.py \
    --sequence "target sequence consisting of an exon and its flanking 50-nucleotide intronic sequences"  \
    --length "Length of N region" \
    --min_region_length "Minimum allowed length for each region in a dual-targeting ASO" \
    --generation_type "Generation method: single, double, triple, or dual"
    --output "Output file name"
    ```

### Fine-tuning and gene-addaptation
For users who would like to **fine-tune pretrained hyenaDNA on their own datasets** or **adapt the eSkip2-base (checkpoints.zip) to a specific gene of interest**, we provide a Google Colab notebook.  
This notebook demonstrates how to:

- Prepare input data for gene-addaptation
- Load pretrained weights from eSkip2-base (checkpoints.zip)
- Run training with your custom dataset  

Please open the Colab notebook for step-by-step instructions:

https://colab.research.google.com/drive/1R-LkMtCbEGK31URf5-DtYRVr1SGYkaZb#scrollTo=WbGIEquVUH-w

## Requirements

- Python 3.8+  
- All dependencies listed in the [HyenaDNA](https://github.com/HazyResearch/hyena-dna) repository  



## Citation

```text
```bibtex
@article{chiba2026,
  title   = {eSkip2 prioritizes exon-skipping antisense oligonucleotide target regions across exon–intron contexts},
  author  = {Shuntaro Chiba, Katsuhiko Kunitake, Satomi Shirakaki, Umme Sabrina Haque, Harry Wilton-Clark, Md Nur Ahad Shah, Jamie Leckie, Kosuke Matsui, Fumie Uno-Ono, Toshifumi Yokota, Yoshitsugu Aoki, Yasushi Okuno},
  journal = {Preprint},
  year    = {2026}
}
```
