# Post-Processing Matters: Enhancing Post-Hoc Explanation Methods in Medical Imaging

This repository contains the official PyTorch implementation of the paper: **Post-Processing Matters: Enhancing Post-Hoc Explanation Methods in Medical Imaging**. 

The codebase supports training, explanation generation, and robustness and fidelity evaluation of post-hoc explanation methods in medical imaging, including ensemble strategies and adversarial robustness analysis.


## Dependencies and Reproducibility

In order to improve the reproducibility of our experiments, we released our anaconda environment, containing all dependencies and corresponding SW versions. 
The environment can be installed by running the following command: 

```shell
conda env create -f environment.yml
```
Once the environment is created, we can use it by typing `conda activate XAI-for-Healthcare`.

## Code Structure

The code is structured as follows: 

- **training**: Utilities for fine-tuning models.
- **evaluation**: Utilities for computing robustness and fidelity.
- **agreement**: Useful functions to compute similarity between explanations.
- **ExplainableModels**: Functions to load datasets and models, generate explanations, and execute adversarial attacks.
- **MedDataset.py**: Code for creating and handling a custom data loader for medmnist.
- **plot.py**: Main functions for evaluating explanation robustness under natural data corruption and fidelity.
- **fine_tune.py**: Main script to download datasets (clean and corrupted) and finetuning models.
- **test_robustness.py**: Main script to compute robustness and fidelity of post-processed explanations.
- **test_robustness_mean.py**: Main script to compute robustness and fidelity of ensembled explanations (use --no-post_processing to evaluate raw explanations only).
- **test_robustness_topk.py**: Main script to compute robustness and fidelity of ensembled explanations(use --no-post_processing to evaluate raw explanations only).
- **compute_adversarial_robustness.py**: Main script to compute adversarial robustness of post-processed explanations.
- **compute_adversarial_robustness_ensemble.py**: Main script to compute adversarial robustness of ensembled explanations (supports mean_ensemble and topk; use --no-post_processing for raw explanations).
- **run_attack.py**: Main script to launch adversarial attacks targeting model explanations.


## Running Experiments 

### 1. DATASET PREPARATION AND MODEL FINE-TUNING

**Download clean data**
```shell
python fine_tune.py --model model  --data_name dataset --prepare_data
```
**Download corrupted data**
```shell
python fine_tune.py --model model  --data_name dataset --prepare_data --corrupt --corruption_type corruption_type --split_to_use test
```
**Finetune the model**
```shell
python fine_tune.py --model model  --data_name dataset --fine_tune
```

### 2. EXPLANATION GENERATION

Then, explanations can be computed using the following commands:

```shell
python ExplainableModels.py --model_name model --train_data_name dataset --n_classes classes --data_name dataset 
```

```shell
python ExplainableModels.py --model_name model --train_data_name dataset --n_classes classes --data_name dataset_corrupted_{corruption_type} 
```

After having executed the main functiont, a folder structure inside **attributions**" will be created containing npz files with explanations.

### 3. ROBUSTNESS AGAINST NATURAL CORRUPTIONS AND FIDELITY 

To evaluate robustness under natural corruptions and fidelity for postprocessede explanations, use:

```shell
python test_robustness.py  --model_name model --n_classes classes --data_name dataset --corruption_type corruption_type
```

To evaluate robustness under natural corruptions and fidelity for mean ensemble, use:

```shell
python test_robustness_mean.py  --model_name model --n_classes classes --data_name dataset --corruption_type corruption_type
```
ADD --ensemble_type no_postprocessing to evaluate raw explanations instead of post-processed ones.


To evaluate robustness under natural corruptions and fidelity for topk ensemble, use:

```shell
python test_robustness_topk.py  --model_name model --n_classes classes --data_name dataset --corruption_type corruption_type
```
ADD --ensemble_type no_postprocessing to evaluate raw explanations instead of post-processed ones.


### 4. ROBUSTNESS AGAINST ADVERSARIAL NOISE

To evaluate robustness under adversarial, use:

```shell
python compute_adversarial_robustness.py  --model_name model --n_classes classes --data_name dataset --epsilon epsilon
```

To evaluate robustness under adversarial for mean ensemble, use:

```shell
python compute_adversarial_robustness_ensemble.py  --model_name model --n_classes classes --data_name dataset --epsilon epsilon --ensemble_type mean_ensemble
```
ADD --no-post_processing to evaluate raw explanations instead of post-processed ones. 

To evaluate robustness under adversarial for topk ensemble, use:

```shell
python compute_adversarial_robustness_ensemble.py  --model_name model --n_classes classes --data_name dataset --epsilon epsilon --ensemble_type topk
```
ADD --no-post_processing to evaluate raw explanations instead of post-processed ones.


After having executed the main function, a folder structure inside **results** will be created containing json files with results.

To illustrate and analyze the execution flow of the different experiments, the file evaluation.ipynb is provided.

## Acknowledgements
 The authors would like to thank the contributors of [captum](https://github.com/pytorch/captum) for having facilitated the development of this project.
 
This work has been partially supported by project FISA-2023-00128 funded by the MUR program “Fondo italiano per le scienze applicate”; the EU—NGEU National Sustainable Mobility Center (CN00000023), Italian Ministry of University and Research Decree n. 1033—17/06/2022 (Spoke 10); the project Sec4AI4Sec, under the EU’s Horizon Europe Research and Innovation Programme (grant agreement no.101120393); and projects SERICS (PE00000014) and FAIR (PE0000013) under the MUR NRRP funded by the EU—NGEU.

<!--img <src="git_images/sec4AI4sec.png" alt="sec4ai4sec" style="width:70px;"/> &nbsp;&nbsp; 
<img src="git_images/elsa.png" alt="elsa" style="width:70px;"/> &nbsp;&nbsp; 
<img src="git_images/FundedbyEU.png" alt="europe" style="width:240px;" />-->