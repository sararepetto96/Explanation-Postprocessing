import numpy as np
import matplotlib.pyplot as plt
import os
import json

from matplotlib.colors import PowerNorm
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,Normalize, SymLogNorm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

parser = argparse.ArgumentParser(description='explanation analyzing')
parser.add_argument('--corruption_type', type=str, default = 'gaussian_noise')
parser.add_argument('--model_name', type=str, default = 'resnet50')
parser.add_argument('--data_name', type=str, default = 'dermamnist')
parser.add_argument('--agreement_measure', type=str, default = 'l1')
parser.add_argument('--n_classes', type=int, default = 7)
parser.add_argument('--normalization', type=str, default = 'quantil_local')
parser.add_argument('--occluded_most_important', action='store_true', help='Enable my flag')
parser.set_defaults(occluded_most_important=True)
parser.add_argument('--no-occluded_most_important', dest='occluded_most_important', action='store_false', help='Disable my flag')
args = parser.parse_args()

#datasets = ["dermamnist", "bloodmnist","tissuemnist","retinamnist", "pathmnist", "organcmnist","octmnist","pneumoniamnist","breastmnist"]
datasets = ["tissuemnist"]
models = ["resnet50", "densenet121","regnety_008"]
model_names_tex = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "regnety_008": "RegNety008"
}

dataset_names_tex = {
    "tissuemnist": "TissueMNIST",
    "retinamnist": "RetinaMNIST",
    "dermamnist": "DermaMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "octmnist": "OctMNIST",
    "organcmnist": "OrganMNIST",
    "breastmnist": "BreastMNIST",
    "pathmnist": "PathMNIST",
    "bloodmnist": "BloodMNIST",
}

pixels = {
    "tissuemnist": {
        "densenet121": 4,
        "resnet50": 2,
        "regnety_008": 6,
    },
    "retinamnist": {
        "densenet121": 1,
        "resnet50": 1,
        "regnety_008": 1,
    },
    "dermamnist": {
        "densenet121": 2,
        "resnet50": 2,
        "regnety_008": 8,
    },
    "pneumoniamnist": {
        "densenet121": 4,
        "resnet50": 3,
        "regnety_008": 4,
    },
    "octmnist": {
        "densenet121": 3,
        "resnet50": 3,
        "regnety_008": 2,
    },
    "organcmnist": {
        "densenet121": 2,
        "resnet50": 3,
        "regnety_008": 3,
    },
    "breastmnist": {
        "densenet121": 7,
        "resnet50": 7,
        "regnety_008": 10,
    },
    "bloodmnist": {
        "densenet121": 2,
        "resnet50": 2,
        "regnety_008": 2,
    },
    "pathmnist": {
        "densenet121": 10,
        "resnet50": 8,
        "regnety_008": 8,
    },
}


configs = [(d, m) for d in datasets for m in models]
n_rows = len(configs)

tech_abbrev = {
    "DeepLift": "DL",
    "GradientShap": "GS",
    "GuidedBackprop": "GB",
    "GuidedGradCam": "GGC",
    "InputXGradient": "IxG",
    "IntegratedGradients": "IG",
    "Saliency": "SA",
    "SmoothGrad": "SG"
}

titles = [
    r'$\mathtt{F}$',
    r'$\mathtt{R}^{n_g}$',
    r'$\mathtt{R}^{n_p}$',
    r'$\mathtt{R}^{n_b}$',
    r'$\mathtt{R}^{a}$'
]

corruption_types = [
    "gaussian_noise",   # gaussian_noise (o quello passato)
    "pixelate",
    "brightness_up"
]
cmaps = ["mako"] * 5

def build_matrices(data_name, model_name, args):
    results_list = []

    for corr in corruption_types:
        path_corr = f'results/{data_name}/fidelity_and_robustness/quantil_local/l1/{model_name}/{corr}'
        file_path_corr = f'{path_corr}/results'
        with open(file_path_corr, 'r') as f:
            results_list.append(json.load(f))


    ##fare in modo che la working che vede è quella principale
    path = f'results/{data_name}/fidelity_and_robustness/quantil_local/l1/{model_name}/{args.corruption_type}'
    file_path = f'{path}/results'
    adv_path = f'results/{data_name}/fidelity_and_robustness/quantil_local/l1/{model_name}/adversarial_robustness_new_2'

    with open(file_path, 'r') as f:
        results = json.load(f)

    with open(adv_path, 'r') as f:
        results_adv = json.load(f)

    tech_abbrev = {
        "DeepLift": "DL",
        "GradientShap": "GS",
        "GuidedBackprop": "GB",
        "GuidedGradCam": "GGC",
        "InputXGradient": "IxG",
        "IntegratedGradients": "IG",
        "Saliency": "SA",
        "SmoothGrad": "SG"
    }

    # DataFrame
    df = pd.DataFrame(results)
    df_adv = pd.DataFrame(results_adv)

    df["kernel_size_str"] = df["kernel_size"].astype(str).str.lower().str.strip()
    df_adv["kernel_size_str"] = df_adv["kernel_size"].astype(str).str.lower().str.strip()
    df_normal = df[df["kernel_size_str"] != "none"].copy()
    df_normal_adv = df_adv[df_adv["kernel_size_str"] != "none"].copy()
    df_normal["fidelity_value"] = df_normal["accuracy_curve"].apply(
        lambda x: x[pixels[data_name][model_name]] if isinstance(x, (list, tuple)) and len(x) > pixels[data_name][model_name] else np.nan
    )
    df_normal["kernel_size"] = pd.to_numeric(df_normal["kernel_size"], errors="coerce")
    df_normal_adv["kernel_size"] = pd.to_numeric(df_normal_adv["kernel_size"], errors="coerce")

    matrix_fidelity = df_normal.pivot_table(index="technique", columns="kernel_size", values="fidelity_value")
    matrix_fidelity = matrix_fidelity[sorted(matrix_fidelity.columns)]
    #matrix_natural_noise = df_normal.pivot_table(index="technique", columns="kernel_size", values="robustness")
    #matrix_natural_nosie = matrix_natural_noise[sorted(matrix_natural_noise.columns)]
    matrix_adv_noise = df_normal_adv.pivot_table(index="technique", columns="kernel_size", values="adversarial_robustness")
    matrix_adv_nosie = matrix_adv_noise[sorted(matrix_adv_noise.columns)]

    natural_matrices = []

    for results_corr in results_list:
        df_corr = pd.DataFrame(results_corr)
        df_corr["kernel_size_str"] = df_corr["kernel_size"].astype(str).str.lower().str.strip()
        df_corr = df_corr[df_corr["kernel_size_str"] != "none"].copy()
        df_corr["kernel_size"] = pd.to_numeric(df_corr["kernel_size"], errors="coerce")

        matrix = df_corr.pivot_table(
            index="technique",
            columns="kernel_size",
            values="robustness"
        )
        matrix = matrix[sorted(matrix.columns)]
        matrix = matrix.rename(index=tech_abbrev)
        natural_matrices.append(matrix)

    title = [
        r'$\mathtt{F}$',
        r'$\mathtt{R}^{n_g}$',
        r'$\mathtt{R}^{n_p}$',
        r'$\mathtt{R}^{n_b}$',
        r'$\mathtt{R}^{a}$'
    ]

    matrix_fidelity = matrix_fidelity.rename(index=tech_abbrev)
    #matrix_natural = matrix_natural_noise.rename(index=tech_abbrev)
    matrix_adv_noise = matrix_adv_noise.rename(index=tech_abbrev)
    #matrices = [
        #matrix_fidelity,
        #matrix_natural_noise,
        #matrix_adv_noise
    #]
    matrices = [
    matrix_fidelity,
    natural_matrices[0],   # gaussian
    natural_matrices[1],   # pixelate
    natural_matrices[2],   # brightness
    matrix_adv_noise
    ]

    return matrices 

fig = plt.figure(figsize=(28, 5.2 * n_rows))
gs = GridSpec(
    n_rows, 6,
    width_ratios=[-0.07, 1, 1, 1, 1, 1],
    hspace=0.45,
    wspace=0.08
)

for r, (dataset, model) in enumerate(configs):

    matrices = build_matrices(dataset, model, args)
    matrix_fidelity = matrices[0]

    axes = [fig.add_subplot(gs[r, c]) for c in range(1, 6)]
    ax_labels = fig.add_subplot(gs[r, 0], sharey=axes[0])

    dummy = np.zeros((len(matrix_fidelity.index), 1))

    sns.heatmap(
        dummy,
        cmap="Greys",
        cbar=False,
        linewidths=0.4,
        ax=ax_labels,
        yticklabels=matrix_fidelity.index,
        xticklabels=False
    )

    ax_labels.set_ylabel("Explanation Technique", fontsize=14)
    ax_labels.tick_params(axis="y", labelsize=10)
    row_title = f"{dataset_names_tex[dataset]} – {model_names_tex[model]}"

    ax_labels.text(
    2.5, 0.5,
    row_title,
    transform=ax_labels.transAxes,
    rotation=90,
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold"
)

    # ---- norms per riga ----
    norms = [
        PowerNorm(0.5, matrices[0].min().min(), matrices[0].max().max()),
        PowerNorm(0.5, matrices[1].min().min(), matrices[1].max().max()),
        PowerNorm(0.5, matrices[2].min().min(), matrices[2].max().max()),
        PowerNorm(0.5, matrices[3].min().min(), matrices[3].max().max()),
        SymLogNorm(
            linthresh=1e-3,
            vmin=matrices[4].min().min(),
            vmax=matrices[4].max().max()
        )
    ]

    # ---- cerchi gialli (SOLO dalla fidelity) ----
    yellow_indexes = []

    for tech in matrix_fidelity.index:
        row = matrix_fidelity.loc[tech]
        
        best = row.min()                        # il migliore valore nella riga (min)
        best_kernel = row.idxmin()
        threshold = best * 1.05                  # massimo consentito (+5%)
        
        # seleziona i valori <= threshold
        #candidates = row[row <= threshold]
        candidates = row[
                (row <= threshold) &
                (row.index > best_kernel)
            ]
        
        if len(candidates) == 0:
            yellow_indexes.append(None)
        else:
            # prendi il massimo tra i candidati validi
            chosen_col = candidates.idxmax()
            yellow_indexes.append(list(matrix_fidelity.columns).index(chosen_col))
        
    red_indexes=[]

    # ---- heatmaps ----
    for i, ax in enumerate(axes):
        sns.heatmap(
            matrices[i],
            cmap=cmaps[i],
            norm=norms[i],
            annot=True,        # annot solo prima riga
            fmt=".3f",
            linewidths=0.4,
            cbar=False,
            ax=ax,
            yticklabels=True,
            annot_kws={"size": 13}
        )

        for el, tech in enumerate(matrices[i].index):

            # rosso
            if i == 0:
                j = matrices[i].loc[tech].idxmin()
                j_index = list(matrices[i].columns).index(j)
                red_indexes.append(j_index)
            else:
                j_index = red_indexes[el]

            ax.add_patch(mpatches.Circle(
                (j_index + 0.5, el + 0.5),
                0.45, fill=False, edgecolor="red", linewidth=2.2
            ))

            # giallo
            yj = yellow_indexes[el]
            if yj is not None and yj != j_index:
                ax.add_patch(mpatches.Circle(
                    (yj + 0.5, el + 0.5),
                    0.45, fill=False, edgecolor="yellow", linewidth=2.2
                ))

        ax.set_title(titles[i], fontsize=18, weight="bold")
        ax.set_xlabel(r"Window Size $s$")
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("")
        

        # colorbar
        cax = inset_axes(ax, "40%", "3%", loc="lower center",
                         bbox_to_anchor=(0, -0.25, 1, 1),
                         bbox_transform=ax.transAxes)
        plt.colorbar(
            plt.cm.ScalarMappable(norm=norms[i], cmap=cmaps[i]),
            cax=cax, orientation="horizontal"
        )

plt.savefig(f"matrix_all_models_{dataset}.png", dpi=300, bbox_inches="tight")
plt.show()