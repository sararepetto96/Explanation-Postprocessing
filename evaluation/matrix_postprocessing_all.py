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

corruption_types = [
    "gaussian_noise",   
    "pixelate",
    "brightness_up"
]

print("working in: ",os.getcwd())

results_list = []

for corr in corruption_types:
    path_corr = f'results/{args.data_name}/fidelity_and_robustness/{args.normalization}/{args.agreement_measure}/{args.model_name}/{corr}'
    file_path_corr = f'{path_corr}/results_0'
    with open(file_path_corr, 'r') as f:
        results_list.append(json.load(f))


path = f'results/{args.data_name}/fidelity_and_robustness/{args.normalization}/{args.agreement_measure}/{args.model_name}/{args.corruption_type}'
file_path = f'{path}/results_0'
adv_path = f'results/{args.data_name}/fidelity_and_robustness/{args.normalization}/{args.agreement_measure}/{args.model_name}/adversarial_robustness_new_2'

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
    lambda x: x[2] if isinstance(x, (list, tuple)) and len(x) > 2 else np.nan
)
df_normal["kernel_size"] = pd.to_numeric(df_normal["kernel_size"], errors="coerce")
df_normal_adv["kernel_size"] = pd.to_numeric(df_normal_adv["kernel_size"], errors="coerce")

matrix_fidelity = df_normal.pivot_table(index="technique", columns="kernel_size", values="fidelity_value")
matrix_fidelity = matrix_fidelity[sorted(matrix_fidelity.columns)]
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
matrix_adv_noise = matrix_adv_noise.rename(index=tech_abbrev)

matrices = [
    matrix_fidelity,
    natural_matrices[0],   # gaussian
    natural_matrices[1],   # pixelate
    natural_matrices[2],   # brightness
    matrix_adv_noise
]


fig = plt.figure(figsize=(28, 4.5))
gs = GridSpec(1, 6, width_ratios=[-0.05, 1, 1, 1, 1, 1], wspace=0.08)

ax_labels = fig.add_subplot(gs[0])
axes = [fig.add_subplot(gs[i]) for i in range(1, 6)]

norms = [
    PowerNorm(
        gamma=0.5,
        vmin=matrix_fidelity.min().min(),
        vmax=matrix_fidelity.max().max()
    ),
    PowerNorm(gamma=0.5,
              vmin=natural_matrices[0].min().min(),
              vmax=natural_matrices[0].max().max()),

    PowerNorm(gamma=0.5,
              vmin=natural_matrices[1].min().min(),
              vmax=natural_matrices[1].max().max()),

    PowerNorm(gamma=0.5,
              vmin=natural_matrices[2].min().min(),
              vmax=natural_matrices[2].max().max()),
    SymLogNorm(
        linthresh=1e-3,
        linscale=0.5,
        vmin=matrix_adv_noise.min().min(),
        vmax=matrix_adv_noise.max().max(),
        base=10
    )
    ]

cmaps = ["mako"] * 5


ax_labels.set_xlim(0, 1)
ax_labels.set_ylim(0, matrix_fidelity.shape[0])
ax_labels.invert_yaxis()

ax_labels.set_xticks([])
ax_labels.set_yticks(np.arange(matrix_fidelity.shape[0]) + 0.5)
ax_labels.set_yticklabels(matrix_fidelity.index, fontsize=14)
ax_labels.set_ylabel("Technique", fontsize=14)

for spine in ax_labels.spines.values():
    spine.set_visible(False)

red_indexes =[]
# ---- Heatmap ----
for i, ax in enumerate(axes):
    sns.heatmap(
        matrices[i],
        annot=True,
        fmt=".3f",
        cmap=cmaps[i],
        norm=norms[i],
        linewidths=0.4,
        cbar=False,
        ax=ax,
        yticklabels=False,
        annot_kws={"fontsize": 13}
    )

    for el, row in enumerate(matrices[i].index):
        if i ==0:
            j = matrices[i].loc[row].idxmin()
            j_index = list(matrices[i].columns).index(j)
        else:
            j_index = red_indexes[el]

        circle = mpatches.Circle(
            (j_index + 0.5, el + 0.5),
            radius=0.45,
            fill=False,
            edgecolor="red",
            linewidth=2.5
        )
        ax.add_patch(circle)
        if i ==0:
            red_indexes.append(j_index)
        
    columns ={1:2,2:2,3:3,4:3,5:4,6:3,7:4,8:3}

    for el, row in enumerate(matrices[i].index):
        print(el+1)
        j_index = columns[el+1]

        if el not in [2,3,7]:

            circle = mpatches.Circle(
                (j_index + 0.5, el + 0.5),
                radius=0.45,
                fill=False,
                edgecolor="yellow",
                linewidth=2.5
            )
            ax.add_patch(circle)

    ax.set_xlabel("Window Size", fontsize=17)
    ax.set_title(f" {title[i]}", fontsize=19, weight="bold")
    ax.set_ylabel("")

    # --- Colorbar under every heatmap ---
    cax = inset_axes(
        ax,
        width="40%",
        height="3%",
        loc="lower center",
        bbox_to_anchor=(0, -0.20, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    sm = plt.cm.ScalarMappable(cmap=cmaps[i], norm=norms[i])
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=9)

plt.savefig(f'{path}/matrix_all.png', dpi=300, bbox_inches="tight")
plt.show()

