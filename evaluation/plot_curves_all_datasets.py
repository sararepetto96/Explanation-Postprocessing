import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from matplotlib.lines import Line2D


def select_best_techniques(results, n_best=3):
    """Seleziona le migliori tecniche basate su accuracy_curve[2] con kernel_size=1"""
    results_k1 = [r for r in results if r["kernel_size"] == 1]
    results_k1 = sorted(results_k1, key=lambda r: r["accuracy_curve"][2], reverse=True)
    return [r["technique"] for r in results_k1[:n_best]]


def plot_accuracy_curves(ax, results, selected_techniques, steps,
                         technique_colors, linestyles, kernel_markers,
                         central_idx=5):
    """Plotta le curve di accuratezza per le tecniche selezionate,
       mostrando solo 2 valori prima e 2 dopo central_idx."""
    

    start_idx = 0
    end_idx = min(len(steps), central_idx + 3)  
    selected_steps = steps[start_idx:end_idx]   

    for t_idx, tech in enumerate(selected_techniques):
        base_color = technique_colors[t_idx]
        linestyle = linestyles[t_idx]

        tech_results = [r for r in results if r["technique"] == tech and r['kernel_size'] != 11]

        for r in tech_results:
            ks = r["kernel_size"]
            marker = kernel_markers.get(ks, "o")

            selected_curve = r["accuracy_curve"][start_idx:end_idx]

            ax.plot(
                selected_steps,
                selected_curve,
                color=base_color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=5,
                markevery=1  
            )

            if len(selected_steps) > 5:
                ax.set_xticks(selected_steps[::2])  
            else:
                ax.set_xticks(selected_steps)

    ax.set_ylabel("$\mathtt{F}$", fontsize=14)
    ax.set_xlabel("p", fontsize=14, labelpad=10)
    ax.xaxis.set_label_coords(0.5, -0.15)  

    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)


parser = argparse.ArgumentParser(description="Explanation analysis plot")
parser.add_argument("--corruption_type", type=str, default="gaussian_noise")
parser.add_argument("--agreement_measure", type=str, default="l1")
parser.add_argument("--normalization", type=str, default="quantil_local")
parser.add_argument("--models", nargs='+', default=["resnet50", "densenet121", "regnety_008"])
args = parser.parse_args()

datasets = ["dermamnist", "bloodmnist", "tissuemnist", "retinamnist",
            "pathmnist", "organcmnist", "octmnist", "pneumoniamnist", "breastmnist"]

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
# ------------------------------
# Plot settings
# ------------------------------
steps = np.linspace(0, 0.5, 11)

kernel_markers = {1: "o", 3: "s", 5: "D", 7: "^"}

technique_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#9467bd"}
linestyles = {0: "-", 1: "--", 2: ":"}


for model_name in args.models:
 
    all_values = []
    for dataset in datasets:
        file_path = f"results/{dataset}/fidelity_and_robustness/" \
                    f"{args.normalization}/{args.agreement_measure}/" \
                    f"{model_name}/{args.corruption_type}/results"
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r") as f:
            results = json.load(f)
        for r in results:
            all_values.extend(r["accuracy_curve"])
    ymin, ymax = min(all_values), max(all_values)
    margin = 0.02
    ymin -= margin
    ymax += margin

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        file_path = f"results/{dataset}/fidelity_and_robustness/" \
                    f"{args.normalization}/{args.agreement_measure}/" \
                    f"{model_name}/{args.corruption_type}/results"
        if not os.path.exists(file_path):
            print(f"File non trovato: {file_path}")
            ax.axis('off')
            continue

        with open(file_path, "r") as f:
            results = json.load(f)

        selected_techniques = select_best_techniques(results)
        central_idx = pixels[dataset][model_name]
        plot_accuracy_curves(ax, results, selected_techniques, steps,
                             technique_colors, linestyles, kernel_markers,central_idx)
        
        ax.set_title(dataset_names_tex[dataset], fontsize=14)

 
    for ax in axes[len(datasets):]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    legend_elements = []
    technique_labels = ["Best technique", "2nd best technique", "3rd best technique"]
    for i, label in enumerate(technique_labels):
        legend_elements.append(Line2D([0], [0], color=technique_colors[i],
                                      linestyle=linestyles[i], lw=3, label=label))
    for k, m in kernel_markers.items():
        legend_elements.append(Line2D([0], [0], color="black", marker=m,
                                      linestyle="None", markersize=7, label=f"Window size s = {k}"))

    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.09), ncol=5, fontsize=13, frameon=True)

    
    save_path = f"results/all_datasets_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {save_path}")
