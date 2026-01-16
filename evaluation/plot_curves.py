import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from matplotlib.lines import Line2D
from collections import defaultdict


def select_best_techniques(results, n_best=3):
    """
    Select best techniques based on accuracy_curve[2] with kernel_size = 1
    """
    results_k1 = [r for r in results if r["kernel_size"] == 1]
    results_k1 = sorted(results_k1, key=lambda r: r["accuracy_curve"][2])
    return [r["technique"] for r in results_k1[:n_best]]

def plot_accuracy_curves(ax, results, selected_techniques, steps,
                         technique_colors, linestyles, kernel_markers):

    for t_idx, tech in enumerate(selected_techniques):
        base_color = technique_colors[t_idx]
        linestyle = linestyles[t_idx]

        tech_results = [r for r in results if r["technique"] == tech and r['kernel_size']!=11]

        for r in tech_results:
            ks = r["kernel_size"]
            marker = kernel_markers.get(ks, "o")

            ax.plot(
                steps[:len(r["accuracy_curve"])],
                r["accuracy_curve"],
                color=base_color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=5,
                markevery=2,
            )

    ax.set_title("$\mathtt{F}$")
    ax.set_xlabel("p")
    ax.set_ylabel("F")
    ax.grid(True)

def collect_robustness_curves(file_paths, selected_techniques):
    """
    Build robustness curves across multiple corruption levels
    """
    robustness_curve = defaultdict(lambda: defaultdict(list))

    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            results = json.load(f)

        for r in results:
            if r["technique"] in selected_techniques and r['kernel_size']!=11:
                tech = r["technique"]
                ks = r["kernel_size"]
                robustness_curve[tech][ks].append(r["robustness"])

    return robustness_curve

def collect_adv_robustness_curves(file_paths, selected_techniques):
    """
    Build robustness curves across multiple corruption levels
    """
    robustness_curve = defaultdict(lambda: defaultdict(list))


    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            results = json.load(f)

        for r in results:
            if r["technique"] in selected_techniques and r['kernel_size']!=11:
                tech = r["technique"]
                ks = r["kernel_size"]
                robustness_curve[tech][ks].append(r["adversarial_robustness"])

    return robustness_curve

def plot_robustness_curves(ax, robustness_curve,
                           selected_techniques,
                           technique_colors, linestyles, kernel_markers,
                           x_axes):

    for t_idx, tech in enumerate(selected_techniques):
        base_color = technique_colors[t_idx]
        linestyle = linestyles[t_idx]

        for ks, curve in robustness_curve[tech].items():
            marker = kernel_markers.get(ks, "o")

            ax.plot(
                x_axes,
                curve,
                color=base_color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=5,
                markevery=2,
            )

    ax.set_ylabel("R")
    ax.grid(True)

def plot_adv_robustness_curves(ax, robustness_curve,
                           selected_techniques,
                           technique_colors, linestyles, kernel_markers):

    for t_idx, tech in enumerate(selected_techniques):
        base_color = technique_colors[t_idx]
        linestyle = linestyles[t_idx]

        for ks, curve in robustness_curve[tech].items():
            marker = kernel_markers.get(ks, "o")

            ax.plot(
                [2,4,6,8,10,12],
                curve,
                color=base_color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=5,
                markevery=2,
            )

    ax.set_ylabel("R")
    ax.grid(True)


# --------------------------------------------------
# Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Explanation analysis plot")
parser.add_argument("--corruption_type", type=str, default="gaussian_noise")
parser.add_argument("--model_name", type=str, default="resnet50")
parser.add_argument("--agreement_measure", type=str, default="l1")
parser.add_argument("--normalization", type=str, default="quantil_local")
parser.add_argument("--dataset", type=str, default="dermamnist")
args = parser.parse_args()


#metrics = ['Fidelity','Gaussian Noise','Pixelate','Brightness up','Adversarial Noise']
metrics = ['Fidelity','Gaussian Noise','Pixelate','Brightness up','Adversarial Noise']


title ={    
            'Gaussian Noise':r'$\mathtt{R}^{n_g}$',
            'Pixelate':r'$\mathtt{R}^{n_p}$',
            'Brightness up':r'$\mathtt{R}^{n_b}$',
            'Adversarial Noise':r'$\mathtt{R}^{a}$'}

x_title ={    
            'Gaussian Noise':'Variance',
            'Pixelate':'Resizing Factor',
            'Brightness up':'Brightness increase factor',
            'Adversarial Noise':r"$\varepsilon$"}

max_value ={
            'Gaussian Noise':6,
            'Pixelate':6,
            'Brightness up':6,
            'Adversarial Noise':12}

noise ={
            'Gaussian Noise':'gaussian_noise',
            'Pixelate':'pixelate',
            'Brightness up':'brightness_up'
        }

x_axes ={
            'Gaussian Noise':[0.012, 0.04,0.08, 0.12, 0.18, 0.26],
            'Pixelate':[0.8,0.7, 0.5, 0.40, 0.30, 0.25],
            'Brightness up':[0.9, 0.8, 0.7, 0.6, 0.5,0.4]
        }






# --------------------------------------------------
# Plot settings
# --------------------------------------------------
steps = np.linspace(0, 0.5, 11)

kernel_markers = {
    1: "o",
    3: "s",
    5: "D",
    7: "^"
}

technique_colors = {
    0: "#1f77b4",  # blue
    1: "#ff7f0e",  # orange
    2: "#9467bd" #purple,
}

linestyles ={
    0: "-",  
    1: "--",  
    2: ":" 
}


# --------------------------------------------------
# Create multi-panel figure
# --------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    # ---------------- Fidelity ----------------
    if metric == "Fidelity":
        file_path = f"results/{args.dataset}/fidelity_and_robustness/" \
                    f"{args.normalization}/{args.agreement_measure}/" \
                    f"{args.model_name}/{args.corruption_type}/results_0"

        with open(file_path, "r") as f:
            results = json.load(f)

        selected_techniques = select_best_techniques(results)
        plot_accuracy_curves(
            ax, results, selected_techniques, steps,
            technique_colors, linestyles, kernel_markers
        )

    # ---------------- Robustness metrics ----------------
    else:
        file_paths = []

        for i in range(max_value[metric]):
            if metric == "Adversarial Noise":
                if i % 2 == 0:
                    file_paths.append(
                        f"results/{args.dataset}/fidelity_and_robustness/"
                        f"{args.normalization}/{args.agreement_measure}/"
                        f"{args.model_name}/adversarial_robustness_new_{i+1}"
                    )
                else:
                    continue
            else:
                file_paths.append(
                    f"results/{args.dataset}/fidelity_and_robustness/"
                    f"{args.normalization}/{args.agreement_measure}/"
                    f"{args.model_name}/{noise[metric]}/results_{i}"
                )
        if metric == "Adversarial Noise":

            robustness_curve = collect_adv_robustness_curves(
                file_paths, selected_techniques
            )


            plot_adv_robustness_curves(
                ax, robustness_curve,
                selected_techniques,
                technique_colors, linestyles, kernel_markers
            )

            ax.set_title(r'$\mathtt{R}^{a}$')

        else:

            robustness_curve = collect_robustness_curves(
                file_paths, selected_techniques
            )

            plot_robustness_curves(
                ax, robustness_curve,
                selected_techniques,
                technique_colors, linestyles, kernel_markers,x_axes=x_axes[metric]
            )



        ax.set_title(title[metric])
        ax.set_xlabel(x_title[metric])


plt.tight_layout(rect=[0, 0.08, 1, 0.92])

legend_elements = []

technique_labels = [
    "Best technique",
    "2nd best technique",
    "3rd best technique",
]

for i, label in enumerate(technique_labels):
    legend_elements.append(
        Line2D(
            [0], [0],
            color=technique_colors[i],
            linestyle=linestyles[i],
            lw=3,
            label=label
        )
    )

for k, m in kernel_markers.items():
    legend_elements.append(
        Line2D(
            [0], [0],
            color="black",
            marker=m,
            linestyle="None",
            markersize=7,
            label=f"Window size s = {k}"
        )
    )


fig.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.00, 0.5),  
    ncol=1,
    fontsize=11,
    frameon=True
)
# --------------------------------------------------
# Save
# --------------------------------------------------


plt.savefig("results/all_datasets_best.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved plot: all_datasets_best.png")
