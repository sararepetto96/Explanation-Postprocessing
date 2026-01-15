import json
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
pixels = {
    "tissuemnist": {"densenet121": 4, "resnet50": 2, "regnety_008": 6},
    "retinamnist": {"densenet121": 1, "resnet50": 1, "regnety_008": 1},
    "dermamnist": {"densenet121": 2, "resnet50": 2, "regnety_008": 8},
    "pneumoniamnist": {"densenet121": 4, "resnet50": 3, "regnety_008": 4},
    "octmnist": {"densenet121": 3, "resnet50": 3, "regnety_008": 2},
    "organcmnist": {"densenet121": 2, "resnet50": 3, "regnety_008": 3},
    "breastmnist": {"densenet121": 7, "resnet50": 7, "regnety_008": 10},
    "bloodmnist": {"densenet121": 2, "resnet50": 2, "regnety_008": 2},
    "pathmnist": {"densenet121": 10, "resnet50": 8, "regnety_008": 8},
}

model_names_tex = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "regnety_008": "RegNet"
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

# ============================================================
# UTILS
# ============================================================
def load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

def bold(x): return f"\\textbf{{{x}}}"
def underline(x): return f"\\underline{{{x}}}"
def bold_underline(x): return f"\\textbf{{\\underline{{{x}}}}}"

# ============================================================
# DATA CONTAINER
# ============================================================
data_rob_bri = {}
data_rob_pix = {}

# ============================================================
# DATA LOADING (COME NEL TUO CODICE)
# ============================================================
for model in ["resnet50", "densenet121", "regnety_008"]:
    model_tex = model_names_tex[model]
    data_rob_bri[model_tex] = {}
    data_rob_pix[model_tex] = {}

    for d in dataset_names_tex:
        dataset_tex = dataset_names_tex[d]
        idx = pixels[d][model]

        base = Path(f"results/{d}/fidelity_and_robustness/quantil_local/l1/{model}")

        # ---------- Brightness robustness ----------
        res_post_bri = load_json(base / "brightness_up/results")
        mean_bri = load_json(base / "brightness_up/results_mean_ensemble_no_postprocessing")
        topk_bri = load_json(base / "brightness_up/results_ensemble_topk_no_postprocessing")
        mean_post_bri = load_json(base / "brightness_up/results_mean_ensemble_postprocessing")
        topk_post_bri = load_json(base / "brightness_up/results_ensemble_topk_postprocessing")

        if not all([res_post_bri, mean_bri, topk_bri, mean_post_bri, topk_post_bri]):
            continue

        res_bri = [r for r in res_post_bri if r.get("kernel_size") == 1]
        res_post_bri = [r for r in res_post_bri if r.get("kernel_size") != 1]
        res_sorted_bri = sorted(res_bri, key=lambda x: x["accuracy_curve"][idx])
        res_sorted_post_bri = sorted(res_post_bri, key=lambda x: x["accuracy_curve"][idx])

        # ---------- Pixelate robustness ----------
        res_post_pix = load_json(base / "pixelate/results")
        mean_pix  = load_json(base / "pixelate/results_mean_ensemble_no_postprocessing")
        topk_pix  = load_json(base / "pixelate/results_ensemble_topk_no_postprocessing")
        mean_post_pix  = load_json(base / "pixelate/results_mean_ensemble_postprocessing")
        topk_post_pix  = load_json(base / "pixelate/results_ensemble_topk_postprocessing")

        if not all([res_post_pix, mean_pix , topk_pix , mean_post_pix , topk_post_pix ]):
            continue

        res_pix  = [r for r in res_post_pix  if r.get("kernel_size") == 1]
        res_post_pix  = [r for r in res_post_pix  if r.get("kernel_size") != 1]
        res_sorted_pix  = sorted(res_pix , key=lambda x: x["accuracy_curve"][idx])
        res_sorted_post_pix  = sorted(res_post_pix , key=lambda x: x["accuracy_curve"][idx])


        rob_bri = [
            round(res_sorted_bri[0]["robustness"], 3),
            round(mean_bri[0]["robustness"], 3),
            round(next(r for r in topk_bri if r["technique"] == "topk_fidelity2")["robustness"], 3),
            round(next(r for r in topk_bri if r["technique"] == "topk_fidelity3")["robustness"], 3),
            round(res_sorted_post_bri[0]["robustness"], 3),
            round(mean_post_bri[0]["robustness"], 3),
            round(next(r for r in topk_post_bri if "topk_fidelity2" in r["technique"])["robustness"], 3),
            round(next(r for r in topk_post_bri if "topk_fidelity3" in r["technique"])["robustness"], 3),
        ]

        rob_pix = [
            round(res_sorted_pix[0]["robustness"], 3),
            round(mean_pix[0]["robustness"], 3),
            round(next(r for r in topk_pix if r["technique"] == "topk_fidelity2")["robustness"], 3),
            round(next(r for r in topk_pix if r["technique"] == "topk_fidelity3")["robustness"], 3),
            round(res_sorted_post_pix[0]["robustness"], 3),
            round(mean_post_pix[0]["robustness"], 3),
            round(next(r for r in topk_post_pix if "topk_fidelity2" in r["technique"])["robustness"], 3),
            round(next(r for r in topk_post_pix if "topk_fidelity3" in r["technique"])["robustness"], 3),
        ]

        data_rob_bri[model_tex][dataset_tex] = rob_bri
        data_rob_pix[model_tex][dataset_tex] = rob_pix
       

# ============================================================
# TABLE GENERATION
# ============================================================
def generate_table(rob_bri, rob_pix):
    tex = r"""
\begin{table*}[t]
\centering
\tiny
\setlength{\tabcolsep}{0.1cm}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{l||cc|cc|cc|cc||cc|cc|cc|cc}
\toprule
& \multicolumn{2}{c|}{\textbf{best expl}}
& \multicolumn{2}{c|}{\textbf{W1}}
& \multicolumn{2}{c|}{\textbf{W2-top2}}
& \multicolumn{2}{c||}{\textbf{W2-top3}}
& \multicolumn{2}{c|}{\textbf{best expl post}}
& \multicolumn{2}{c|}{\textbf{W1 post}}
& \multicolumn{2}{c|}{\textbf{W2-top2 post}}
& \multicolumn{2}{c}{\textbf{W2-top3 post}} \\
\textbf{Dataset} &
$\mathtt{R}^{n_b}$ & $\mathtt{R}^{n_p}$ &
$\mathtt{R}^{n_b}$ & $\mathtt{R}^{n_p}$ &
$\mathtt{R}^{n_b}$ & $\mathtt{R}^{n_p}$ &
$\mathtt{R}^{n_b}$ & $\mathtt{R}^{n_p}$ &
$\mathtt{R}^{n_b}$ & $\mathtt{R}^{n_p}$ &
$\mathtt{R}^{n_b}$ & $\mathtt{R}^{n_p}$ &
$\mathtt{R}^{n_b}$ & $\mathtt{R}^{n_p}$ &
$\mathtt{R}^{n_b}$ & $\mathtt{R}^{n_p}$ \\
\hline
"""

    for model in rob_bri:
        tex += rf"\multicolumn{{17}}{{c}}{{\textbf{{\emph{{{model}}}}}}} \\ \hline" + "\n"

        for ds in rob_bri[model]:
            rvals_bri = rob_bri[model][ds]
            rvals_pix = rob_pix[model][ds]

            row = [ds]

            for i in range(8):
                row.append(f"{rvals_bri[i]:.3f}")
                row.append(f"{rvals_pix[i]:.3f}")

            tex += " & ".join(row) + r" \\" + "\n"

    tex += r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    return tex

latex = generate_table(data_rob_bri, data_rob_pix)

with open("results/tabella_ensemble_natural_noise.tex", "w") as f:
    f.write(latex)

print(latex)
