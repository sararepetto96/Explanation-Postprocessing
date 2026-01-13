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
data_fid = {}
data_rob = {}
data_adv = {}

# ============================================================
# DATA LOADING (COME NEL TUO CODICE)
# ============================================================
for model in ["resnet50", "densenet121", "regnety_008"]:
    model_tex = model_names_tex[model]
    data_fid[model_tex] = {}
    data_rob[model_tex] = {}
    data_adv[model_tex] = {}

    for d in dataset_names_tex:
        dataset_tex = dataset_names_tex[d]
        idx = pixels[d][model]

        base = Path(f"results/{d}/fidelity_and_robustness/quantil_local/l1/{model}")

        # ---------- Fidelity & robustness ----------
        res_post = load_json(base / "gaussian_noise/results")
        mean = load_json(base / "gaussian_noise/results_mean_ensemble_no_postprocessing")
        topk = load_json(base / "gaussian_noise/results_ensemble_topk_no_postprocessing")
        mean_post = load_json(base / "gaussian_noise/results_mean_ensemble_postprocessing")
        topk_post = load_json(base / "gaussian_noise/results_ensemble_topk_postprocessing")

        if not all([res_post, mean, topk, mean_post, topk_post]):
            continue

        res = [r for r in res_post if r.get("kernel_size") == 1]
        res_post = [r for r in res_post if r.get("kernel_size") != 1]
        res_sorted = sorted(res, key=lambda x: x["accuracy_curve"][idx])
        res_sorted_post = sorted(res_post, key=lambda x: x["accuracy_curve"][idx])

        fid = [
            round(res_sorted[0]["accuracy_curve"][idx], 3),
            round(mean[0]["accuracy_curve"][idx], 3),
            round(next(r for r in topk if r["technique"] == "topk_fidelity2")["accuracy_curve"][idx], 3),
            round(next(r for r in topk if r["technique"] == "topk_fidelity3")["accuracy_curve"][idx], 3),
            round(res_sorted_post[0]["accuracy_curve"][idx], 3),
            round(mean_post[0]["accuracy_curve"][idx], 3),
            round(next(r for r in topk_post if "topk_fidelity2" in r["technique"])["accuracy_curve"][idx], 3),
            round(next(r for r in topk_post if "topk_fidelity3" in r["technique"])["accuracy_curve"][idx], 3),
        ]

        rob = [
            round(res_sorted[0]["robustness"], 3),
            round(mean[0]["robustness"], 3),
            round(next(r for r in topk if r["technique"] == "topk_fidelity2")["robustness"], 3),
            round(next(r for r in topk if r["technique"] == "topk_fidelity3")["robustness"], 3),
            round(res_sorted_post[0]["robustness"], 3),
            round(mean_post[0]["robustness"], 3),
            round(next(r for r in topk_post if "topk_fidelity2" in r["technique"])["robustness"], 3),
            round(next(r for r in topk_post if "topk_fidelity3" in r["technique"])["robustness"], 3),
        ]

        # ---------- Adversarial robustness (FILE SEPARATI) ----------
        adv_res_post = load_json(base / "adversarial_robustness_new_2")
        adv_mean = load_json(base / "adversarial_robustness_mean_ensemble_no_postprocessing_new")
        adv_topk = load_json(base / "adversarial_robustness_ensemble_topk_no_postprocessing_new")
        adv_mean_post = load_json(base / "adversarial_robustness_mean_ensemble_postprocessing_new")
        adv_topk_post = load_json(base / "adversarial_robustness_ensemble_topk_postprocessing_new")

        if not all([adv_res_post, adv_mean, adv_topk, adv_mean_post, adv_topk_post]):
            continue

        

        adv_res = [r for r in adv_res_post if r.get("kernel_size") == 1]
        best_technique =res_sorted[0]['technique']
        best_kernel = res_sorted_post[0]['kernel_size']
        adv_res_post = [r for r in adv_res_post if r.get("kernel_size") != 1]
  
        best_adv_res =[r for r in adv_res if r.get("technique") == best_technique]
        best_adv_res_post =[r for r in adv_res_post if r.get("technique") == best_technique and r.get("kernel_size") == best_kernel]
        #adv_sorted = sorted(adv_res, key=lambda x: x["adversarial_robustness"])
        #adv_sorted_post = sorted(adv_res_post, key=lambda x: x["adversarial_robustness"])

        adv = [
            round(best_adv_res[0]["adversarial_robustness"], 3),
            round(adv_mean[0]["adversarial_robustness"], 3),
            round(next(r for r in adv_topk if r["technique"] == "top2")["adversarial_robustness"], 3),
            round(next(r for r in adv_topk if r["technique"] == "top3")["adversarial_robustness"], 3),
            round(best_adv_res_post[0]["adversarial_robustness"], 3),
            round(adv_mean_post[0]["adversarial_robustness"], 3),
            round(next(r for r in adv_topk_post if r["technique"] == "top2")["adversarial_robustness"], 3),
            round(next(r for r in adv_topk_post if r["technique"] == "top3")["adversarial_robustness"], 3),
        ]

        data_fid[model_tex][dataset_tex] = fid
        data_rob[model_tex][dataset_tex] = rob
        data_adv[model_tex][dataset_tex] = adv

# ============================================================
# TABLE GENERATION
# ============================================================
def generate_table(fid, rob, adv):
    tex = r"""
\begin{table*}[t]
\resizebox{\textwidth}{!}{
\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc|ccc|ccc|ccc}
\toprule
 & \multicolumn{3}{c|}{\uline{best expl}}
 & \multicolumn{3}{c|}{\uline{W1}}
 & \multicolumn{3}{c|}{\uline{W2-top2}}
 & \multicolumn{3}{c|}{\uline{W2-top3}}
 & \multicolumn{3}{c|}{\uline{best expl post}}
 & \multicolumn{3}{c|}{\uline{W1 post}}
 & \multicolumn{3}{c|}{\uline{W2-top2 post}}
 & \multicolumn{3}{c}{\uline{W2-top3 post}} \\
Dataset &
$F$ & $R$ & $\tilde R$ &
$F$ & $R$ & $\tilde R$ &
$F$ & $R$ & $\tilde R$ &
$F$ & $R$ & $\tilde R$ &
$F$ & $R$ & $\tilde R$ &
$F$ & $R$ & $\tilde R$ &
$F$ & $R$ & $\tilde R$ &
$F$ & $R$ & $\tilde R$ \\
\midrule
"""

    for model in fid:
        tex += rf"\multicolumn{{25}}{{c}}{{\textbf{{{model}}}}} \\ \midrule" + "\n"

        for ds in fid[model]:
            fvals = fid[model][ds]
            rvals = rob[model][ds]
            avals = adv[model][ds]

            min_f = min(fvals)
            best = fvals[0]
            best_post = fvals[4]

            row = [ds]

            for i in range(8):
                f = fvals[i]
                cell = f"{f:.3f}"

                if f == min_f:
                    cell = bold(cell)

                if (i in [1,2,3] and f < best) or (i in [5,6,7] and f < best_post):
                    cell = underline(cell)

                row += [
                    cell,
                    f"{rvals[i]:.3f}",
                    f"{avals[i]:.3f}",
                ]

            tex += " & ".join(row) + r" \\" + "\n"

    tex += r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    return tex

latex = generate_table(data_fid, data_rob, data_adv)

with open("results/tabella_ensemble.tex", "w") as f:
    f.write(latex)

print(latex)
