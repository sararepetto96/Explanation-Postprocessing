import json 
from pathlib import Path


# Mappatura per i nomi LaTeX
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

expl_name_tex = {
    "DeepLift": "DL",
    "IntegratedGradients": "IG",
    "GuidedBackprop": "GB",
    "GuidedGradCam":"GGC",
    "SmoothGrad":"SG",
    "GradientShap":"GS",
    "InputXGradient":"ixG",
    "Saliency":"SA"

}


data_robustness_bri = {}
data_robustness_pix = {}
best_techniques= {}

for model_name in ['resnet50', 'densenet121', 'regnety_008']:
    data_robustness_bri[model_names_tex[model_name]] = {}
    data_robustness_pix[model_names_tex[model_name]] = {}
    best_techniques[model_names_tex[model_name]]= {}

    for data_name in ['tissuemnist', 'retinamnist','dermamnist','pneumoniamnist','octmnist', 'organcmnist','breastmnist','pathmnist','bloodmnist']:
    #for data_name in [ 'bloodmnist','pathmnist']:
        
        base_path = Path(f"results/{data_name}/fidelity_and_robustness/quantil_local/l1/{model_name}")
        results_file_bri = base_path / "brightness_up/results"
        results_file_pix = base_path / "pixelate/results"
        results_file_adv = Path(f"results/{data_name}/fidelity_and_robustness/quantil_local/l1/{model_name}/adversarial_robustness_new_2")
        # Valori di default se mancano i file
        list_results = [None] * 6

        # Se il file non esiste, continuo
        if not results_file_bri.exists():
            print(f"⚠️ File mancante: {results_file_bri}")
            data_robustness_bri[model_names_tex[model_name]][data_name] = list_results
            continue

        if not results_file_pix.exists():
            print(f"⚠️ File mancante: {results_file_pix}")
            data_robustness_pix[model_names_tex[model_name]][data_name] = list_results
            continue


        # Leggi file JSON
        with open(results_file_bri, 'r') as f:
            results_bri = json.load(f)

        with open(results_file_pix, 'r') as f:
            results_pix = json.load(f)

        # Controllo che ci siano abbastanza risultati
        if len(results_bri) == 0:
            print(f"⚠️ Nessun risultato in: {results_file_bri}")
            data_robustness_bri[model_names_tex[model_name]][data_name] = list_results
            continue
        if len(results_pix) == 0:
            print(f"⚠️ Nessun risultato in: {results_file_pix}")
            data_robustness_pix[model_names_tex[model_name]][data_name] = list_results
            continue

        # Separazione post-processed / non post-processed
        results_no_postprocessed_bri = [r for r in results_bri if r['kernel_size'] == 1]
        results_postprocessed_bri = [r for r in results_bri if r['kernel_size'] != 1]

        results_no_postprocessed_pix = [r for r in results_pix if r['kernel_size'] == 1]
        results_postprocessed_pix = [r for r in results_pix if r['kernel_size'] != 1]

        # Ordinamento
        results_sorted = sorted(results_postprocessed_bri, key=lambda x: x["accuracy_curve"][pixels[data_name][model_name]])
        results_sorted_no_postprocessed = sorted(results_no_postprocessed_bri, key=lambda x: x["accuracy_curve"][pixels[data_name][model_name]])

        results_sorted_pix = sorted(results_postprocessed_pix, key=lambda x: x["accuracy_curve"][pixels[data_name][model_name]])
        results_sorted_no_postprocessed_pix = sorted(results_no_postprocessed_pix, key=lambda x: x["accuracy_curve"][pixels[data_name][model_name]])

        # Prendo fino a 3
        top3_post = results_sorted[:3]
        top3_no_post = results_sorted_no_postprocessed[:3]

        top3_post_pix = results_sorted_pix[:3]
        top3_no_post_pix = results_sorted_no_postprocessed_pix[:3]
    
        def get_matching_post(lst_no_post, i, results_postprocessed):

            if i >= len(lst_no_post):
                return None

            technique = lst_no_post[i]["technique"]

            # solo post con stessa tecnica
            matching_posts = [
                r for r in results_postprocessed
                if r["technique"] == technique
            ]

            if not matching_posts:
                return None

            # ordina per PEGGIORE fidelity
            matching_posts_sorted = sorted(
                matching_posts,
                key=lambda x: x["accuracy_curve"][pixels[data_name][model_name]]
            )

            worst_post = matching_posts_sorted[0]

            return round(worst_post["robustness"], 3)

            

        # Riempio list_results in modo sicuro
        def safe_get(lst, i):
            return round(lst[i]['robustness'], 3) if i < len(lst) else None
            
        best_techniques[model_names_tex[model_name]][data_name]=[expl_name_tex[top3_no_post[0]['technique']],expl_name_tex[top3_no_post[1]['technique']],expl_name_tex[top3_no_post[2]['technique']]]
        
        list_results_robustness_bri = [
            safe_get(top3_no_post, 0),
            get_matching_post(top3_no_post, 0, results_postprocessed_bri),
            safe_get(top3_no_post, 1),
            get_matching_post(top3_no_post, 1, results_postprocessed_bri),
            safe_get(top3_no_post, 2),
            get_matching_post(top3_no_post, 2, results_postprocessed_bri),
        ]
       
        data_robustness_bri[model_names_tex[model_name]][data_name] = list_results_robustness_bri

        list_results_robustness_pix = [
            safe_get(top3_no_post_pix, 0),
            get_matching_post(top3_no_post_pix, 0, results_postprocessed_pix),
            safe_get(top3_no_post_pix, 1),
            get_matching_post(top3_no_post_pix, 1, results_postprocessed_pix),
            safe_get(top3_no_post_pix, 2),
            get_matching_post(top3_no_post_pix, 2, results_postprocessed_pix),
        ]
       
        data_robustness_pix[model_names_tex[model_name]][data_name] = list_results_robustness_pix
        
        

def format_fidelity(value, is_best=False, is_better_than_best=False):
    if value is None:
        return "--"

    val_str = f"{value:.3f}" if isinstance(value, float) else str(value)

    if is_best and is_better_than_best:
        return rf"\textbf{{\uline{{{val_str}}}}}"
    elif is_best:
        return rf"\textbf{{{val_str}}}"
    elif is_better_than_best:
        return rf"\uline{{{val_str}}}"
    else:
        return val_str
    

def generate_simple_latex_table(data):
    """
    data: dict structured as {model: {dataset: [(E, F, R, A), ...]}}
    Genera una tabella LaTeX con solo E e due robuste (R^n_g, R^a)
    """

    header = r"""
    \begin{table*}[t]
    \centering
    \tiny
    \setlength{\tabcolsep}{0.1cm}
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{l||ccc|ccc||ccc|ccc||ccc|ccc}
    \toprule
    \multicolumn{1}{l||}{} &
    \multicolumn{3}{c|}{\textbf{best expl}} &
    \multicolumn{3}{c||}{\textbf{best expl post}} &
    \multicolumn{3}{c|}{\textbf{2nd best expl}} &
    \multicolumn{3}{c||}{\textbf{2nd best expl post}} &
    \multicolumn{3}{c|}{\textbf{3rd best expl}} &
    \multicolumn{3}{c}{\textbf{3rd best expl post}}\\
    Dataset &
    $\mathscr{E}$ & $\mathtt{R}^{n_g}$ & $\mathtt{R}^{a}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_g}$ & $\mathtt{R}^{a}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_g}$ & $\mathtt{R}^{a}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_g}$ & $\mathtt{R}^{a}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_g}$ & $\mathtt{R}^{a}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_g}$ & $\mathtt{R}^{a}$ \\
    \hline
    """

    body = ""

    for model, datasets_dict in data.items():
        body += f"\\multicolumn{{19}}{{c}}{{\\textbf{{\\emph{{{model}}}}}}} \\\\\n"
        body += "\\hline\n"

        for dataset, blocks in datasets_dict.items():
            row = [f"\\emph{{{dataset}}}"]

            for (E, R, A) in blocks:
                row += [
                    E if E is not None else "--",
                    R if R is not None else "--",
                    A if A is not None else "--",
                ]

            body += " & ".join(map(str, row)) + r" \\" + "\n"

        body += "\\hline\n"

    footer = r"""\bottomrule
    \end{tabular}
    \end{table*}"""

    return header + body + footer



def generate_unified_table(data):
    header = r"""
    \begin{tabular}{l||ccc|ccc||ccc|ccc||ccc|ccc}
    \toprule
    \multicolumn{1}{l||}{} &
    \multicolumn{3}{c|}{\textbf{best expl}} &
    \multicolumn{3}{c||}{\textbf{best expl post}} &
    \multicolumn{3}{c|}{\textbf{2nd best expl}} &
    \multicolumn{3}{c||}{\textbf{2nd best expl post}} &
    \multicolumn{3}{c|}{\textbf{3rd best expl}} &
    \multicolumn{3}{c}{\textbf{3rd best expl post}}\\
    Dataset &
    $\mathscr{E}$ & $\mathtt{R}^{n_b}$ & $\mathtt{R}^{p}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_b}$ & $\mathtt{R}^{p}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_b}$ & $\mathtt{R}^{p}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_b}$ & $\mathtt{R}^{p}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_b}$ & $\mathtt{R}^{p}$ &
    $\mathscr{E}$ & $\mathtt{R}^{n_b}$ & $\mathtt{R}^{p}$ \\
    \hline
    """
    body = ""

    for model, datasets in data.items():
        body += f"\\multicolumn{{19}}{{c}}{{\\textbf{{{model}}}}} \\\\\n"
        body += "\\hline\\hline\n"

        for dataset, blocks in datasets.items():

            fidelities = [F for (_, F, _, _) in blocks if F is not None]
            best_fid = min(fidelities) if fidelities else None

            best_expl_fid = blocks[0][1]  # best expl raw

            row = [dataset]

            for (E, F, R, A) in blocks:

                is_best = (F == best_fid)
                is_better_than_best = (
                    best_expl_fid is not None and
                    F is not None and
                    F < best_expl_fid
                )

                row += [
                    E if E is not None else "--",
                    format_fidelity(F, is_best, is_better_than_best),
                    R if R is not None else "--",
                    A if A is not None else "--",
                ]

            body += " & ".join(map(str, row)) + r" \\" + "\n"


    footer = r"""
    \end{tabular}}
    \end{table*}
    """
    return header + body + footer

data_unified = {}


for model in data_robustness_bri.keys():
    data_unified[model] = {}

    for dataset in data_robustness_bri[model].keys():
        rob_bri = data_robustness_bri[model][dataset]
        rob_pix = data_robustness_pix[model][dataset]

        # tecniche best / II / III (non post)
        techniques = best_techniques[model][dataset]

        # le tecniche stanno in top3_no_post → le hai implicitamente
        # recuperiamole dai risultati fidelity (kernel_size == 1)
        # qui assumiamo che l'ordine sia coerente
        # se vuoi essere ultra-sicuro, posso aiutarti a rifinirlo
        #techniques = [
           # "IG", "IG", "IG",  # placeholder: sostituisci se vuoi il nome reale
        #]

        row = []

        for i in range(3):
            # non post
            row.append((
                techniques[i],
                rob_bri[2*i],
                rob_pix[2*i],
            ))
            # post
            row.append((
                techniques[i],
                rob_bri[2*i + 1],
                rob_pix[2*i + 1],

            ))

        data_unified[model][dataset] = row



latex_code = generate_simple_latex_table(data_unified)

with open("results/tabella_postprocessing_natural_noise.tex", "w") as f:
    f.write(latex_code)
output_filename = "results/tabella_postprocessing.tex"

print(latex_code)