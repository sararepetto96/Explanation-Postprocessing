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




data_fidelity = {}
data_robustness = {}
data_adv_robustness = {}
best_techniques= {}

for model_name in ['resnet50', 'densenet121', 'regnety_008']:
    data_fidelity[model_names_tex[model_name]] = {}
    data_robustness[model_names_tex[model_name]] = {}
    data_adv_robustness[model_names_tex[model_name]] = {}
    best_techniques[model_names_tex[model_name]]= {}

    for data_name in ['tissuemnist', 'retinamnist','dermamnist','pneumoniamnist','octmnist', 'organcmnist','breastmnist','pathmnist','bloodmnist']:
    #for data_name in [ 'bloodmnist','pathmnist']:
        
        base_path = Path(f"results/{data_name}/fidelity_and_robustness/quantil_local/l1/{model_name}/gaussian_noise")
        results_file = base_path / "results"
        results_file_adv = Path(f"results/{data_name}/fidelity_and_robustness/quantil_local/l1/{model_name}/adversarial_robustness_new_2")
        # Valori di default se mancano i file
        list_results = [None] * 6

        # Se il file non esiste, continuo
        if not results_file.exists():
            print(f"⚠️ File mancante: {results_file}")
            data_fidelity[model_names_tex[model_name]][data_name] = list_results
            data_robustness[model_names_tex[model_name]][data_name] = list_results
            continue


        # Leggi file JSON
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Controllo che ci siano abbastanza risultati
        if len(results) == 0:
            print(f"⚠️ Nessun risultato in: {results_file}")
            data_fidelity[model_names_tex[model_name]][data_name] = list_results
            data_robustness[model_names_tex[model_name]][data_name] = list_results
            data_adv_robustness[model_names_tex[model_name]][data_name] = list_results
            continue

        # Separazione post-processed / non post-processed
        results_no_postprocessed = [r for r in results if r['kernel_size'] == 1]
        results_postprocessed = [r for r in results if r['kernel_size'] != 1]

        # Ordinamento
        results_sorted = sorted(results_postprocessed, key=lambda x: x["accuracy_curve"][pixels[data_name][model_name]])
        results_sorted_no_postprocessed = sorted(results_no_postprocessed, key=lambda x: x["accuracy_curve"][pixels[data_name][model_name]])

        # Prendo fino a 3
        top3_post = results_sorted[:3]
        top3_no_post = results_sorted_no_postprocessed[:3]
    
        def get_matching_post(lst_no_post, i, results_postprocessed, value):

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

            if value == "fidelity":
                return round(
                    worst_post["accuracy_curve"][pixels[data_name][model_name]],
                    3
                )

            if value == "robustness":
                return round(worst_post["robustness"], 3)

            if value == "adv_robustness":
                return round(
                    next(
                        (
                            r["adversarial_robustness"]
                            for r in results_adv
                            if r["technique"] == worst_post["technique"]
                            and r["kernel_size"] == worst_post["kernel_size"]
                        ),
                        None
                    ),
                    3
                )

        # Riempio list_results in modo sicuro
        def safe_get(lst, i,value):
            if value=='fidelity':
                return round(lst[i]['accuracy_curve'][pixels[data_name][model_name]], 3) if i < len(lst) else None
            if value=='robustness':
                return round(lst[i]['robustness'], 3) if i < len(lst) else None
            if value=='adv_robustness':
                technique = lst[i]['technique']
                kernel_size = lst[i]['kernel_size']
                return round(
                            next(
                                (
                                    r["adversarial_robustness"]
                                    for r in results_adv
                                    if r["technique"] == technique and r["kernel_size"] == kernel_size
                                ),
                                None
                            ),
                            3
                        ) if i < len(lst) else None
                #return round([r["adversarial_robustness"] for r in results_adv if r["technique"] == technique and r["kernel_size"] == kernel_size][0],3)
            


        #list_results_fidelity = [
            #safe_get(top3_no_post, 0,'fidelity'),
            #safe_get(top3_post, 0,'fidelity'),
            #safe_get(top3_no_post, 1,'fidelity'),
            #safe_get(top3_post, 1,'fidelity'),
            #safe_get(top3_no_post, 2,'fidelity'),
            #safe_get(top3_post, 2,'fidelity'),
        #]
        list_results_fidelity = [
            safe_get(top3_no_post, 0, 'fidelity'),
            get_matching_post(top3_no_post, 0, results_postprocessed, 'fidelity'),
            safe_get(top3_no_post, 1, 'fidelity'),
            get_matching_post(top3_no_post, 1, results_postprocessed, 'fidelity'),
            safe_get(top3_no_post, 2, 'fidelity'),
            get_matching_post(top3_no_post, 2, results_postprocessed, 'fidelity'),
        ]
    
        best_techniques[model_names_tex[model_name]][data_name]=[expl_name_tex[top3_no_post[0]['technique']],expl_name_tex[top3_no_post[1]['technique']],expl_name_tex[top3_no_post[2]['technique']]]
        


        list_results_robustness = [
            safe_get(top3_no_post, 0, 'robustness'),
            get_matching_post(top3_no_post, 0, results_postprocessed, 'robustness'),
            safe_get(top3_no_post, 1, 'robustness'),
            get_matching_post(top3_no_post, 1, results_postprocessed, 'robustness'),
            safe_get(top3_no_post, 2, 'robustness'),
            get_matching_post(top3_no_post, 2, results_postprocessed, 'robustness'),
        ]
        data_fidelity[model_names_tex[model_name]][data_name] = list_results_fidelity
        data_robustness[model_names_tex[model_name]][data_name] = list_results_robustness

        if not results_file_adv.exists():
            print(f"⚠️ File mancante: {results_file_adv}")
            data_adv_robustness[model_names_tex[model_name]][data_name] = list_results
            continue

        with open(results_file_adv, 'r') as f:
            results_adv = json.load(f)

        list_results_robustness_adv = [
        safe_get(top3_no_post, 0, 'adv_robustness'),
        get_matching_post(top3_no_post, 0, results_postprocessed, 'adv_robustness'),
        safe_get(top3_no_post, 1, 'adv_robustness'),
        get_matching_post(top3_no_post, 1, results_postprocessed, 'adv_robustness'),
        safe_get(top3_no_post, 2, 'adv_robustness'),
        get_matching_post(top3_no_post, 2, results_postprocessed, 'adv_robustness'),
    ]
        
        data_adv_robustness[model_names_tex[model_name]][data_name] = list_results_robustness_adv

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


def generate_unified_table(data):
    header = r"""
    \begin{table*}[t]
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{l|cccc|cccc|cccc|cccc|cccc|cccc}
    \toprule
    \multicolumn{1}{l}{} &
    \multicolumn{4}{c|}{\uline{best expl}} &
    \multicolumn{4}{c|}{\uline{best expl post}} &
    \multicolumn{4}{c|}{\uline{II best expl}} &
    \multicolumn{4}{c}{\uline{II best expl post}} &
    \multicolumn{4}{c|}{\uline{III best expl}} &
    \multicolumn{4}{c}{\uline{III best expl post}}\\
    Dataset &
    $\mathscr{E}$ &$\mathtt{F}$ & $\mathtt{R}$ & $\tilde{R}$ &
    $\mathscr{E}$ &$\mathtt{F}$ & $\mathtt{R}$ & $\tilde{R}$ &
    $\mathscr{E}$ &$\mathtt{F}$ & $\mathtt{R}$ & $\tilde{R}$ &
    $\mathscr{E}$ &$\mathtt{F}$ & $\mathtt{R}$ & $\tilde{R}$ &
    $\mathscr{E}$ &$\mathtt{F}$ & $\mathtt{R}$ & $\tilde{R}$ &
    $\mathscr{E}$ &$\mathtt{F}$ & $\mathtt{R}$ & $\tilde{R}$ \\
    \hline\hline
    """
    body = ""

    for model, datasets in data.items():
        body += f"\\multicolumn{{25}}{{c}}{{\\textbf{{{model}}}}} \\\\\n"
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


for model in data_fidelity.keys():
    data_unified[model] = {}

    for dataset in data_fidelity[model].keys():
        fid = data_fidelity[model][dataset]
        rob = data_robustness[model][dataset]
        adv = data_adv_robustness[model][dataset]

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
                fid[2*i],
                rob[2*i],
                adv[2*i]
            ))
            # post
            row.append((
                techniques[i],
                fid[2*i + 1],
                rob[2*i + 1],
                adv[2*i + 1]
            ))

        data_unified[model][dataset] = row



latex_code = generate_unified_table(data_unified)

with open("tabella_postprocessing_unificata.tex", "w") as f:
    f.write(latex_code)
output_filename = "results/tabella_postprocessing.tex"
breakpoint()
print(latex_code)