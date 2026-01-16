import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import os
import json

from ExplainableModels import ExplainableModel
import pandas as pd
from agreement import batch_agreement, attributions_preprocessing
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from pathlib import Path
from evaluation.fidelity import occlusion_robustness_curve_fast,occlusion_robustness_fast,obtaining_std,compute_means_fast
from collections import defaultdict
from tqdm import tqdm

def get_shaded_color(base_color, fraction):
    
    base_rgb = np.array(mcolors.to_rgb(base_color))
    target = np.array([1, 1, 1])  
    return base_rgb + (target - base_rgb) * fraction

def get_dataloader(data_name, data_split, batch_size):
    dataset = ExplainableModel.load_data(data_name, data_split)
    return DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)

def get_normalized_attributions(model, algorithm, data_name, data_split, batch_size,corruption_type,kernel_size,agreement_measure,normalization):
    train_data_name = data_name.split("corrupted")[0].strip()
    raw_attrs = model.explain_dataset(algorithm, train_data_name,data_name, data_split, batch_size,save_attributions=False)
    std,minimum, maximum  = obtaining_std(raw_attrs)
    data_name_corrupted = f'{data_name}_corrupted_{corruption_type}'
    raw_attrs_corrupted = model.explain_dataset(algorithm, train_data_name,data_name_corrupted, data_split, batch_size,save_attributions=False)
    return {img: attributions_preprocessing(attr,agreement_measure=agreement_measure,std=std,kernel_size=kernel_size,normalization=normalization,minimum =minimum, maximum= maximum) for img, attr in raw_attrs.items()},{img: attributions_preprocessing(attr_corrupted,agreement_measure='l1',std=std,kernel_size=kernel_size,normalization=normalization, minimum =minimum, maximum= maximum) for img, attr_corrupted in raw_attrs_corrupted.items()}

def get_single_algorithm_attributions(data_name, model_name, algorithm,corruption_type,kernel_size,agreement_measure, data_split='test', batch_size=512,normalization='quantil',n_classes=7):
    model = ExplainableModel(model_name=model_name, train_data_name=data_name, n_classes=n_classes)
    attrs,attrs_corrupted = get_normalized_attributions(model, algorithm, data_name, data_split, batch_size,corruption_type,kernel_size=kernel_size,agreement_measure=agreement_measure,normalization=normalization)
    dataloader = get_dataloader(data_name, data_split, batch_size)
    return attrs,attrs_corrupted, dataloader, model.model

def robustness_and_fidelity(data_name,model_name,algorithm,corruption_type,kernel_size,agreement_measure,data_split='test',occlude_most_important=True,aggregation_type='None',occlusion='class_mean',normalization='quantil',n_classes=7,ensemble_type = 'postprocessing'):
    attrs,attrs_corrupted, dl, model = get_single_algorithm_attributions(data_name, model_name, algorithm,corruption_type, kernel_size,agreement_measure,data_split,batch_size=512,normalization=normalization,n_classes=n_classes)

    base_path = Path(f"results/{data_name}/fidelity_and_robustness/{normalization}/{agreement_measure}/{model_name}/{corruption_type}")
    results_file = base_path / f"results"
    if results_file.parent.is_dir() and results_file.exists():
        with open(results_file, 'r') as f:
                results = json.load(f)
        robustness=[d["robustness"] for d in results if d["technique"] == algorithm and d["kernel_size"] == kernel_size].pop()
    else:
        robustness,_,_ = batch_agreement(attributions_test_1=attrs, attributions_test_2=attrs_corrupted,agreement_function=agreement_measure,preprocessing=False,std=0,minimum =0,maximum=0)
        
    # Check if occlusion curves already exist for any corruption type
    corruptions = ["pixelate", "brightness_up", "gaussian_noise"]
    base_path = Path(f"results/{data_name}/fidelity_and_robustness/{normalization}/{agreement_measure}/{model_name}")

    results_file = None
    for corruption in corruptions:
        candidate = base_path / corruption / f"results"
        if candidate.exists():
            results_file = candidate
            break

    if results_file is None:
        path = f'/data/srepetto/Explanation_variability/results/{data_name}/fidelity_and_robustness'
        os.makedirs(path, exist_ok=True)
        data_set_mean = compute_means_fast(dl, device='cuda', num_classes=n_classes)
        steps, accuracies = occlusion_robustness_curve_fast(dl, attrs, model, data_set_mean,occlude_most_important=occlude_most_important,
                                                        occlusion=occlusion)
    else:
        steps = np.linspace(0, 0.5, 11)
        with open(results_file, 'r') as f:
                results = json.load(f)
        accuracies = [d["accuracy_curve"] for d in results if d["technique"] == algorithm and d["kernel_size"] == kernel_size].pop()
    
    return steps,accuracies,robustness
    


def plot(data_name:str,model_name:str,corruption_type:str,agreement_measure:str,normalization:str, occluded_most_important:bool=True,n_classes:int=7):

    techniques = ['SmoothGrad',
        'IntegratedGradients', 'GuidedBackprop', 'DeepLift', 'Saliency',
        'InputXGradient', 'GradientShap','GuidedGradCam'
    ]
    
    kernel_sizes = [1, 3, 5, 7,11]
 
    results = []

    base_path = Path(f"results/{data_name}/fidelity_and_robustness/{normalization}/{agreement_measure}/{model_name}/{corruption_type}")
    results_file = base_path / f"results"

    if results_file.exists():
        print("✅  Results already present, loading from file...")
        with open(results_file, 'r') as f:
            results = json.load(f)
        results = [d for d in results if d['kernel_size'] not in [15,21]]
        
    else: 

        for technique in techniques:
            for k in kernel_sizes:
                steps,accs, rob = robustness_and_fidelity(
                    data_name=data_name,
                    model_name=model_name, 
                    algorithm=technique,
                    corruption_type=corruption_type,  
                    kernel_size=k,
                    agreement_measure=agreement_measure,
                    normalization=normalization,
                    occlude_most_important=occluded_most_important,
                    n_classes=n_classes
                )
                
                results.append({
                    'technique': technique,
                    'kernel_size': k,
                    'accuracy_curve': accs,
                    'robustness':rob
                })
                
    path =f'results/{data_name}/fidelity_and_robustness/{normalization}/{agreement_measure}/{model_name}/{corruption_type}'
    os.makedirs(path, exist_ok=True)
    file_name = os.path.join(path, f'results')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)
            
    colors = cm.get_cmap('tab10', len(techniques))

            
    kernel_markers = {
        1: 'o',   
        3: 'v',  
        5: 's',   
        7: 'D',   
        11: '^' 
    }
    steps=np.linspace(0, 0.5, 11)
   
    results_sorted = sorted(
    results,
    key=lambda r: (r['technique'], r['kernel_size'])
    )
    # Plot
    plt.figure(figsize=(16, 10))   

    for entry in results_sorted:
        technique = entry['technique']
        kernel_size = entry['kernel_size']
        acc_curve = entry['accuracy_curve']
        robustness = entry['robustness']
                    
        color_idx = techniques.index(technique)
        base_color = colors(color_idx)

        label = f"{technique} (k={kernel_size}) | R={robustness:.2f}"
        marker = kernel_markers.get(kernel_size, 'o')

        plt.plot(
            steps,
            acc_curve,
            label=label,
            color=base_color,
            linewidth=2.3,
            alpha=0.9,
            marker=marker,
            markersize=6,
            markevery=2
        )

    plt.title("Accuracy Curves with Occlusion\n(by Technique and Kernel Size)")
    plt.xlabel("Occlusion Step")
    plt.ylabel("Accuracy")
    plt.grid(True)

   
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=8,
        ncol=1,          
        borderaxespad=0.
    )

    plt.tight_layout(rect=[0, 0, 0.75, 1])

    plt.savefig(f'{path}/plot_results.png', dpi=300)
    plt.show()



@torch.no_grad()
def get_topk_per_step_fast(results, k=3):
    num_steps = len(results[0]['accuracy_curve'])
    occlusion_steps = np.linspace(0, 0.5, 11)
    topk_per_step = {}

    # Stack accuracies per step 
    for step in range(num_steps):
        step_values = [
            (r['accuracy_curve'][step], r['technique'], r['kernel_size'])
            for r in results
        ]
        step_values.sort(key=lambda x: x[0])  
        topk = [{'technique': t, 'kernel_size': ksz, 'accuracy': acc}
                for acc, t, ksz in step_values[:k]]
        topk_per_step[f'{occlusion_steps[step]}'] = topk
    return topk_per_step


@torch.no_grad()
def get_topk_robustness_fast(results, k=5):
    results_sorted = sorted(results, key=lambda r: r['robustness'])
    return results_sorted[:k]


@torch.no_grad()
def compute_mean_attributions_fast_old(topk_entries,data_name, model_name, corruption_type,
                              agreement_measure, normalization, device='cuda',n_classes =7):
    mean_attributions = defaultdict(list)
    mean_attributions_corrupted = defaultdict(list)
    dl, model = None, None
    print(n_classes)

    for entry in topk_entries:
        tech, ksz = entry['technique'], entry['kernel_size']

        attributions, attributions_corrupted, dl, model = get_single_algorithm_attributions(
            data_name=data_name,
            model_name=model_name,
            algorithm=tech,
            corruption_type=corruption_type,
            kernel_size=ksz,
            agreement_measure=agreement_measure,
            normalization=normalization,
            n_classes = n_classes
        )

        for img_id, attr in attributions.items():
            mean_attributions[img_id].append(attr.to(device))
        for img_id, attr in attributions_corrupted.items():
            mean_attributions_corrupted[img_id].append(attr.to(device))

    mean_attr_final = {img_id: torch.stack(attrs, dim=0).mean(0)
                       for img_id, attrs in mean_attributions.items()}
    mean_attr_corr_final = {img_id: torch.stack(attrs, dim=0).mean(0)
                            for img_id, attrs in mean_attributions_corrupted.items()}
    return mean_attr_final, mean_attr_corr_final, dl, model

@torch.no_grad()
def compute_mean_attributions_fast(topk_entries, data_name, model_name, corruption_type,
                                   agreement_measure, normalization, device='cuda', n_classes=7):

    mean_attr = {}
    mean_attr_corr = {}
    count_clean = defaultdict(int)
    count_corr = defaultdict(int)

    dl, model = None, None

    for entry in topk_entries:
        tech, ksz = entry['technique'], entry['kernel_size']

        attributions, attributions_corrupted, dl, model = get_single_algorithm_attributions(
            data_name=data_name,
            model_name=model_name,
            algorithm=tech,
            corruption_type=corruption_type,
            kernel_size=ksz,
            agreement_measure=agreement_measure,
            normalization=normalization,
            n_classes=n_classes,
        )

        for img_id, attr in attributions.items():
            attr = attr.to(device)
            count_clean[img_id] += 1

            if img_id not in mean_attr:
                mean_attr[img_id] = attr.clone()
            else:
                mean_attr[img_id] += (attr - mean_attr[img_id]) / count_clean[img_id]

        # ---- CORRUPTED ATTRIBUTIONS ----
        for img_id, attr in attributions_corrupted.items():
            attr = attr.to(device)
            count_corr[img_id] += 1

            if img_id not in mean_attr_corr:
                mean_attr_corr[img_id] = attr.clone()
            else:
                mean_attr_corr[img_id] += (attr - mean_attr_corr[img_id]) / count_corr[img_id]

        
    return mean_attr, mean_attr_corr, dl, model


@torch.no_grad()
def topk_robustness_evaluation_fast(
    model_name, corruption_type, data_name,
    occlude_most_important, aggregation_type, occlusion,
    results, k, agreement_measure, path, normalization,n_classes
):
    topk_rob = get_topk_robustness_fast(results, k)
    mean_attr, mean_attr_corr, dl, model = compute_mean_attributions_fast(topk_rob,data_name, model_name, corruption_type,
                            agreement_measure, normalization,n_classes=n_classes)

    mean_rob, var_rob, rob_img = batch_agreement(
        attributions_test_1=mean_attr,
        attributions_test_2=mean_attr_corr,
        agreement_function=agreement_measure,
        preprocessing=False,
        std=0,
        minimum=0,
        maximum=0
    )


    dataset_mean = compute_means_fast(dl, device='cuda',num_classes=n_classes)
    steps, accuracies = occlusion_robustness_curve_fast(
        dataloader=dl,
        attributions=mean_attr,
        model=model,
        dataset_mean=dataset_mean,
        occlude_most_important=occlude_most_important,
        occlusion=occlusion,
        device='cuda'
    )

    return topk_rob, accuracies, mean_rob


@torch.no_grad()
def topk_fidelity_evaluation_fast(
    model_name, corruption_type, data_name,
    occlude_most_important, aggregation_type, occlusion,
    results, k, agreement_measure, normalization,n_classes,ensemble_type
):
    topk_per_step = get_topk_per_step_fast(results, k=k)
    occlusion_steps = np.linspace(0, 0.5, len(next(iter(topk_per_step.values()))))
    accuracy_curve, robustness_curve = [], []

    for step_key in tqdm(topk_per_step.keys(), desc="Top-k fidelity evaluation"):
        topk_step = topk_per_step[step_key]
        mean_attr, mean_attr_corr, dl, model = compute_mean_attributions_fast(topk_step,data_name, model_name, corruption_type,
                              agreement_measure, normalization, n_classes=n_classes)

        mean_rob, var_rob, _ = batch_agreement(
            attributions_test_1=mean_attr,
            attributions_test_2=mean_attr_corr,
            agreement_function=agreement_measure,
            preprocessing=False,
            std=0,
            minimum=0,
            maximum=0
        )
        
        corruptions = ["pixelate", "brightness_up", "gaussian_noise"]
        base_path = Path(f"results/{data_name}/fidelity_and_robustness/{normalization}/{agreement_measure}/{model_name}")

        results_file = None
        for corruption in corruptions:
            candidate = base_path / corruption / f"results_ensemble_topk_{ensemble_type}"
            if candidate.exists():
                results_file = candidate
                break
        if results_file is None:
            pct = float(step_key)
            dataset_mean = compute_means_fast(dl, device='cuda',num_classes=n_classes)
            acc, acc_original, drop_acc = occlusion_robustness_fast(
                dataloader=dl,
                attributions=mean_attr,
                model=model,
                dataset_mean=dataset_mean,
                percentage=pct,
                occlude_most_important=occlude_most_important,
                occlusion=occlusion,
                device='cuda'
            )

            accuracy_curve.append(acc)

        else:
            with open(results_file, 'r') as f:
                results_old = json.load(f)
        
            if ensemble_type=='no_postprocessing':
                accuracy_curve = [d["accuracy_curve"] for d in results_old if d["technique"] == f'topk_fidelity{k}'].pop()
            else:
                accuracy_curve = [d["accuracy_curve"] for d in results_old if d["technique"] == f'topk_fidelity{k}_postprocessing'].pop()
        robustness_curve.append(mean_rob)

    return topk_per_step, accuracy_curve, mean_rob, robustness_curve

def mean_ensemble_evaluation_fast(
    model_name, corruption_type, data_name,
    occlude_most_important, aggregation_type, occlusion,
    results, agreement_measure, path, normalization,
    n_classes, ensemble_type='postprocessing', algorithm=None
):

    if ensemble_type == 'no_postprocessing':
        explanations = [r for r in results if r["kernel_size"] == 1]

    elif ensemble_type == 'postprocessing':
        explanations = results

    else:
        raise ValueError(f"ensemble_type non valido: {ensemble_type}")

    occlusion_steps = np.linspace(0, 0.5, 11)
    mean_attr, mean_attr_corr, dl, model = compute_mean_attributions_fast(
        explanations,
        data_name,
        model_name,
        corruption_type,
        agreement_measure,
        normalization,
        n_classes=n_classes
    )

    mean_rob, var_rob, rob_img = batch_agreement(
        attributions_test_1=mean_attr,
        attributions_test_2=mean_attr_corr,
        agreement_function=agreement_measure,
        preprocessing=False,
        std=0,
        minimum=0,
        maximum=0
    )

    corruptions = ["pixelate", "brightness_up", "gaussian_noise"]
    base_path = Path(f"results/{data_name}/fidelity_and_robustness/{normalization}/{agreement_measure}/{model_name}")
    results_file = None

    for corruption in corruptions:
        candidate = base_path / corruption / f"results_mean_ensemble_{ensemble_type}"
        if candidate.exists():
            results_file = candidate
            break

    if results_file is None:
        dataset_mean = compute_means_fast(dl, device='cuda', num_classes=n_classes)
        steps, accuracies = occlusion_robustness_curve_fast(
            dataloader=dl,
            attributions=mean_attr,
            model=model,
            dataset_mean=dataset_mean,
            occlude_most_important=occlude_most_important,
            occlusion=occlusion,
            device='cuda'
        )
    else:
        with open(results_file, 'r') as f:
            results_old = json.load(f)
        
        if ensemble_type == 'no_postprocessing':
            accuracies = [d["accuracy_curve"] for d in results_old if d["technique"] == 'mean_ensemble'].pop()
        elif ensemble_type == 'postprocessing':
            accuracies = [d["accuracy_curve"] for d in results_old if d["technique"] == 'mean_ensemble_postprocessing'].pop()

    return accuracies, mean_rob


def plot_with_topk(
    data_name: str,
    model_name: str,
    corruption_type: str,
    agreement_measure: str,
    batch_size: int = 32,
    data_split: str = 'test',
    occluded_most_important: bool = True,
    aggregation_type: str = 'None',
    occlusion: str = 'class_mean',
    normalization: str = 'quantil',
    n_classes: int = 7,
    ensemble_type: str = 'postprocessing'
):

    techniques = [
        'IntegratedGradients', 'GuidedBackprop', 'DeepLift', 'Saliency',
        'InputXGradient', 'SmoothGrad', 'GradientShap', 'GuidedGradCam'
    ]
    kernel_sizes = [1, 3, 5, 7, 11]
    results = []
    max_values = {}

    base_path = Path(f"results/{data_name}/fidelity_and_robustness/{normalization}/{agreement_measure}/{model_name}/{corruption_type}")
    results_file = base_path / f"results"

    if results_file.exists():
        print("✅ Results already present, loading from file...")
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        print("🧮 Calculating basic results...")
        for technique in techniques:
            for k in kernel_sizes:
                steps, accs, rob = robustness_and_fidelity(
                    data_name=data_name,
                    model_name=model_name,
                    algorithm=technique,
                    corruption_type=corruption_type,
                    kernel_size=k,
                    agreement_measure=agreement_measure,
                    occlude_most_important=occluded_most_important,
                    normalization=normalization,
                    n_classes=n_classes,
                    ensemble_type = ensemble_type
                )
                results.append({
                    'technique': technique,
                    'kernel_size': k,
                    'accuracy_curve': accs,
                    'robustness': rob
                })
        os.makedirs(base_path, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

    if ensemble_type == 'no_postprocessing':
        results = [r for r in results if r['kernel_size'] == 1]
        print("🧩 Ensemble type: NO SMOOTHING (only kernel=1)")
    else:
        print("🧩 Ensemble type: ALL (all kernel sizes and methods)")

    results_with_topk = []
    results_file_topk = base_path / f"results_ensemble_topk_{ensemble_type}"

    topk_robustness_total = {}
    topk_fidelity_total = {}
    robustness_curve_total = {}
    fidelity_total = {}
    robustness_total = {}

    max_values['postprocessing'] = 5
    max_values['no_postprocessing'] = 5
   

    if results_file_topk.exists():
        print("✅ Top-k results already present, loading...")
        with open(results_file_topk, 'r') as f:
            results_with_topk = json.load(f)
    else:
        print("⚙️ Calculating top-k results...")

        for k in (range(1,max_values[f'{ensemble_type}'])): 

                topk_fidelity_per_step, accuracy_f, robustness_f, robustness_curve = topk_fidelity_evaluation_fast(
                            model_name=model_name,
                            corruption_type=corruption_type,
                            data_name=data_name,
                            occlude_most_important=occluded_most_important,
                            aggregation_type=aggregation_type,
                            occlusion=occlusion,
                            results=results,
                            k=k,
                            agreement_measure=agreement_measure,
                            normalization=normalization,
                            n_classes=n_classes,
                            ensemble_type=ensemble_type
                        )

                
                        #top-k robustness
                topk_robustness, accuracy_r, robustness_r = topk_robustness_evaluation_fast(
                            model_name=model_name,
                            corruption_type=corruption_type,
                            data_name=data_name,
                            occlude_most_important=occluded_most_important,
                            aggregation_type=aggregation_type,
                            occlusion=occlusion,
                            results=results,
                            k=k,
                            agreement_measure=agreement_measure,
                            path=base_path,
                            normalization=normalization,
                            n_classes=n_classes
                        )

                 
                topk_robustness_total[f'{k}'] = topk_robustness
                topk_fidelity_total[f'{k}'] = topk_fidelity_per_step
                robustness_curve_total[f'{k}'] = robustness_curve

                fidelity_total[f'topk_fidelity{k}'] = accuracy_f
                robustness_total[f'topk_fidelity{k}'] = robustness_f
                fidelity_total[f'topk_robustness{k}'] = accuracy_r
                robustness_total[f'topk_robustness{k}'] = robustness_r

      
                if ensemble_type=='postprocessing':
                    results_with_topk.append({
                                'technique': f'topk_fidelity{k}_postprocessing',
                                'kernel_size': 'none',
                                'accuracy_curve': accuracy_f,
                                'robustness': robustness_f
                            })
                    results_with_topk.append({
                                'technique': f'topk_robustness{k}_postprocessing',
                                'kernel_size': 'none',
                                'accuracy_curve': accuracy_r,
                                'robustness': robustness_r
                            })
                else:
                    results_with_topk.append({
                                'technique': f'topk_fidelity{k}',
                                'kernel_size': 'none',
                                'accuracy_curve': accuracy_f,
                                'robustness': robustness_f
                            })
                    results_with_topk.append({
                                'technique': f'topk_robustness{k}',
                                'kernel_size': 'none',
                                'accuracy_curve': accuracy_r,
                                'robustness': robustness_r
                            })



        os.makedirs(base_path, exist_ok=True)

        with open(results_file_topk, 'w') as f:
            json.dump(results_with_topk, f, indent=4)

        with open(os.path.join(base_path, f'topk_robustness_total_{ensemble_type}'), 'w') as f:
            json.dump(topk_robustness_total, f, indent=4)

        with open(os.path.join(base_path, f'topk_fidelity_total_{ensemble_type}'), 'w') as f:
            json.dump(topk_fidelity_total, f, indent=4)

        with open(os.path.join(base_path, f'robustness_curve_total_{ensemble_type}'), 'w') as f:
            json.dump(robustness_curve_total, f, indent=4)


    techniques = sorted(set(r['technique'] for r in results_with_topk))
    path =f'results/{data_name}/fidelity_and_robustness/{normalization}/{agreement_measure}/{model_name}/{corruption_type}'

    kernel_sizes = [1, 3,5, 7,11]

    colors = cm.get_cmap('tab10', len(techniques))
    steps=np.linspace(0, 0.5, 11)

    kernel_markers = {
        1: 'o',   
        3: 'v',  
        5: 's',   
        7: 'D',   
        11: '^' 
    }

    fig, ax = plt.subplots(figsize=(18, 12)) 

 
    for entry in results_with_topk:
        technique = entry['technique']
        kernel_size = entry['kernel_size']
        acc_curve = entry['accuracy_curve']
        robustness = entry['robustness']

        color_idx = techniques.index(technique)
        base_color = colors(color_idx)

        if kernel_size != 'none':
            min_k, max_k = min(kernel_sizes), max(kernel_sizes)
            shade_fraction = 0.1 + 0.9 * (kernel_size - min_k) / (max_k - min_k + 1e-5)
            color = get_shaded_color(base_color, shade_fraction)
            label = f"{technique} (k={kernel_size}) | R={robustness:.2f}"
            marker = kernel_markers.get(kernel_size, 'o')
            ax.plot(
                steps, acc_curve, label=label, color=base_color,
                linewidth=2.0, alpha=0.9,
                marker=marker, markersize=5, markevery=2
            )
        else:
            label = f"{technique} (k={kernel_size}) | R={robustness:.2f}"

            ax.plot(steps, acc_curve, label=label, color=base_color, linewidth=2.0, alpha=0.9,linestyle='--')

    ax.set_title("Accuracy Curves with Occlusion\n(by Technique and Kernel Size)")
    ax.set_xlabel("Occlusion Step")
    ax.set_ylabel("Accuracy")
    ax.grid(True)

  
    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        fontsize=8,
        borderaxespad=0.,
        handlelength=1.5
    )
   
    plt.tight_layout(rect=[0, 0, 0.78, 1])  
    plt.savefig(f'{path}/plot_ensemble_topk_{ensemble_type}.png', dpi=300)


def plot_mean_ensemble(
    data_name: str,
    model_name: str,
    corruption_type: str,
    agreement_measure: str,
    batch_size: int = 32,
    data_split: str = 'test',
    occluded_most_important: bool = True,
    aggregation_type: str = 'None',
    occlusion: str = 'class_mean',
    normalization: str = 'quantil',
    n_classes: int = 7,
    ensemble_type: str = 'postprocessing'
):
    
    techniques = [
        'IntegratedGradients', 'GuidedBackprop', 'DeepLift', 'Saliency',
        'InputXGradient', 'SmoothGrad', 'GradientShap', 'GuidedGradCam'
    ]
    kernel_sizes = [1, 3, 5, 7, 11]
    results = []

    base_path = Path(f"results/{data_name}/fidelity_and_robustness/{normalization}/{agreement_measure}/{model_name}/{corruption_type}")
    results_file = base_path / f"results"

    if results_file.exists():
        print("✅ Results already available, loading from file..")
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        print("🧮 Calculating basic results...")
        for technique in techniques:
            for k in kernel_sizes:
                steps, accs, rob = robustness_and_fidelity(
                    data_name=data_name,
                    model_name=model_name,
                    algorithm=technique,
                    corruption_type=corruption_type,
                    kernel_size=k,
                    agreement_measure=agreement_measure,
                    occlude_most_important=occluded_most_important,
                    normalization=normalization,
                    n_classes=n_classes
                )
                results.append({
                    'technique': technique,
                    'kernel_size': k,
                    'accuracy_curve': accs,
                    'robustness': rob
                })

        os.makedirs(base_path, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

    
    results_with_topk = []
   
    results_file_ensemble = base_path / f"results_mean_ensemble_{ensemble_type}"

    if results_file_ensemble.exists():
        print("✅ Mean ensemble results already present, loading...")
        with open(results_file_ensemble, 'r') as f:
            results_with_topk = json.load(f)
    else:
        print(f"🧩 Mean ensemble calculation: {ensemble_type}")

        results_with_topk = []

        accuracy_r, robustness_r = mean_ensemble_evaluation_fast(
                model_name=model_name,
                corruption_type=corruption_type,
                data_name=data_name,
                occlude_most_important=occluded_most_important,
                aggregation_type=aggregation_type,
                occlusion=occlusion,
                results=results,
                agreement_measure=agreement_measure,
                path=base_path,
                normalization=normalization,
                n_classes=n_classes,
                ensemble_type=ensemble_type
            )
        
        name = {'postprocessing':'mean_ensemble_postprocessing', 'no_postprocessing':'mean_ensemble'}

        results_with_topk.append({
                'technique': name[ensemble_type],
                'kernel_size': 'none',
                'accuracy_curve': accuracy_r,
                'robustness': robustness_r
            })

        with open(results_file_ensemble, 'w') as f:
            json.dump(results_with_topk, f, indent=4)


    kernel_sizes = [1,3, 5, 7,11]
    techniques = sorted(set(r['technique'] for r in results_with_topk))

    colors = cm.get_cmap('tab10', len(techniques))
    steps=np.linspace(0, 0.5, 11)


    kernel_markers = {
        1: 'o',   
        3: 'v',  
        5: 's',  
        7: 'D',   
        11: '^' 
    }

    fig, ax = plt.subplots(figsize=(18, 12))  

 
    for entry in results_with_topk:
        technique = entry['technique']
        kernel_size = entry['kernel_size']
        acc_curve = entry['accuracy_curve']
        robustness = entry['robustness']

        color_idx = techniques.index(technique)
        base_color = colors(color_idx)

        if kernel_size != 'none':
            label = f"{technique} (k={kernel_size}) | R={robustness:.2f}"
            marker = kernel_markers.get(kernel_size, 'o')
            ax.plot(
                steps, acc_curve, label=label, color=base_color,
                linewidth=2.0, alpha=0.9,
                marker=marker, markersize=5, markevery=2
            )
        else:
            label = f"{technique} (k={kernel_size}) | R={robustness:.2f}"

            ax.plot(steps, acc_curve, label=label, color=base_color, linewidth=2.0, alpha=0.9,linestyle='--')

    ax.set_title("Accuracy Curves with Occlusion\n(by Technique and Kernel Size)")
    ax.set_xlabel("Occlusion Step")
    ax.set_ylabel("Accuracy")
    ax.grid(True)

    
    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        fontsize=8,
        borderaxespad=0.,
        handlelength=1.5
    )
   
    plt.tight_layout(rect=[0, 0, 0.78, 1]) 
    plt.savefig(f'{base_path}/plot_mean_ensemble_{ensemble_type}.png', dpi=300)
