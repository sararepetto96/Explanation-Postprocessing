import torch
import argparse
import json
from multiprocessing import Pool, get_context
from agreement import batch_agreement
import os
from collections import defaultdict
from compute_adversarial_robustness import preprocess_explanations
import numpy as np

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


def get_topk_methods(results, k,model,data_name):
    results_sorted = sorted(results, key=lambda r: r['accuracy_curve'][[pixels[data_name][model]][0]])
    return results_sorted[:k]

def load_explanations_npz(path: str) -> dict:
    data = np.load(path)
    return {k: torch.from_numpy(data[k]) for k in data.files}

def compute_mean_attributions(topk_entries,train_data_name, model_name,
                            device='cuda'):
    mean_attributions = defaultdict(list)
    mean_attributions_adversarial = defaultdict(list)

    for entry in topk_entries:
        tech, ksz = entry['technique'], entry['kernel_size']
        origin_path = f"original_explanations/{model_name}/{train_data_name}/{train_data_name}_test/{tech}_2.npz"
        adv_path = f"adversarial_explanations/{model_name}/{train_data_name}/{train_data_name}_test/{tech}_2.npz" 
        if not os.path.exists(origin_path) or not os.path.exists(adv_path):
            print("First, we need to compute adversarial robustness.")
            break

        attributions = load_explanations_npz(origin_path)
        attributions_adversarial = load_explanations_npz(adv_path)
        attributions = preprocess_explanations(attributions,kernel_size=ksz)
        attributions_adversarial = preprocess_explanations(attributions_adversarial,kernel_size=ksz)


        for img_id, attr in attributions.items():
            mean_attributions[img_id].append(attr.to(device))
        for img_id, attr in attributions_adversarial.items():
            mean_attributions_adversarial[img_id].append(attr.to(device))

    mean_attr_final = {img_id: torch.stack(attrs, dim=0).mean(0)
                       for img_id, attrs in mean_attributions.items()}
    mean_attr_corr_final = {img_id: torch.stack(attrs, dim=0).mean(0)
                            for img_id, attrs in mean_attributions_adversarial.items()}
    return mean_attr_final, mean_attr_corr_final

def main_ensemble(
    model_name: str = "resnet50",
    train_data_name: str = "dermamnist",
    agreement_measure: str = "l1",
    epsilon: int = 2,
    ensemble_type: str = "topk",
    post_processing: bool = True,
    device: str = "cuda",
):

    if post_processing and ensemble_type == "topk":
        results_file = f"results/{train_data_name}/fidelity_and_robustness/quantil_local/{agreement_measure}/{model_name}/adversarial_robustness_ensemble_topk_postprocessing_new"
    elif not post_processing and ensemble_type == "topk":
        results_file = f"results/{train_data_name}/fidelity_and_robustness/quantil_local/{agreement_measure}/{model_name}/adversarial_robustness_ensemble_topk_no_postprocessing_new"
    elif post_processing and ensemble_type == "mean_ensemble":
        results_file = f"results/{train_data_name}/fidelity_and_robustness/quantil_local/{agreement_measure}/{model_name}/adversarial_robustness_mean_ensemble_postprocessing_new"
    else:
        results_file = f"results/{train_data_name}/fidelity_and_robustness/quantil_local/{agreement_measure}/{model_name}/adversarial_robustness_mean_ensemble_no_postprocessing_new"

    if os.path.exists(results_file):
        print("Already existing results", flush=True)
        return

    adversarial_robustness = []

    file_path = f"results/{train_data_name}/fidelity_and_robustness/quantil_local/{agreement_measure}/{model_name}/adversarial_robustness_new_{epsilon}"
    fidelity_path = f"results/{train_data_name}/fidelity_and_robustness/quantil_local/{agreement_measure}/{model_name}/gaussian_noise/results"

    with open(file_path, "r") as f:
        results = json.load(f)

    with open(fidelity_path, "r") as f:
        results_fidelity = json.load(f)

    if not post_processing:
        results = [r for r in results if r["kernel_size"] == 1]
        results_fidelity = [r for r in results_fidelity if r["kernel_size"] == 1]

    if ensemble_type == "topk":
        results_top2 = get_topk_methods(results_fidelity, k=2, model=model_name, data_name=train_data_name)
        results_top3 = get_topk_methods(results_fidelity, k=3, model=model_name, data_name=train_data_name)

        orig2, adv2 = compute_mean_attributions(results_top2, train_data_name, model_name, device)
        orig3, adv3 = compute_mean_attributions(results_top3, train_data_name, model_name, device)

        l1_top2, _, _ = batch_agreement(orig2, adv2, agreement_function="l1",
                                        std=None, minimum=None, maximum=None, preprocessing=False)
        l1_top3, _, _ = batch_agreement(orig3, adv3, agreement_function="l1",
                                        std=None, minimum=None, maximum=None, preprocessing=False)

        adversarial_robustness.extend([
            {"technique": "top2", "kernel_size": "none", "adversarial_robustness": l1_top2},
            {"technique": "top3", "kernel_size": "none", "adversarial_robustness": l1_top3},
        ])

    else:
        orig, adv = compute_mean_attributions(results, train_data_name, model_name, device)
        l1_mean, _, _ = batch_agreement(orig, adv, agreement_function="l1",
                                        std=None, minimum=None, maximum=None, preprocessing=False)

        adversarial_robustness.append({
            "technique": "mean_ensemble",
            "kernel_size": "none",
            "adversarial_robustness": l1_mean,
        })

    with open(results_file, "w") as f:
        json.dump(adversarial_robustness, f, indent=4)

    print("Finished successfully", flush=True)

 

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='explanation analyzing')
    parser.add_argument('--model_name', type=str, default = 'resnet50')
    parser.add_argument('--train_data_name', type=str, default = 'dermamnist')
    parser.add_argument('--agreement_measure', type=str, default = 'l1')
    parser.add_argument('--n_classes', type=int, default = 7)
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--epsilon', type=int, default = 2)
    parser.add_argument('--ensemble_type', type=str, default = 'topk')
    parser.add_argument('--post_processing', action='store_true', help='Enable my flag')
    parser.set_defaults(post_processing=True)
    parser.add_argument('--no-post_processing', dest='post_processing', action='store_false', help='Disable my flag')

    args = parser.parse_args()
    main_ensemble(**vars(args))

        
