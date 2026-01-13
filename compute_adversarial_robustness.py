import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np
from ExplainableModels import ExplainableModel
import argparse
import re
from agreement import attributions_preprocessing
from agreement import batch_agreement
import json
import os



def topk_transformation(attributions: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    # if (C, H, W) → aggregate on channels
    if attributions.ndim == 3:
        attributions = np.mean(attributions, axis=0)
    original_shape = attributions.shape  # (H, W)
    # flatten
    flat = attributions.reshape(-1)
    # convert to torch
    flat_t = torch.from_numpy(flat)
    # number of pixel to preserve
    k = int(flat_t.numel() * ratio)
    # top-k in abs
    _, topk_indices = torch.topk(torch.abs(flat_t), k=k)
    # binary mask
    mask = torch.zeros_like(flat_t, dtype=torch.float)
    mask[topk_indices] = 1.0
    return mask.view(original_shape).numpy()

def load_images(data_name: str, model_name: str, epsilon: int):
    original = np.load(
        f'original_examples/{model_name}/{data_name}/{data_name}_test/DeepLift_{epsilon}.npz'
    )
    adversarial = np.load(
        f'adversarial_examples/{model_name}/{data_name}/{data_name}_test/DeepLift_{epsilon}.npz'
    )
    return original, adversarial

def extract_labels(keys):
    pattern = re.compile(r"/test/(\d+)/")
    return torch.tensor(
        [int(pattern.search(k).group(1)) for k in keys],
        dtype=torch.long
    )
def extract_images(images: dict):
    keys = list(images.keys())
    batch = torch.from_numpy(np.stack([images[k] for k in keys])).float()
    return batch, keys

def compute_explanations(
    model: ExplainableModel,
    technique: str,
    batch: torch.Tensor,
    labels: torch.Tensor,
    keys: list,
    data_name: str
) -> dict:
    
    with torch.no_grad():

        expl = model.applyXAI(
            algorithm=technique,
            input_tensor=batch,
            target_classes=labels,
            data_name=data_name,
            post_processing=False
        )

    return {k: expl[i].detach().cpu() for i, k in enumerate(keys)}


def preprocess_explanations(
    explanations: dict,
    kernel_size: int
) -> dict:

    return {
        k: attributions_preprocessing(
            v,
            agreement_measure='l1',
            std='None',
            kernel_size=kernel_size,
            normalization='quantil_local',
            maximum='None',
            minimum='None'
        )
        for k, v in explanations.items()
    }


def compute_adversarial_robustness(
    processed_original: dict,
    processed_adv: dict
) -> dict:

    zero_count = sum(
        np.all(v.cpu().detach().numpy() == 0)
        for v in processed_original.values()
    )

    l1_mean, l1_var, _ = batch_agreement(
        processed_original,
        processed_adv,
        agreement_function='l1',
        std=None,
        minimum=None,
        maximum=None,
        preprocessing=False
    )

    return {
        "adversarial_robustness": float(l1_mean),
        "var": float(l1_var),
        "count": int(zero_count)
    }

def compute_explanations_streaming(
    model: ExplainableModel,
    technique: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    keys: list,
    data_name: str,
    device="cuda",
    batch_size=1
) -> dict:

    explanations = {}

    model.model.to(device)
    model.model.eval()

    for i in range(0, len(images), batch_size):

        x = images[i:i+batch_size].to(device)
        y = labels[i:i+batch_size].to(device)
        k = keys[i:i+batch_size]

        expl = model.applyXAI(
            algorithm=technique,
            input_tensor=x,
            target_classes=y,
            data_name=data_name,
            post_processing=False
        )

        for j, key in enumerate(k):
            explanations[key] = expl[j].detach().cpu()

        del x, y, expl
        torch.cuda.empty_cache()

    return explanations


def plot_guided_gradcam_examples(
    original_image,
    adv_image,
    processed_original,
    processed_adv,
    original_expl_deeplift,
    adv_expl_deeplift,
    save_path
):
    keys = list(original_image.keys())[:7]

    fig, axes = plt.subplots(6, 7, figsize=(30, 12))
    row_titles = [
        "Original image",
        "Adversarial image",
        "Original DeepLift",
        "Adversarial DeepLift",
        "Original GuidedGradCam",
        "Adversarial GuidedGradCam",
    ]

    for col, key in enumerate(keys):
        images = [
            original_image[key],
            adv_image[key],
            topk_transformation(original_expl_deeplift[key]),
            topk_transformation(adv_expl_deeplift[key]),
            topk_transformation(processed_original[key].cpu().detach().numpy()),
            topk_transformation(processed_adv[key].cpu().detach().numpy()),
        ]

        for row, img in enumerate(images):
            ax = axes[row, col]
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.squeeze(0) if img.shape[0] == 1 else np.transpose(img, (1,2,0))
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_titles[row])


    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_explanations_npz(
    explanations: dict,
    save_path: str
):
    """
    explanations: dict[key -> torch.Tensor or np.ndarray]
    """
    save_dict = {
        k: v.cpu().numpy() if torch.is_tensor(v) else v
        for k, v in explanations.items()
    }
    np.savez_compressed(save_path, **save_dict)

def main(
        data_name,
        epsilon,
        model_name,
        n_classes,
    ):

    results_path = f'results/{data_name}/fidelity_and_robustness/quantil_local/l1/{model_name}/adversarial_robustness_new_{epsilon}'

    if not os.path.exists(results_path):

        techniques = [
            'GuidedGradCam',
            'DeepLift', 'GuidedBackprop',
            'GradientShap', 'SmoothGrad',
            'IntegratedGradients', 'Saliency', 'InputXGradient'
        ]

        kernel_sizes = [1, 3, 5, 7, 11]

        path = f"adversarial_examples/{model_name}/{data_name}/{data_name}_test/DeepLift_{epsilon}.npz"
        original_path = f"original_examples/{model_name}/{data_name}/{data_name}_test/DeepLift_{epsilon}.npz"

        if not os.path.exists(path) or not os.path.exists(original_path):
            
            explainableModel = ExplainableModel(model_name = model_name,
                                        train_data_name = data_name,
                                        n_classes = n_classes)
            explainableModel.attack(algorithm='DeepLift',
                                    data_name=data_name,
                                    batch_size=8,
                                    ε = epsilon,
                                    n_steps=100,
                                    new_process=False,
                                    dataset_subset_size=100,
                                    kernel_size = 1) 

        original_image, adv_image = load_images(data_name, model_name, epsilon)

        #original_expl_deeplift = np.load(
            #f'original_explanations/{args.model_name}/pneumoniamnist/pneumoniamnist_test/DeepLift_{args.epsilon}.npz'
        #)
        #adv_expl_deeplift = np.load(
            #f'adversarial_explanations/{args.model_name}/pneumoniamnist/pneumoniamnist_test/DeepLift_{args.epsilon}.npz'
        #)

        adv_batch, keys = extract_images(adv_image)
        original_batch, _ = extract_images(original_image)
        labels = extract_labels(keys)

        model = ExplainableModel(
            model_name=model_name,
            train_data_name=data_name,
            n_classes=n_classes
        )

        results = []

        for technique in techniques:

            original_expl = compute_explanations_streaming(
                model, technique, original_batch, labels, keys, data_name, batch_size=32
            )
            adv_expl = compute_explanations_streaming(
                model, technique, adv_batch, labels, keys, data_name, batch_size=32
            )

            save_dir_origin = f"original_explanations/{model_name}/{data_name}/{data_name}_test"
            save_dir_adv = f"adversarial_explanations/{model_name}/{data_name}/{data_name}_test"
            os.makedirs(save_dir_origin, exist_ok=True)
            os.makedirs(save_dir_adv, exist_ok=True)


            orig_path = f"{save_dir_origin}/{technique}_{epsilon}.npz"
            adv_path = f"{save_dir_adv}/{technique}_{epsilon}.npz"

            save_explanations_npz(original_expl, orig_path)
            save_explanations_npz(adv_expl, adv_path)

            for kernel_size in kernel_sizes:

                proc_orig = preprocess_explanations(original_expl, kernel_size)
                proc_adv = preprocess_explanations(adv_expl, kernel_size)

                metrics = compute_adversarial_robustness(proc_orig, proc_adv)

                results.append({
                    "technique": technique,
                    "kernel_size": kernel_size,
                    **metrics
                })
            
                if False:
                    path = f"{args.data_name}/fidelity_and_robustness/quantil_local/l1/{args.model_name}/adversarial_robustness_{args.epsilon}_{kernel_size}.png"
                    plot_guided_gradcam_examples(original_image=original_image,adv_image=adv_image, processed_original=proc_orig,
                                                processed_adv=proc_adv,original_expl_deeplift=original_expl_deeplift,
                                                adv_expl_deeplift=adv_expl_deeplift,save_path=path)

            
            del original_expl
            del adv_expl
            torch.cuda.empty_cache()

        with open(
            f"results/{data_name}/fidelity_and_robustness/quantil_local/l1/{model_name}/adversarial_robustness_new_{epsilon}",
            "w") as f:
            json.dump(results, f, indent=4)
    else:
        print('Results already computed')
        


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Explaining model')
    argparser.add_argument("--data_name", type=str, help='data name', default='dermamnist')
    argparser.add_argument("--epsilon", type=int, help='attack', default=2)
    argparser.add_argument("--model_name", type=str, help='method', default='resnet50')
    argparser.add_argument("--n_classes", type=int, help='method', default=7)
    args = argparser.parse_args()

    main(**vars(args))



            

            

                
                



