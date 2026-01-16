import torch
import numpy as np
from tqdm import tqdm
import os

def build_imagenet_path_map(base_path):
    path_map = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.JPEG'):
                image_code = os.path.splitext(file)[0]  # remove extension
                full_path = os.path.join(root, file)
                path_map[image_code] = full_path
    return path_map


def obtaining_std(att_1):
    all_values = []
    for final_img_path, tensor in att_1.items():
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, device='cuda')
        else:
            tensor = tensor.detach().flatten().to('cuda')
        all_values.append(tensor)
    
    all_values = torch.cat(all_values)
   
    std = torch.std(all_values)
    minimum = np.quantile(all_values.cpu().numpy(), 0.10)
    maximum = np.quantile(all_values.cpu().numpy(), 0.90)
    minimum = torch.tensor(minimum, device=all_values.device, dtype=all_values.dtype)
    maximum = torch.tensor(maximum, device=all_values.device, dtype=all_values.dtype)
    
    return std,minimum,maximum

@torch.no_grad()
def compute_means_fast(dataloader, device='cuda', num_classes=7):
    global_sum = torch.zeros(3, device=device)
    global_count = 0
    class_sums = torch.zeros(num_classes, 3, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    for images, labels, _ in tqdm(dataloader, desc="Computing means", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        b, c, h, w = images.shape
        global_sum += images.sum(dim=(0, 2, 3))
        global_count += b * h * w

        # Calculate mean per class in batch
        for cls in range(num_classes):
            mask = labels == cls
            if mask.any():
                selected = images[mask]
                class_sums[cls] += selected.sum(dim=(0, 2, 3))
                class_counts[cls] += selected.shape[0] * h * w

    global_mean = global_sum / global_count
    class_means = {
        cls: (class_sums[cls] / class_counts[cls]) if class_counts[cls] > 0 else global_mean
        for cls in range(num_classes)
    }
    return {'global': global_mean, 'per_class': class_means}


@torch.no_grad()
def occlusion_robustness_fast(
    dataloader, attributions, model, dataset_mean, 
    device='cuda', percentage=0.1, occlude_most_important=True,
    occlusion='all_mean'
):
    model.eval().to(device)
    pad = (21 - 1) // 2

    preds_all, labels_all, preds_orig_all = [], [], []

    for images, labels, names in tqdm(dataloader, desc="Occlusion robustness", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        occluded = images.clone()

        # Batch occlusion
        for i, name in enumerate(names):
            attr = attributions[name].to(device)
            h, w = attr.shape
            num_pixels = int(percentage *224 * 224)
            flat_attr = attr.flatten()

            # Important top/bottom indices
            indices = torch.argsort(flat_attr.abs(), descending=occlude_most_important)[:num_pixels]
            y_idx = indices // w
            x_idx = indices % w

            if occlusion == 'all_mean':
                fill_val = dataset_mean['global'].to(device)
            elif occlusion == 'class_mean':
                fill_val = dataset_mean['per_class'][labels[i].item()].to(device)
            elif occlusion == 'zero':
                fill_val = torch.zeros(3, device=device)
            else:
                raise ValueError("Invalid occlusion function")

            for c in range(3):
                occluded[i, c, y_idx + pad, x_idx + pad] = fill_val[c]
                if occlude_most_important==False:
                    occluded[i, c, :pad, :] = fill_val[c]
                    occluded[i, c, -pad:, :] = fill_val[c]
                    occluded[i, c, :, :pad] = fill_val[c]
                    occluded[i, c, :, -pad:] = fill_val[c]


        preds_orig = model(images).argmax(dim=1)
        preds_occ = model(occluded).argmax(dim=1)

        preds_all.append(preds_occ)
        preds_orig_all.append(preds_orig)
        labels_all.append(labels)

    preds_all = torch.cat(preds_all)
    preds_orig_all = torch.cat(preds_orig_all)
    labels_all = torch.cat(labels_all)

    acc = (preds_all == labels_all).float().mean().item()
    acc_original = (preds_orig_all == labels_all).float().mean().item()
    return acc, acc_original, acc_original - acc


@torch.no_grad()
def occlusion_robustness_curve_fast(
    dataloader, attributions, model, dataset_mean,
    device='cuda', occlusion_steps=np.linspace(0, 0.5, 11),
    occlude_most_important=True, occlusion='all_mean'
):
    model.eval().to(device)
    accuracies = []
    
    pad = (21 - 1) // 2

    for pct in tqdm(occlusion_steps, desc="Occlusion curve"):
        preds_all, labels_all = [], []

        for images, labels, names in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            occluded = images.clone()

            for i, name in enumerate(names):
                attr = attributions[name].to(device)
                h, w = attr.shape
                num_pixels = int(pct * 224 * 224)
                flat_attr = attr.flatten()
                indices = torch.argsort(flat_attr.abs(), descending=occlude_most_important)[:num_pixels]
                y_idx = indices // w
                x_idx = indices % w

                if occlusion == 'all_mean':
                    fill_val = dataset_mean['global'].to(device)
                elif occlusion == 'class_mean':
                    fill_val = dataset_mean['per_class'][labels[i].item()].to(device)
                elif occlusion == 'zero':
                    fill_val = torch.zeros(3, device=device)
                else:
                    raise ValueError("Invalid occlusion function")

                for c in range(3):
                    occluded[i, c, y_idx + pad, x_idx + pad] = fill_val[c]
                    if occlude_most_important==False:
                        occluded[i, c, :pad, :] = fill_val[c]
                        occluded[i, c, -pad:, :] = fill_val[c]
                        occluded[i, c, :, :pad] = fill_val[c]
                        occluded[i, c, :, -pad:] = fill_val[c]


            preds = model(occluded).argmax(dim=1)
            preds_all.append(preds)
            labels_all.append(labels)
        
        acc = (torch.cat(preds_all) == torch.cat(labels_all)).float().mean().item()
        accuracies.append(acc)
        
    return occlusion_steps, accuracies
