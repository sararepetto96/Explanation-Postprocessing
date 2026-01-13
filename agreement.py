from typing import Tuple, Dict
import numpy as np
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F

def attributions_check(attributions_a: np.ndarray, attributions_b: np.ndarray):
    assert attributions_a.shape == attributions_b.shape, "attributions should have same shape"
    assert attributions_a.ndim == 3, "attributions should be 3D numpy arrays (3 channels images)"
    assert attributions_a.shape[0] == 3, "attributions should have 3 channels"
    assert attributions_a.shape[1] == 224 and attributions_a.shape[2] == 224, \
        "attributions should have 224x224 resolution"  #224x224 resolution

def attributions_preprocessing(attributions: np.ndarray,agreement_measure: str,std:int,kernel_size:int,normalization:str,minimum:int,maximum:int)-> np.ndarray:       

    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions, dtype=torch.float32)
    else:
        attributions = attributions.to(torch.float32)
    
    if normalization == 'naive':
        if std != 0:
            attributions = attributions / std
            attributions = torch.clamp(attributions, -50, 50)
    
    elif normalization == 'quantil_image_wise':
        if std != 0:
            attributions = attributions / std
        
        q10 = torch.quantile(attributions, 0.10)
        q90 = torch.quantile(attributions, 0.90)
        
        attributions = torch.clamp(attributions, q10, q90)

    elif normalization=='quantil_local':
        std = torch.std(attributions)
        if std!=0:
            attributions = attributions/std
        min = torch.quantile(attributions,0.10)
        max = torch.quantile(attributions,0.90)
        attributions = torch.clamp(attributions,min,max)

    elif normalization=='quantil_global':
        attributions = torch.clamp(attributions,minimum,maximum)
        if std!=0:
            attributions = attributions/std
    
    else:
        
        raise ValueError("Invalid normalization function.")
    
    if agreement_measure == "topk":

            attributions = patching(attributions)
    
    attributions = smooth_transformation(attributions,kernel_size=kernel_size)

    return attributions


def spearman(attributions_a: torch.Tensor, attributions_b: torch.Tensor) -> torch.Tensor:
    a = attributions_a.flatten()
    b = attributions_b.flatten()

    # Rank calculation
    ranks_a = torch.argsort(torch.argsort(a))
    ranks_b = torch.argsort(torch.argsort(b))

    # Pearson's rank correlation
    mean_a = ranks_a.float().mean()
    mean_b = ranks_b.float().mean()
    cov = ((ranks_a - mean_a) * (ranks_b - mean_b)).mean()
    std_a = ranks_a.float().std(unbiased=False)
    std_b = ranks_b.float().std(unbiased=False)
    spearman_corr = cov / (std_a * std_b + 1e-8)
    return spearman_corr


# L1 sorting distance (rank difference)
def l1_sorting(attributions_a: torch.Tensor, attributions_b: torch.Tensor) -> torch.Tensor:
    a = attributions_a.flatten()
    b = attributions_b.flatten()

    ranks_a = torch.argsort(torch.argsort(a))
    ranks_b = torch.argsort(torch.argsort(b))

    l1_dist = torch.norm((ranks_a - ranks_b).float(), p=1) / len(ranks_a)
    return l1_dist


# Kendall Tau 
def tau(attributions_a: torch.Tensor, attributions_b: torch.Tensor) -> torch.Tensor:
    a = attributions_a.flatten()
    b = attributions_b.flatten()
    n = len(a)

    ai = a.unsqueeze(0).repeat(n, 1)
    bi = b.unsqueeze(0).repeat(n, 1)
    concordant = ((ai - ai.T) * (bi - bi.T)) > 0
    discordant = ((ai - ai.T) * (bi - bi.T)) < 0
    tau_val = (concordant.sum() - discordant.sum()) / (n * (n - 1))
    return tau_val


# Cosine distance
def cosine_distance(attributions_a: torch.Tensor, attributions_b: torch.Tensor) -> torch.Tensor:
    a = attributions_a.flatten().float()
    b = attributions_b.flatten().float()
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    cosine_sim = torch.dot(a, b)
    return 1 - cosine_sim  


# L1 distance
def l1_distance(attributions_a: torch.Tensor, attributions_b: torch.Tensor) -> torch.Tensor:
    if attributions_a.device.type == 'cpu':
        attributions_a = attributions_a.cuda()
    if attributions_b.device.type =='cpu':
        attributions_b = attributions_b.cuda()
    try:
        diff = (attributions_a - attributions_b).abs()
    except:
        breakpoint()
    return diff.mean()


# SSIM (Structural Similarity)

def SSIM(attributions_a: torch.Tensor, attributions_b: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    
    if attributions_a.ndim == 2:
        attributions_a = attributions_a.unsqueeze(0).unsqueeze(0)
        attributions_b = attributions_b.unsqueeze(0).unsqueeze(0)
    elif attributions_a.ndim == 3:
        attributions_a = attributions_a.unsqueeze(0)
        attributions_b = attributions_b.unsqueeze(0)

    mu1 = F.avg_pool2d(attributions_a, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(attributions_b, window_size, stride=1, padding=window_size//2)

    sigma1_sq = F.avg_pool2d(attributions_a ** 2, window_size, stride=1, padding=window_size//2) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(attributions_b ** 2, window_size, stride=1, padding=window_size//2) - mu2 ** 2
    sigma12 = F.avg_pool2d(attributions_a * attributions_b, window_size, stride=1, padding=window_size//2) - mu1 * mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def smooth_transformation(attributions: np.ndarray,kernel_size:int) -> float:

    pad_size = (kernel_size-1) // 2
    pad_max = ((21-1)//2)
    attributions = attributions.unsqueeze(0)
    kernel = torch.nn.AvgPool3d((3,kernel_size,kernel_size), stride=1)
    attributions = kernel(attributions).squeeze(0).squeeze(0)

    pad = pad_max-pad_size
    if pad!=0:
        attributions = attributions[pad:-pad,pad:-pad]

    return attributions  
   
def patching(attributions: torch.Tensor, size: int = 4):
    
    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions, dtype=torch.float32)
    else:
        attributions = attributions.to(torch.float32)

    H, W = attributions.shape[-2], attributions.shape[-1]

    H_trim = H - (H % size)
    W_trim = W - (W % size)
    attributions = attributions[..., :H_trim, :W_trim]

    attributions = attributions.view(
        H_trim // size, size,
        W_trim // size, size
    )

    attributions = attributions.mean(dim=(1, 3))

    return attributions 

def topk_transformation(attributions: np.ndarray) -> float:

    original_shape = attributions.shape
    k = (attributions.shape[0]*attributions.shape[0])//10
    attributions = attributions.reshape(-1)
    topk_vals, topk_indices = torch.topk(attributions, k=k)
    bottomk_vals, bottomk_indices = torch.topk(-attributions, k=k)

    #Create binary mask (all 0)
    mask = torch.zeros_like(attributions, dtype=torch.float)
    mask[topk_indices] = 1
    mask[bottomk_indices] = -1

    binary_mask = mask.view(original_shape)

    return binary_mask

def topk_agreement(attributions_a: np.ndarray, attributions_b: np.ndarray) -> float:

    attributions_a = topk_transformation(attributions_a)
    attributions_b = topk_transformation(attributions_b)
    distance = l1_distance(attributions_a=attributions_a,attributions_b=attributions_b)

    return distance
 

def agreement_selector(attributions_a: np.ndarray, attributions_b: np.ndarray,
                        agreement_function:str, 
                        std:int,
                        maximum:int,
                        minimum:int,
                        preprocessing:bool=True,
                        kernel_size:int=4, #kernel size for convolution,
                        )-> float: 
    

    if preprocessing==True:
        attributions_check(attributions_a, attributions_b)
        attributions_a = attributions_preprocessing(attributions_a,agreement_measure=agreement_function, std=std,kernel_size=kernel_size,minimum =minimum, maximum= maximum,normalization='quantil_local')
        attributions_b = attributions_preprocessing(attributions_b,agreement_measure=agreement_function, std = std,kernel_size=kernel_size,minimum =minimum, maximum= maximum,normalization ='quantil_local')

    if agreement_function =="l1":
        value = l1_distance(attributions_a=attributions_a,
                            attributions_b=attributions_b)
        
    elif agreement_function =="l1_sorting":
        value = l1_sorting(attributions_a=attributions_a,
                            attributions_b=attributions_b)
        
    elif agreement_function=="spearman":
        value = spearman(attributions_a=attributions_a,
                            attributions_b=attributions_b)
     

    
    elif agreement_function.startswith("SSIM") :
      
        value = SSIM(attributions_a=attributions_a,
                    attributions_b=attributions_b)

    
    elif agreement_function == 'topk':
       
        value = topk_agreement(attributions_a=attributions_a,attributions_b=attributions_b)

    elif agreement_function == 'tau':
        
        value = tau(attributions_a=attributions_a,attributions_b=attributions_b)

    elif agreement_function == 'cosine_distance':
        
        value = cosine_distance(attributions_a=attributions_a,attributions_b=attributions_b)
        
    else:
        raise ValueError("Invalid agreement function.")
    
    return value

def batch_agreement(attributions_test_1: Dict[str, torch.Tensor], attributions_test_2: Dict[str,torch.Tensor],
                    agreement_function:str, 
                    std:int,
                    minimum:int,
                    maximum:int,
                    preprocessing:bool=True,
                    kernel_size:int=4, #kernel size for convolution
                    )-> tuple[float, float, Dict[str, float]]: #mean agreement, variance agreement, agreement per image
    
    assert len(attributions_test_1) == len(attributions_test_2), "attributions should have same len"
    
    assert agreement_function in ["density", "l1", "SSIM","topk","l1_smoothed", "SSIM_smoothed","spearman","l1_sorting","tau","cosine_distance"], "Invalid agreement function. Choose between convolution, density oe consistency"
    
   
    agreement_per_image = {}
    
    for image1, image2 in tqdm(zip(attributions_test_1.keys(), attributions_test_2.keys()), total=len(attributions_test_1), desc="Calculating agreement"):
        agreement_per_image[image1] = agreement_selector(attributions_a=attributions_test_1[image1],
                                        attributions_b=attributions_test_2[image2],
                                        agreement_function=agreement_function,
                                        std = std,
                                        preprocessing=preprocessing,
                                        kernel_size=kernel_size,
                                        minimum = minimum, 
                                        maximum = maximum
                                        )
        
    values = list(agreement_per_image.values())

    tensor_values = torch.tensor(values, dtype=torch.float32)

    mask = ~torch.isnan(tensor_values)
    valid_values = tensor_values[mask]

    mean_val = torch.mean(valid_values)
    var_val = torch.var(valid_values, unbiased=False)  

    return float(mean_val), float(var_val), agreement_per_image
   
def clean_path(p,corruption_type):
        
        p = p.replace(f'dermamnist_corrupted_{corruption_type}_224','dermamnist_224')
        p = p.replace(f'__{corruption_type}', '')
        return p

def batch_agreement_only_true_labels(attributions_test_1: Dict[str, np.ndarray], attributions_test_2: Dict[str, np.ndarray],
                    corrected_1 ,
                    corrected_2 ,
                    agreement_function:str,      
                    corruption_type:str = 'None', 
                    kernel_size:Tuple[int, int]=(3,3), #kernel size for convolution
                    )-> tuple[float, float, Dict[str, float]]: #mean agreement, variance agreement, agreement per image
    
    assert len(attributions_test_1) == len(attributions_test_2), "attributions should have same len"
    
    assert agreement_function in ["topk", "l1", "SSIM","l1_smooothed", "SSIM_smooted"], "Invalid agreement function. Choose between convolution, density oe consistency"
    

    
    corrected_2 = [clean_path(p,corruption_type=corruption_type) for p in corrected_2]
    
    images=list(set(corrected_1) & set(corrected_2))
    
    agreement_per_image = {}
    
    for image in tqdm(images, total=len(images), desc="Calculating agreement"):

        image = image.replace('\\', '/')
        image_2 = image.replace("dermamnist_224", f"dermamnist_corrupted_{corruption_type}_224")
            
        base, ext = os.path.splitext(image_2)
        image_2 = base + f'__{corruption_type}' + ext

        agreement_per_image[image] = agreement_selector(attributions_a=attributions_test_1[image],
                                        attributions_b=attributions_test_2[image_2],
                                        agreement_function=agreement_function,
                                        kernel_size=kernel_size)

    return float(np.average(list(agreement_per_image.values()))), float(np.var(list(agreement_per_image.values()))), agreement_per_image
