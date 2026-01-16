import torch 
from typing import  List,Dict
from captum import attr
import argparse
from timm.models import create_model
from medmnist import INFO
from MedDataset import MedDataset
from torchvision import transforms
from torch.utils.data import DataLoader,Subset
import torchvision
from tqdm import tqdm
import numpy as np
import os
from multiprocessing import Pool, get_context
import gc
import time
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
from collections import defaultdict
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLinearRegression
import urllib.request
from torchvision import datasets 
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Union
from adv_lib.utils.visdom_logger import VisdomLogger
import numbers
from functools import partial
from adv_lib.utils.projections import clamp_
from tqdm import tqdm, trange
from agreement import attributions_preprocessing, l1_distance
import json

class Normalizer(torch.nn.Module):
    def __init__(self,device: str ="cuda"):
        super(Normalizer, self).__init__()
        mean_const, std_const = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        mean = torch.as_tensor(mean_const)[None, :, None, None]
        std = torch.as_tensor(std_const)[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.device = device

    def forward(self, x):
        return x.sub(self.mean.to(self.device)).div(self.std.to(self.device))

class NormalizedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalizer = Normalizer(device=device)

    def forward(self, x):
        x = self.normalizer(x)
        return self.model(x)
    
imagenet_synsets = []
with open("imagenet_idx.txt", "r") as f:
    for line in f:
        synset = line.strip().split()[0]
        imagenet_synsets.append(synset)

# Mapping from synset (e.g., n01440764) → ImageNet 1k index
synset_to_imagenet_idx = {syn: idx for idx, syn in enumerate(imagenet_synsets)}

# --- 4. Custom dataset to return correct label ---
class ImageNetSubset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, local_label = self.samples[index]
        synset = self.classes[local_label]  # folder name
        imagenet_label = synset_to_imagenet_idx.get(synset, -1)  # fallback -1 if missing
        image = self.loader(path)
        filename = path.split('/')[-1]
        if self.transform is not None:
            image = self.transform(image)
        return image, imagenet_label,filename

class ExplainableModel():

    @staticmethod
    def get_feature_mask(sub_cube_size=4) -> torch.Tensor:

        base = torch.ones(size=(sub_cube_size,), dtype=torch.long)

        if 224 % sub_cube_size != 0:
            raise ValueError("sub_cube_size must divide 224")

        sub_cubes_per_stride = 224 // sub_cube_size  #16

        j = 0
        stride_tensors = []
        for _ in range(sub_cubes_per_stride):

            base_tensors = []
            for _ in range(sub_cubes_per_stride):
                base_tensors.append(base * j)
                j += 1

            row_tensor = torch.concatenate(base_tensors, dim=0)

            stride_tensor = torch.cat([row_tensor[None, :] for _ in range(sub_cube_size)], dim=0)

            stride_tensors.append(stride_tensor)

        channel_tensor = torch.cat(stride_tensors, dim=0)

        feature_mask = torch.cat([channel_tensor[None, :, :] for _ in range(3)], dim=0)[None, :, :, :]

        return feature_mask

    def __init__(self, model_name:str, train_data_name: str, n_classes: int):

        model_path = f"checkpoints/{model_name}/{train_data_name}/checkpoint_best.pth"
        self.model_name = model_name
        self.train_data_name = train_data_name
        self.n_classes = n_classes
        print(model_path)
        
        self.load_model(model_path, n_classes,train_data_name)
        
        self.available_algorithms = ["IntegratedGradients", "Saliency", "DeepLift", "DeepLiftShap", "GradientShap",
                                     "InputXGradient", "GuidedBackprop", "GuidedGradCam", "Deconvolution",
                                     "FeatureAblation", "Occlusion",
                                     "FeaturePermutation", 
                                     "Lime", "KernelShap","SmoothGrad","ShapleyValueSampling",
                                     "LRP","GradCAM_plusplus","GuidedBackprop_new","GradCAM"
                                        ]
    def load_model(self, model_path: str, n_classes: int, train_data_name: str):
        
        print(f"loading:{model_path}")

        if 'mnist' in train_data_name:
        
            self.model : torch.nn.Module = create_model(self.model_name, num_classes=n_classes, pretrained=True) 

            checkpoint_model = torch.load(model_path,weights_only=False)["model"]
        
            self.model.load_state_dict(checkpoint_model, strict=True, assign=True)

            self.model = NormalizedModel(self.model)

        if train_data_name.startswith('imagenet'):

            self.model : torch.nn.Module = create_model(self.model_name, num_classes=n_classes, pretrained=True)
            self.model = NormalizedModel(self.model)

        self.model = self.model.eval()

        
    def identity_rule(module, relevance_in, relevance_out, *args, **kwargs):
    
                    return relevance_out

    def applyXAI(self, algorithm: str, input_tensor: torch.Tensor,
                   target_classes: torch.Tensor, data_name:str, post_processing=False,kernel_size:int=1)-> torch.Tensor:
        
            assert algorithm in self.available_algorithms, f"Invalid algorithm {algorithm}, choose from {self.available_algorithms}"
            model_path = f"checkpoints/{self.model_name}/{data_name}/checkpoint_best.pth"

           
            self.load_model(model_path, self.n_classes,data_name)
    
        
            input_tensor=input_tensor.cuda()
            target_classes=target_classes.cuda()

            self.model = self.model.cuda()

            if target_classes.shape !=0:
        
                if algorithm == "IntegratedGradients":

                    method = attr.IntegratedGradients(self.model)
                    attributions = method.attribute(input_tensor, target=target_classes, internal_batch_size=6)

            
                elif algorithm == "Saliency":

                    method = attr.Saliency(self.model)
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes,abs=False)
                    
        
                elif algorithm == "DeepLift":

                    method = attr.DeepLift(self.model.cuda())
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes)
                    
        
                elif algorithm == "DeepLiftShap":
            
                    raise NotImplementedError("DeepLiftShap not implemented")
        
        
                elif algorithm == "InputXGradient":
            
                    method = attr.InputXGradient(self.model)
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes)
        
                elif algorithm == "GuidedBackprop":
            
                    method = attr.GuidedBackprop(self.model)
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes)
        
                elif algorithm == "GuidedGradCam":

        
                    if self.model_name == "MedVit":
                        layer = self.__get_target_layer_GradCAM_MedViT()
                    elif self.model_name == "vgg16":
                        layer = self.model.model.features[-1]
                    elif self.model_name == "densenet121":
                        layer = self.model.model.features[-1]
                    elif self.model_name == "resnet50":
                        layer = self.model.model.layer4[-1]
                    elif self.model_name =='rexnet_100':
                        layer = self.model.model.features[-1].conv
                    elif self.model_name == 'efficientnet_b4':
                        layer = self.model.model.conv_head
                    elif self.model_name == 'regnety_008':
                        layer = self.model.model.s4.b2.conv3.conv
                    elif self.model_name =='poolformerv2_s36':
                        layer=self.model.model.stages[-1].blocks[-1].mlp.fc2
                        

                    elif self.model_name == "convnext_base":
                        layer = self.model.model.stages[-1].blocks[-1].conv_dw
                    elif self.model_name == "custom_vit_base_patch16_224":
                        layer = self.model.model.blocks[-1].norm1
                    elif self.model_name == "custom_deit_base_patch16_224":
                        layer = self.model.model.blocks[-1].norm1
                    elif self.model_name == "custom_pit_b_224":
                        layer = self.model.model.transformers[-1].blocks[-1].norm1
                    else:
                        raise ValueError(f"CAM not implemented for model {self.model_name}")
                    
                    method = attr.GuidedGradCam(self.model,layer=layer)
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes,attribute_to_layer_input=False)
        
        
                elif algorithm == "Deconvolution":
            
                    method = attr.Deconvolution(self.model)
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes)
        
                elif algorithm == "FeatureAblation":
            
                    method = attr.FeatureAblation(self.model.forward)
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes, show_progress=True)
        
                elif algorithm == "Occlusion":
                    occlusion = attr.Occlusion(self.model)

                    # Drop batch dimension for sliding window and stride settings
                    sliding_window_shapes = (3, 4, 4)  # for (C, H, W)
                    strides               = (3, 4, 4)  # same length

                    with torch.no_grad():
                        attributions = occlusion.attribute(
                            inputs                 = input_tensor.requires_grad_(True),  # shape (B, C, H, W)
                            target                 = target_classes,                     # shape (B,)
                            sliding_window_shapes  = sliding_window_shapes,
                            strides                = strides,
                            baselines              = 0,
                            perturbations_per_eval = 64
                        )
            
        
                elif algorithm == "FeaturePermutation":
            
                    method = attr.FeatureAblation(self.model)
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes, show_progress=True)
        
                elif algorithm == "ShapleyValueSampling":
        
                    method = attr.ShapleyValueSampling(self.model)
                    attributions = method.attribute(input_tensor, feature_mask=ExplainableModel.get_feature_mask().cuda(),
                                            target=target_classes, show_progress=True)
                

                elif algorithm == "Lime":

                    exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

                    lr_lime = attr.Lime(self.model, interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
                    similarity_func=exp_eucl_distance
                    )
                    
                    attributions = lr_lime.attribute(input_tensor,
                                            target=target_classes)

                elif algorithm == "KernelShap":

                    method = attr.KernelShap(self.model)
                    attributions = method.attribute(input_tensor, target=target_classes,n_samples=50)
            
                # Tell Captum how to handle every Identity layer.
                    custom_rules = {torch.nn.Identity: self.identity_rule}
            
                    method = attr.LRP(self.model,rules   = custom_rules)
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes)

                elif algorithm == "SmoothGrad":
            
                    #method = SmoothGrad(self.model, magnitude = False)
                    method = attr.NoiseTunnel(attr.Saliency(self.model))

                    attributions = method.attribute(input_tensor.requires_grad_(),target = target_classes)
                
                elif algorithm == "GradientShap":

                    baselines = torch.zeros_like(input_tensor)
                    method = attr.GradientShap(self.model.cuda())
                    attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes,baselines=baselines)
        
                else:
                    raise ValueError(f"Invalid algorithm {algorithm}")
                
                if post_processing:
            
                    attributions = torch.stack([
                    attributions_preprocessing(a, agreement_measure='l1', std=0, kernel_size=kernel_size,
                                                normalization='quantil_local', minimum=0, maximum=0)[None, ...]
                        for a in attributions
                    ])
                    
            return attributions

    def __get_attribution_path(self, data_name:str, data_split:str) -> str:

        if 'corrupted' in data_name:

            path = f"attributions/{self.model_name}_attributions/{data_name}/{data_name}_{data_split}/"
            
        else:

            path = f"attributions/{self.model_name}_attributions/{self.train_data_name}/{data_name}_{data_split}/"

        return path
    


    def __save_attributions(self, algorithm: str, data_name:str, data_split:str, attributions: Dict[str, np.ndarray]):

        attributions_path = self.__get_attribution_path(data_name, data_split)
       
        os.makedirs(attributions_path, exist_ok=True)
        
        np.savez_compressed(os.path.join(attributions_path, f"{algorithm}.npz"), **attributions)

    def __save_images(self, algorithm: str, data_name:str, data_split:str, images: Dict[str, np.ndarray]):

        path = f"{data_name}_images"

        os.makedirs(path, exist_ok=True)
        
        np.savez_compressed(os.path.join(path, ".npz"), **images)
    
    def __load_images(self, algorithm: str, data_name:str, data_split:str) -> Dict[str, np.ndarray] | None:
        
        path = f"{data_name}/images.npz"

        if not os.path.exists(path):
            return None, path

        loaded_data = np.load(path)

        return {key: loaded_data[key] for key in loaded_data},path

    def __load_attributions(self, algorithm: str, data_name:str, data_split:str) -> Dict[str, np.ndarray] | None:
        
        attributions_path = self.__get_attribution_path(data_name, data_split)

        print(attributions_path)

        path = os.path.join(attributions_path, f"{algorithm}.npz")
        if not os.path.exists(path):
            return None, attributions_path
        loaded_data = np.load(path)

        return {key: torch.from_numpy(loaded_data[key]).to(torch.float32).to("cuda") for key in loaded_data}, attributions_path



    def explain_dataset(self, algorithm: str, train_data_name :str,data_name: str, data_split: str, batch_size: int,
                        new_process: bool = False, save_attributions:bool = True) -> Dict[str, np.ndarray]:
        
        attributions, attributions_path = self.__load_attributions(algorithm, data_name, data_split)
        if attributions is not None:
            print(f"Attributions for {algorithm} already exist. ({attributions_path})", flush=True)
            return attributions
    
        data = ExplainableModel.load_data(data_name, data_split)
        
        torch.cuda.empty_cache()
        self.model = self.model.cuda()

        loader = DataLoader(data, batch_size=batch_size, pin_memory=True, 
                               num_workers = 8, 
                                shuffle=False)


        attributions_tot = dict()
        

        # Get multiprocessing context
        ctx = get_context("spawn")
        pool = None
        restart_every = 5  # batches
 
        for i, (image_tensor, labels, names) in enumerate(tqdm(loader, desc=f"Explaining batch of images using {algorithm}")): 
                
                if new_process:
                    # Restart pool every N batches
                    if i % restart_every == 0:
                        if pool:
                            pool.close()
                            pool.join()
                        pool = ctx.Pool(1)
                    # Define wrapper function if needed externally

                    r = pool.apply_async(self.applyXAI, args=(algorithm, image_tensor, labels,train_data_name))
                    attributions = r.get()

                else:

                    attributions = self.applyXAI(algorithm, image_tensor, labels,train_data_name)

                
                for j in range(attributions.shape[0]):
                    attributions_tot[names[j]] = attributions[j].cpu().detach().numpy().astype(np.float32)
                    
                del image_tensor, labels, names,attributions
                torch.cuda.empty_cache()
                time.sleep(0.1)  # helps prevent IPC buildup

        if pool:
                pool.close()
                pool.join()

        self.model.cpu()

        if save_attributions== True:

                self.__save_attributions(algorithm, data_name, data_split, attributions_tot)
            
        return attributions_tot
      
    
    def build_transform(input_size = 224):
        resize_im = input_size > 32

        t = []
        if resize_im:
            size = int((256 / 224) * input_size)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(input_size))

        t.append(transforms.ToTensor())
        
        return transforms.Compose(t)
    
    def load_data(data_name:str, data_split:str = "test", to_transform:bool = True) -> MedDataset | None:

        if 'mnist' in data_name:

            data_path = f"data/{data_name}_224/{data_split}"
        
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data path {data_path} does not exist!")
        
            if to_transform:
                return MedDataset(data_path, transform=ExplainableModel.build_transform())
            else:
                return MedDataset(data_path, transform=lambda x: transforms.ToTensor()(x))

        elif data_name.startswith('imagenet'):
           

            #data_path = '/data/datasets/imagenet/val'
            data_path = f'data/{data_name}'

            transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
            imagenet_data = ImageNetSubset(root=data_path, transform=transform)
           
            return imagenet_data
    



            

    def calculate_all_attributions(self, train_data_name : str, data_name:str, data_split:str):
        
        # list of attributions to calculate with batch size optimal for each algorithm (cluster)
        ['Integrated_Gradients', 'GuidedBackprop', 'DeepLift', 'Saliency','InputXGradient', 'Lime', 'KernelShap','SmoothGrad']
        list_attributions = [
            ("IntegratedGradients", self.explain_dataset("IntegratedGradients", train_data_name, data_name, data_split, batch_size=64)),
            ("Saliency", self.explain_dataset("Saliency", train_data_name, data_name, data_split, batch_size=64, new_process=True)),
            ("DeepLift", self.explain_dataset("DeepLift", train_data_name, data_name, data_split, batch_size=64)),
            ("InputXGradient", self.explain_dataset("InputXGradient", train_data_name, data_name, data_split, batch_size=64)),
            ("GuidedBackprop", self.explain_dataset("GuidedBackprop",train_data_name, data_name, data_split, batch_size=64, new_process=True)),
            ("GradientShap",self.explain_dataset("GradientShap", train_data_name,data_name, data_split, batch_size=32)),
            ("SmoothGrad", self.explain_dataset("SmoothGrad", train_data_name,data_name, data_split, batch_size=64)),
            ("GuidedGradCam", self.explain_dataset("GuidedGradCam", train_data_name,data_name, data_split, batch_size=64)),
    
        ]
        
        return list_attributions

    def test(self, data_name:str, data_split:str = "test", batch_size: int = 512):
        
        data = ExplainableModel.load_data(data_name, data_split)
        
        self.model.eval().cuda()
            
        loader = DataLoader(data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)

        im = next(iter(loader))

        acc_per_label = {}
        corrected_per_label = {}

        #with torch.cuda.amp.autocast():
        with torch.inference_mode():
            total_correct = 0
            total_samples = 0
            per_class_correct = defaultdict(int)
            per_class_total = defaultdict(int)


            data_iter = ((inputs, labels) for inputs, labels,_ in tqdm(loader))
            

            for inputs, labels in data_iter:

                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                # Total accuracy
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                # Per class accuracy
                for label, pred in zip(labels, preds):
                    per_class_total[label.item()] += 1
                    if label == pred:
                        per_class_correct[label.item()] += 1

        overall_accuracy = total_correct / total_samples

         # Per class accuracy
        per_class_accuracy = {
                cls: per_class_correct[cls] / per_class_total[cls]
                for cls in per_class_total
            }
        #Filters classes with over 90% accuracy
        high_accuracy_classes = {
        cls: acc for cls, acc in per_class_accuracy.items() if acc > 0.90
        }
        
        url = (
            "https://raw.githubusercontent.com/tensorflow/models/master/"
            "research/slim/datasets/imagenet_lsvrc_2015_synsets.txt"
        )

        response = urllib.request.urlopen(url)
        synset_list = [line.strip().decode('utf-8') for line in response.readlines()]

        return overall_accuracy, per_class_accuracy,high_accuracy_classes
    

    def return_corrected_images(self, data_name:str, data_split:str = "test", batch_size: int = 32):

        data = ExplainableModel.load_data(data_name, data_split)
        
        self.model.eval().cuda()

        loader = DataLoader(data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)

        correct_images = []

        #with torch.cuda.amp.autocast():
        with torch.inference_mode():

            for inputs, labels, names in tqdm(loader):

                inputs, labels = inputs.cuda(), labels.cuda()
                
                outputs = self.model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                
                mask = (predicted == labels).cpu().numpy()  # converte in numpy array (sul CPU)

                # Step 2: filter the list of names using the mask
                images  = [n for n, keep in zip(names, mask) if keep]
                correct_images.extend(images)

                inputs.cpu(), outputs.cpu(), labels.cpu()
                torch.cuda.empty_cache()
   
        self.model.cpu()
       
        return correct_images


    def attack(self,
            data_name:str,
            algorithm: str,
            ε: int,
            expl_loss_function : str="topk",
            loss_function : str="ce",
            n_steps: int = 100,
            lr : int = 0.1,
            batch_size : int = 8,
            new_process: bool = False,
            dataset_subset_size=100,
            kernel_size:int=1
            ) -> None:
        #save metrics only if default parameters are used
        default_params = (expl_loss_function == "topk" and loss_function == "ce" and n_steps == 100 and lr == 0.1 and dataset_subset_size == 100)
        if default_params:
            file_name = f"attack/{self.model_name}/{self.train_data_name}/{data_name}_test/"
            os.makedirs(file_name, exist_ok=True)
            
            file_name = file_name+f"{algorithm}_{ε}_kernel_size:{kernel_size}.json"
            if os.path.exists(file_name):
                print(f"File {file_name} already exists, skipping attack", flush=True)
                return
 
        #if adversarial examples and adversarial explanations already exist, skip attack
        adv_example_file = f"adversarial_examples/{self.model_name}/{self.train_data_name}/{data_name}_test/{algorithm}_{ε}.npz"
        adv_expl_file = f"adversarial_explanations/{self.model_name}/{self.train_data_name}/{data_name}_test/{algorithm}_{ε}.npz"
        
        if (not default_params) and os.path.exists(adv_example_file) and os.path.exists(adv_expl_file):
            print(f"Adversarial examples for {algorithm} already exist, skipping attack", flush=True)
            print(f"Adversarial explanations for {algorithm} already exist, skipping attack", flush=True)
            return
        
        
        if algorithm not in self.available_algorithms:
            raise ValueError(f"Invalid algorithm {algorithm}, choose from {self.available_algorithms}")
        
    
        lst_L_1_exp = []
    
        total_accuracy_original = 0
        total_accuracy_perturbed = 0
        
        batch_cross_entropy = []
        batch_explanation_loss = []
        batch_loss = []
        
        adversarial_examples = {}
        original_examples = {}
        adversarial_explanations = {}

        samples = 0
        epsilon = ε
        ε = ε/255
        
        data = ExplainableModel.load_data(data_name, "test", to_transform=True)
        
        set_all_seeds(0)
        
        subset_indices = torch.randperm(len(data))[:dataset_subset_size]
        subset = Subset(data , subset_indices)
        dataloader = DataLoader(subset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
        
        # Get multiprocessing context
        ctx = get_context("spawn")
        pool = None
        restart_every = 1  # batches
        
        if algorithm == "GuidedBackprop":
            restart_every = 1  # batches
        if algorithm == "Saliency":
            restart_every = 8
        if algorithm == "SmoothGrad":
            restart_every = 5
    
        for i, (inputs, labels, names) in enumerate(dataloader):
                
                torch.cuda.empty_cache()
                
                if new_process:
                    
                    # Restart pool every N batches
                    if i % restart_every == 0:
                        if pool:
                            pool.close()
                            pool.join()
                        pool = ctx.Pool(1)
                    
                    #Done to avoid problems with torch memory leaks
                    #https://github.com/pytorch/pytorch/issues/51978
                    #these two lines will spawn a new process and run the function in it and then return the result to the main process
                    #this way the memory is freed forcefully after the process is done
            
                    r=pool.apply_async(pgd_linf, args=(i, len(dataloader),inputs,labels,names,algorithm,data_name,ε,self,n_steps,lr,expl_loss_function,loss_function),kwds={'kernel_size': kernel_size})
                    metrics, adv_inputs, adversarial_expl,original_inputs = r.get()
                else:
                    metrics, adv_inputs, adversarial_expl,original_inputs  = pgd_linf(i, len(dataloader), inputs, labels, names, algorithm, data_name,ε, self, n_steps, lr,
                                       expl_loss_function, loss_function,kernel_size=kernel_size)
                
                torch.cuda.empty_cache()
                    
                samples += 1
                
                lst_L_1_exp += metrics['l1']
                total_accuracy_original += metrics['accuracy_original']
                total_accuracy_perturbed += metrics['accuracy_perturbed']
                
                batch_cross_entropy += metrics['cross_entropy']
                batch_explanation_loss += metrics['explanation_loss']
                batch_loss += metrics['epoch_loss']
                
                # create a tensor of adversarial inputs
                for adv_input,original_input, adv_expl, name in zip(adv_inputs,original_inputs, adversarial_expl, names):
                    adv_input = adv_input.numpy()
                    adv_expl = adv_expl.numpy()
                    original_input = original_input.numpy()
                    adversarial_examples[name] = adv_input.astype(np.float32)
                    original_examples[name] = original_input.astype(np.float32)
                    adversarial_explanations[name] = adv_expl.astype(np.float32)
                self.model.zero_grad()

                del inputs, labels, names
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(0.5)  # helps prevent IPC buildup
        
        if pool:
            pool.close()
            pool.join()
                
        total_accuracy_perturbed = total_accuracy_perturbed/samples
        total_accuracy_original = total_accuracy_original/samples
        data = {
            'lst_L_1_exp': lst_L_1_exp,
            'lst_L_1_exp_mean': sum(lst_L_1_exp)/len(lst_L_1_exp),
            'total_accuracy_original': total_accuracy_original,
            'total_accuracy_perturbed': total_accuracy_perturbed,
            'batch_cross_entropy' : batch_cross_entropy,
            'batch_explanation_loss' : batch_explanation_loss,
            'batch_loss' : batch_loss,
        }
        
        #save metrics only if default parameters are used
        if default_params:
            with open(file_name, "w") as results_file:
                json.dump(data, results_file, indent=4)
     
        #save adversarial and original examples
        folder_name= f"adversarial_examples/{self.model_name}/{self.train_data_name}/{data_name}_test"
        os.makedirs(folder_name, exist_ok=True)
       
        
        adv_example_file = f"{folder_name}/{algorithm}_{epsilon}.npz"
        if not os.path.exists(adv_example_file):
            np.savez_compressed(adv_example_file, **adversarial_examples)
        
        #save explanations
        folder_name= f"adversarial_explanations/{self.model_name}/{self.train_data_name}/{data_name}_test"
        os.makedirs(folder_name, exist_ok=True)
        
        adv_example_file = f"{folder_name}/{algorithm}_{epsilon}.npz"
        if not os.path.exists(adv_example_file):
            np.savez_compressed(adv_example_file, **adversarial_explanations)
        
        folder_name= f"original_examples/{self.model_name}/{self.train_data_name}/{data_name}_test"
        os.makedirs(folder_name, exist_ok=True)

        adv_example_file = f"{folder_name}/{algorithm}_{epsilon}.npz"
        if not os.path.exists(adv_example_file):
            np.savez_compressed(adv_example_file, **original_examples)
        

    
def pgd_linf(
            n_batch:int,
            total_batch:int,
            inputs: torch.Tensor,
            labels: torch.Tensor,
            names:List[str],
            algorithm: str,
            train_data_name :str,
            ε: float,
            explainableModel : ExplainableModel,
            n_steps: int,
            lr : int,
            expl_loss_function: str,
            loss_function: str,
            restarts: int = 1,
            callback: Optional[VisdomLogger] = None,
            kernel_size:int=1) -> torch.Tensor:
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = len(inputs)

        inputs = inputs.to(device)
        labels = labels.to(device)

        adv_inputs = inputs.clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if isinstance(ε, numbers.Real):
            ε = torch.full_like(adv_found, ε, dtype=inputs.dtype)
    
        pgd_attack = partial(_pgd_linf, 
                            method=algorithm, explainableModel = explainableModel, train_data_name= train_data_name, n_steps=n_steps,
                            expl_loss_function = expl_loss_function,
                            loss_function = loss_function,
                            lr = lr, n_batch=n_batch, total_batch=total_batch,
                            kernel_size = kernel_size)

        for i in range(restarts):
            original_expl, adv_found_run, adv_inputs_run, epoch_loss, cross_entropy, explanation_loss = pgd_attack(inputs=inputs[~adv_found], ε=ε[~adv_found])
            adv_inputs[~adv_found] = adv_inputs_run
            adv_found[~adv_found] = adv_found_run
        
            
            if callback:
                callback.line('success', i + 1, adv_found.float().mean())

            if adv_found.all():
                break
        
        # get the original and adversarial predictions
        model=explainableModel.model.eval().cuda()
        original_labels = model(inputs).argmax(1)
        adv_pred_labels = model(adv_inputs).argmax(1)
        
        # get the original and adversarial explanations
        adversarial_expl = explainableModel.applyXAI(algorithm=algorithm,input_tensor = adv_inputs, target_classes = adv_pred_labels, data_name = train_data_name, kernel_size=kernel_size,post_processing=True)
        
        # calculate the metrics
        accuracy_original = torch.sum(original_labels==labels)/len(inputs)
        accuracy_corrupted = torch.sum(adv_pred_labels == labels)/len(inputs)
        
        metrics={"accuracy_original": accuracy_original.detach().cpu().tolist(), 
                 "accuracy_perturbed": accuracy_corrupted.detach().cpu().tolist(),
                 "l1": [],
                 "l1_avg":[],
                 "epoch_loss": epoch_loss.tolist(),
                 "cross_entropy": cross_entropy.tolist(),
                 "explanation_loss": explanation_loss.tolist()}
        
        for original, adversarial in zip(original_expl, adversarial_expl):
            
            original = original[0]
            adversarial = adversarial[0]
            metrics['l1'].append(l1_distance(original, adversarial).item() )
        
        explainableModel.model.cpu()
        del original_expl
        gc.collect()
        torch.cuda.empty_cache()
        return metrics, adv_inputs.cpu().detach(), adversarial_expl.cpu().detach(),inputs.cpu().detach()

from torch.autograd import grad

def show(name, t):
    if isinstance(t, torch.Tensor):
        print(f"{name}: requires_grad={t.requires_grad}, grad_fn={t.grad_fn}")
    else:
        print(f"{name}: type={type(t)}")



def _pgd_linf(
                n_batch: int,
                total_batch: int,
                inputs: torch.Tensor,
                train_data_name:str,
                ε: torch.Tensor,
                method: str,
                explainableModel : ExplainableModel,
                n_steps: int,
                lr : int,
                expl_loss_function: str,
                loss_function: str,
                new_process: bool=False,
                kernel_size: int=1
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        model = explainableModel.model.eval().cuda()
        
        _loss_functions = {
            'ce': (partial(torch.nn.functional.cross_entropy, reduction='none'), 0.0001),
        }

        _expl_loss_functions = {
            'topk': (partial(topk),1),
        }

        expl_loss_func,expl_multiplier = _expl_loss_functions[expl_loss_function.lower()] 

        loss_func, multiplier = _loss_functions[loss_function.lower()]

        device = inputs.device
        batch_size = len(inputs)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
        lower, upper = torch.maximum(-inputs, -batch_view(ε)), torch.minimum(1 - inputs, batch_view(ε))
       
        
        δ = torch.zeros_like(inputs, requires_grad=True)
        best_adv = inputs.clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
        δ.data.uniform_(-1, 1).mul_(batch_view(ε))
        clamp_(δ, lower=lower, upper=upper)

        optimizer = torch.optim.Adam([δ], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min= lr/ 10)
        logits = model(inputs)
        pred_labels = logits.argmax(1)
      
        original_expl = explainableModel.applyXAI(algorithm=method, input_tensor = inputs, target_classes = pred_labels, data_name = train_data_name,kernel_size=kernel_size, post_processing=True) #maybe we can load them from file
        n = int(inputs.shape[2]//np.sqrt(10))
        target_expl = torch.zeros((1,204,204))
        target_expl[:,:n,:n] = torch.full((1,n,n),1)     
           
        target_expls = ((target_expl.unsqueeze(0), ) * batch_size)
        target=torch.concat((target_expls)).reshape(original_expl.shape).float()
        target=target.to('cuda:0')
       
        
        epoch_loss = []
        cross_entropy = []
        explanation_loss = []
        epoch_efficacy = []
        best_loss = float('inf')
        # Get multiprocessing context
        ctx = get_context("spawn")
        pool = None
        restart_every = 5  # batches
    
        for i in trange(n_steps, desc=f"steps...{n_batch+1}/{total_batch}", leave=False):
            
            
            optimizer.zero_grad()
            x_adv = inputs + δ
            adv_logits = model(x_adv)
        
            
            adv_expl = explainableModel.applyXAI(algorithm=method, input_tensor = x_adv, target_classes = pred_labels, data_name = train_data_name, post_processing=True)
            loss_expl = expl_multiplier*expl_loss_func(adv_expl,original_expl).float()
            cls_loss =  multiplier * loss_func(adv_logits, pred_labels)

            tot_loss = (cls_loss.cuda() + loss_expl.cuda()).mean()
            tot_loss.backward(retain_graph=True)

            optimizer.step()
            scheduler.step()
        
            is_clean = (adv_logits.argmax(1) == pred_labels)
            
            if tot_loss < best_loss:

                best_adv = torch.where(batch_view(is_clean), x_adv.detach(), best_adv)
                adv_found.logical_or_(is_clean)
                best_loss = tot_loss.item()

            clamp_(δ, lower=lower, upper=upper)
            tot_loss = tot_loss.detach().cpu().numpy()
            epoch_loss.append(tot_loss)
            cross_entropy.append((torch.sum(cls_loss)/len(inputs)).detach().cpu().numpy())
            explanation_loss.append((torch.sum(loss_expl)/len(inputs)).detach().cpu().numpy())

            assert (torch.min(δ) >= -ε[0]) & (torch.max(δ) <= ε[0])
            
            assert (torch.min(x_adv) >= 0) & (torch.max(x_adv) <= 1)

            del adv_expl,cls_loss,tot_loss,loss_expl,x_adv

            gc.collect()

            torch.cuda.empty_cache()

            
        epoch_loss = torch.tensor([torch.tensor(elem) for elem in epoch_loss])
        cross_entropy = torch.tensor([torch.tensor(elem) for elem in cross_entropy])
        explanation_loss = torch.tensor([torch.tensor(elem) for elem in explanation_loss])
        
        del δ
        gc.collect()
        torch.cuda.empty_cache()
        
        return original_expl, adv_found, best_adv, epoch_loss,cross_entropy,explanation_loss

def topk(adv_expl, original_expl):
    n = adv_expl.shape[0]
    adv_expl = adv_expl.reshape(n,-1)
    original_expl = original_expl.reshape(n,-1)
    l = adv_expl.shape[1]

    top_k_values, top_k_indices = torch.topk(original_expl, l//10, dim=1)
    top_k_values_adv = torch.gather(adv_expl, 1, top_k_indices)
    return top_k_values_adv.mean(dim=1)

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
if __name__ == "__main__":
       
        # Arguments
        parser = argparse.ArgumentParser(description='explanation downloading')
        parser.add_argument('--model_name', type=str, default = 'resnet50')
        parser.add_argument('--data_name', type=str, default = 'dermamnist')
        parser.add_argument('--train_data_name', type=str, default = 'dermamnist')
        parser.add_argument('--data_split', type=str, default = 'test')
        parser.add_argument('--n_classes', type=int, default = 7)
        args = parser.parse_args()

        
        n_classes = args.n_classes
      
        explainable_model = ExplainableModel(model_name=args.model_name, train_data_name=args.train_data_name, n_classes=n_classes)
    
        explainable_model.calculate_all_attributions(data_name = args.data_name, data_split='test', train_data_name=args.train_data_name)
      
        