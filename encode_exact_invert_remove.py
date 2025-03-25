import os
import argparse
import torch
import pickle
import json
from tqdm import tqdm
import random
import numpy as np
from datasets import load_dataset
from src.prc import KeyGen, Encode, str_to_bin, bin_to_str
import src.pseudogaussians as prc_gaussians
from src.baseline.gs_watermark import Gaussian_Shading_chacha
from src.baseline.treering_watermark import tr_detect, tr_get_noise
from inversion import stable_diffusion_pipe, generate, exact_inversion
from attacks import *
from src.prc import Detect, Decode

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts') # coco 
parser.add_argument('--inf_steps', type=int, default=20)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
parser.add_argument('--attack', type=str, default='stealthy') #white_noise, min_distortion, stealthy
parser.add_argument('--eps', type=float, default=80)
parser.add_argument('--device', type=str, default="cuda:1")
args = parser.parse_args()
print(args)

hf_cache_dir = 'hf_models'
device = args.device
n = 4 * 64 * 64  # the length of a PRC codeword
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
eps = args.eps
attack = args.attack
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}_eps_{eps}_attack_{attack}'

def add_white_noise(input_latent, eps):
    wn = torch.normal(0, 1, input_latent.shape)
    wn_normalized = wn / (torch.sum(wn**2)**0.5) * (eps)
    dev = input_latent.get_device()
    out = input_latent + wn_normalized.to(dev)
    mean, std = torch.mean(out), torch.std(out)
    out = (out-mean)/std
    return out

def add_stealthy_attack(input_latent, eps):
    input_latent_flat = input_latent.flatten().detach().clone().to(device)
    input_latent_flat_sorted = torch.sort((2*input_latent_flat)**2)
    sort_val = input_latent_flat_sorted.values
    sort_idx = input_latent_flat_sorted.indices
    # get squared sum
    
    # mulitplied by 2 because need to flip
    cumulative_sum = torch.cumsum(sort_val, dim=0)
    # get largest k
    largest_k = 0
    while largest_k < len(cumulative_sum):
        if cumulative_sum[largest_k] < eps**2:
            largest_k += 1
        else:
            break
    print(largest_k)
    for i in sort_idx[:largest_k]:
        input_latent_flat[i] *= -1.0
#         print(input_latent_flat[i])
    return input_latent_flat.reshape(input_latent.shape)

def add_clustering_attack(input_latent, eps):
    input_latent_flat = input_latent.flatten().detach().clone()
    curr_k = 0
    curr_l2_sum = 0
    while curr_k < len(input_latent_flat):
        if curr_l2_sum + (2*input_latent_flat[curr_k])**2 < eps**2:
            curr_k += 1
            curr_l2_sum += (2*input_latent_flat[curr_k])**2 
        else:
            break

    for i in range(curr_k+1):
        input_latent_flat[i] *= -1
    return input_latent_flat.reshape(input_latent.shape)
        
def add_min_distortion_attack(input_latent, eps):
    input_latent_flat = input_latent.flatten().detach().clone()
    input_latent_flat_sorted = torch.sort((input_latent_flat)**2)
#     print(input_latent_flat_sorted.values[64*64])
    sort_val = input_latent_flat_sorted.values
    print(sort_val[2*64*64])
    sort_idx = input_latent_flat_sorted.indices
    # get squared sum
    
    
    # get largest k
    cumsum = torch.cumsum(sort_val, dim=0) < eps**2
    
    for i in range(len(input_latent_flat)):
        if cumsum[i] == False:
            print(i)
            break
        
#     for i in sort_idx[:largest_k]:
        input_latent_flat[i] /= abs(input_latent_flat[i])
        input_latent_flat[i] *= 1e-3
    out = input_latent_flat.reshape(input_latent.shape).to(input_latent.dtype)
    
    return out.to(device)

if method == 'prc':
    if not os.path.exists(f'keys/{exp_id}.pkl'):  # Generate watermark key for the first time and save it to a file
        (encoding_key_ori, decoding_key_ori) = KeyGen(n, false_positive_rate=fpr, t=prc_t)  # Sample PRC keys
        with open(f'keys/{exp_id}.pkl', 'wb') as f:  # Save the keys to a file
            pickle.dump((encoding_key_ori, decoding_key_ori), f)
        with open(f'keys/{exp_id}.pkl', 'rb') as f:  # Load the keys from a file
            encoding_key, decoding_key = pickle.load(f)
        assert encoding_key[0].all() == encoding_key_ori[0].all()
    else:  # Or we can just load the keys from a file
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            encoding_key, decoding_key = pickle.load(f)
        print(f'Loaded PRC keys from file keys/{exp_id}.pkl')
elif method == 'gs':
    gs_watermark = Gaussian_Shading_chacha(ch_factor=1, hw_factor=8, fpr=fpr, user_number=10000)
    if not os.path.exists(f'keys/{exp_id}.pkl'):
        watermark_m_ori, key_ori, nonce_ori, watermark_ori = gs_watermark.create_watermark_and_return_w()
        with open(f'keys/{exp_id}.pkl', 'wb') as f:
            pickle.dump((watermark_m_ori, key_ori, nonce_ori, watermark_ori), f)
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            watermark_m, key, nonce, watermark = pickle.load(f)
        assert watermark_m.all() == watermark_m_ori.all()
    else:  # Or we can just load the keys from a file
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            watermark_m, key, nonce, watermark = pickle.load(f)
            print(f'Loaded GS keys from file keys/{exp_id}.pkl')
elif method == 'tr':
    # need to generate watermark key for the first time then save it to a file, we just load previous key here
    tr_key = '7c3fa99795fe2a0311b3d8c0b283c5509ac849e7f5ec7b3768ca60be8c080fd9_0_10_rand'
    # tr_key = '4145007d1cbd5c3e28876dd866bc278e0023b41eb7af2c6f9b5c4a326cb71f51_0_9_rand'
    print('Loaded TR keys from file')
else:
    raise NotImplementedError

if dataset_id == 'coco':
    save_folder = f'./results/{exp_id}_coco/original_images'
else:
    save_folder = f'./results/{exp_id}/original_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print(f'Saving original images to {save_folder}')

random.seed(42)
if dataset_id == 'coco':
    with open('coco/captions_val2017.json') as f:
        all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
else:
    all_prompts = [sample['Prompt'] for sample in load_dataset(dataset_id)['test']]

prompts = random.sample(all_prompts, test_num)

pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

def seed_everything(seed, workers=False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed
cur_inv_order = 0
# for i in tqdm(range(2)):
for i in tqdm(range(test_num)):
    seed_everything(i)
    current_prompt = prompts[i]
    if nowm:
        init_latents_np = np.random.randn(1, 4, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float64).to(device)
    else:
        if method == 'prc':
            prc_codeword = Encode(encoding_key)
            init_latents = prc_gaussians.sample(prc_codeword).reshape(1, 4, 64, 64).to(device)
        elif method == 'gs':
            init_latents = gs_watermark.truncSampling(watermark_m)
        elif method == 'tr':
            shape = (1, 4, 64, 64)
            init_latents, _, _ = tr_get_noise(shape, from_file=tr_key, keys_path='keys/')
        else:
            raise NotImplementedError
    orig_image, _, _ = generate(prompt=current_prompt,
                                init_latents=init_latents,
                                num_inference_steps=args.inf_steps,
                                solver_order=1,
                                pipe=pipe
                                )
    orig_image.save(f'{save_folder}/{i}.png')
    seed_everything(i)
    reversed_latents = init_latents.detach().clone()
    print(torch.sum((reversed_latents - init_latents)**2))
    if attack == 'white_noise':
        reversed_latents_attack = add_white_noise(reversed_latents, eps)
    elif attack == 'stealthy':
        reversed_latents_attack = add_stealthy_attack(reversed_latents, eps)
    elif attack == 'min_distortion':
        reversed_latents_attack = add_min_distortion_attack(reversed_latents, eps)
    elif attack == 'clustering':
        reversed_latents_attack = add_clustering_attack(reversed_latents, eps)
    print(torch.sum((reversed_latents - reversed_latents_attack)**2)**0.5)
    orig_image, _, _ = generate(prompt=current_prompt,
                            init_latents=reversed_latents_attack,
                            num_inference_steps=args.inf_steps,
                            solver_order=1,
                            pipe=pipe
                            )
    orig_image.save(f'{save_folder}/{i}_remove_{attack}.png')
    
        # test watermark 
    reversed_latents = exact_inversion(orig_image,
                                       prompt=current_prompt,
                                       test_num_inference_steps=args.inf_steps,
                                       inv_order=cur_inv_order,
                                       pipe=pipe
                                       )
    if method == 'prc':
        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(1.5)).flatten().cpu()
        detection_result = Detect(decoding_key, reversed_prc)
        decoding_result = (Decode(decoding_key, reversed_prc) is not None)
        combined_result = detection_result or decoding_result
#         combined_results.append(combined_result)
        print(f'{i:03d}: Detection: {detection_result}; Decoding: {decoding_result}; Combined: {combined_result}')
    elif method == 'gs':
        gs_watermark = Gaussian_Shading_chacha(ch_factor=1, hw_factor=8, fpr=fpr, user_number=10000)
        
        gs_watermark.nonce=nonce
        gs_watermark.key=key
        gs_watermark.watermark=watermark
        
        acc_metric = gs_watermark.eval_watermark(reversed_latents)
        combined_results.append(combined_result)

print(f'Done generating {method} images')