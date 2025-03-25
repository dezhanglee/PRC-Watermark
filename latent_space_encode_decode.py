# do watermark removal on latent space
# no stable diffusion involved

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
from inversion import stable_diffusion_pipe, generate
from PIL import Image
from src.prc import Detect, Decode
from inversion import stable_diffusion_pipe, exact_inversion
from src.baseline.gs_watermark import Gaussian_Shading_chacha
import statistics

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=100)
parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts') # coco 
parser.add_argument('--attack', type=str, default='white_noise') #white_noise, min_distortion, stealthy
parser.add_argument('--inf_steps', type=int, default=20)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
parser.add_argument('--eps_low', type=float, default=10)
parser.add_argument('--eps_high', type=float, default=30)
parser.add_argument('--n_eps', type=int, default=1)

args = parser.parse_args()
print(args)

hf_cache_dir = 'hf_models'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
n = 4 * 64 * 64  # the length of a PRC codeword
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t

attack = args.attack
eps_low = args.eps_low
eps_high = args.eps_high
n_eps = args.n_eps
gap = (eps_high-eps_low)/n_eps
eps_arr = [eps_low + gap*(i) for i in range(n_eps+1)]

exp_id = f'{method}_{attack}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}_eps_low_{eps_low}_eps_high_{eps_high}_n_eps_{n_eps}'


def add_white_noise(input_latent, eps):
    wn = torch.normal(0, 1, input_latent.shape)
    wn_normalized = wn / (torch.sum(wn**2)**0.5) * (eps)
    dev = input_latent.get_device()
    out = input_latent + wn_normalized.to(dev)
    mean, std = torch.mean(out), torch.std(out)
    out = (out-mean)/std
    return out

def add_stealthy_attack(input_latent, eps):
    input_latent_flat = input_latent.flatten().detach().clone().to('cuda:1')
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
    
    return out.to('cuda:1')
       

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
# if dataset_id == 'coco':
#     with open('coco/captions_val2017.json') as f:
#         all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
# else:
#     all_prompts = [sample['Prompt'] for sample in load_dataset(dataset_id)['test']]

# prompts = random.sample(all_prompts, test_num)

# pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id, cache_dir=hf_cache_dir)
# pipe.set_progress_bar_config(disable=True)

def seed_everything(seed, workers=False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed

skip = 0
# for i in tqdm(range(2)):
counter = 0
for eps in tqdm(eps_arr):
    print(eps)
    if skip:
        print(skip)
        skip -= 1
        continue
    combined_results = []
    for i in tqdm(range(test_num)):

        seed_everything(counter)
        counter += 1
        # generate watermarked latent 
        if method == 'prc':
            prc_codeword = Encode(encoding_key)
            init_latents = prc_gaussians.sample(prc_codeword).reshape(1, 4, 64, 64).to(device)
        elif method == 'gs':
            init_latents = gs_watermark.truncSampling(watermark_m)
        else:
            raise NotImplementedError

        # modify latent
        if attack == 'white_noise':
            init_latents = add_white_noise(init_latents, eps)
        elif attack == 'stealthy':
            init_latents = add_stealthy_attack(init_latents, eps)
        elif attack == 'min_distortion':
            init_latents = add_min_distortion_attack(init_latents, eps)
        # decode

        if method == 'prc':
            init_latents = prc_gaussians.recover_posteriors(init_latents.to(torch.float64).flatten().cpu(), variances=float(1.0)).flatten().cpu()

            detection_result = Detect(decoding_key, init_latents)
            decoding_result = (Decode(decoding_key, init_latents) is not None)
            combined_result = detection_result and decoding_result
            combined_results.append((eps, combined_result))
            print(f'{i:03d}: Detection: {detection_result}; Decoding: {decoding_result}; Combined: {combined_result}')
        elif method == 'gs':
            gs_watermark.nonce=nonce
            gs_watermark.key=key
            gs_watermark.watermark=watermark

            acc_metric = gs_watermark.eval_watermark(init_latents)
            print((eps, acc_metric >= 0.5))
            combined_results.append((eps, acc_metric >= 0.5))
    with open(f'res_latent/latent_space_remove_{exp_id}.txt', 'a') as f:
        tpr = statistics.mean([int(rr[1]) for rr in combined_results])
        f.write(f'{eps} {tpr}\n')
    if all([not rr[1] for rr in combined_results]): # if all are removed, no point trying increased distortion 
        break
    if all([rr[1] for rr in combined_results]): # if all are removed, no point trying increased distortion 
        skip = 5
   


print(f'Decoded results saved to '+f'res_latent/latent_space_remove_{exp_id}.txt')