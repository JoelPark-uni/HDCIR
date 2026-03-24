import os
from typing import List, Dict

import argparse
import clip
import numpy as np
import termcolor
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import compute_results
import data_utils
import cir_datasets as datasets
from encoder import HDlm, encode_index_features_hdc
import prompts
import utils


def main():
    ### Load Input Arguments.
    parser = argparse.ArgumentParser()
    # Base Arguments
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--device", type=int, default=0, 
                        help="GPU ID to use.")
    parser.add_argument("--preload", nargs='+', type=str, default=['img_features','captions','mods'],
                        help='List of properties to preload is computed once before.')    
    parser.add_argument("--blip", type=str, default='Salesforce/blip2-flan-t5-xxl',
                        help="BLIP2 model ID from HuggingFace.")
    
    # Base Model Choices
    parser.add_argument("--clip", type=str, default='ViT-B/32', 
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50x4', 'ViT-bigG-14',
                                 'ViT-B-32','ViT-B-16','ViT-L-14','ViT-H-14','ViT-g-14'],
                        help="Which CLIP text-to-image retrieval model to use"),
    
    #################################################################################################################
    # Dataset Arguments ['dress', 'toptee', 'shirt']
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=['cirr', 'circo',
                                 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
                                 'toyfashion_iq',
                                 'genecis_change_attribute', 'genecis_change_object', 'genecis_focus_attribute', 'genecis_focus_object'],
                        help="Dataset to use")
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on. Some datasets require special testing protocols s.a. cirr/circo.')
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument("--caption-file", type=str, default=None,
                        help="Path to caption file for ToyFashionIQ (e.g., valid_cap.dress.val.json)")
    available_prompts = [f'prompts.{x}' for x in prompts.__dict__.keys() if '__' not in x]
    parser.add_argument("--llm_prompt", default='prompts.simple_modifier_prompt', type=str, choices=available_prompts,
                        help='Denotes the base prompt to use to probe the LLM. Has to be available in prompts.py')

    parser.add_argument("--llm_batch_size", default=16, type=int,
                        help='Batch size to use when generating LLM-based caption modifications. Default is 8, but can be set to 1 for lower GPU memory usage.')
    parser.add_argument("--use_hdc", action='store_true',
                        help='Use HDlm style encoder for positive/negative text features and HD cosine metric.')
    parser.add_argument("--HD_DIM", default=10000, type=int,
                        help='Hypervector dimension for HDlm encoder.')
    #################################################################################################################

    parser.add_argument("--weight-path", type=str, default='',
                        help='Where to store OpenCLIP weights.')
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    # LLM & BLIP Prompt Arguments.
    parser.add_argument("--blip_prompt", default='prompts.blip_prompt', type=str, choices=available_prompts,
                        help='Denotes the base prompt to use alongside BLIP. Has to be available in prompts.py')    
    # Text-to-Image Retrieval Arguments.
    parser.add_argument("--retrieval", type=str, default='default', choices=['default'],
                        help='Type of T2I Retrieval method.')
    args = parser.parse_args()


    ### Set Device.
    termcolor.cprint(f'Starting evaluation on {args.dataset.upper()} (split: {args.split})\n', color='green', attrs=['bold'])
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


    ### Argument Checks.
    preload_dict = {key: None for key in ['img_features', 'captions', 'mods', 'hd_index_features']}
    preload_str = f'{args.exp_name}_{args.dataset}_{args.blip}_{args.clip}_{args.split}'.replace('/', '-')    
        
    if len(args.preload):
        os.makedirs('precomputed', exist_ok=True)    
    if 'img_features' in args.preload:
        # # CLIP embeddings only have to be computed when CLIP model changes.
        # img_features_load_str = f'{args.dataset}_{args.clip}_{args.split}'.replace('/', '-')    
        preload_dict['img_features'] = os.path.join('precomputed', preload_str + '_img_features.pkl')
    
    if 'captions' in args.preload:
        # # BLIP captions only have to be computed when BLIP model or BLIP prompt changes.
        caption_load_str = f'{args.dataset}_{args.blip}_{args.split}'.replace('/', '-')    
        if args.blip_prompt != 'prompts.blip_prompt':
            preload_dict['captions'] = os.path.join('precomputed', caption_load_str + f'{args.exp_name}_captions_{args.blip_prompt.split(".")[-1]}.pkl')
        else:
            preload_dict['captions'] = os.path.join('precomputed', caption_load_str + f'{args.exp_name}_captions.pkl')
            
    if 'mods' in args.preload:
        # # LLM-based caption modifications have to be queried only when BLIP model or BLIP prompt changes.
        mod_load_str = f'{args.dataset}_{args.blip}_{args.split}'.replace('/', '-')    
        preload_dict['mods'] = os.path.join('precomputed', mod_load_str + f'{args.exp_name}_mods_{args.llm_prompt.split(".")[-1]}.json')

    if args.use_hdc and len(args.preload):
        preload_dict['hd_index_features'] = os.path.join(
            'precomputed',
            preload_str + f'_hd_index_features_HD{args.HD_DIM}.pt'
        )
    
    if args.split == 'test':
        preload_dict['test'] = preload_str + f'{args.exp_name}_{args.blip_prompt.split(".")[-1]}_{args.llm_prompt.split(".")[-1]}_test_submission.json'
    
    ### Load CLIP model, BLIP model & Preprocessing.    
    print(f'Loading CLIP {args.clip}... ', end='')
          
    if args.clip in ['ViT-bigG-14','ViT-B-32','ViT-B-16','ViT-L-14','ViT-H-14','ViT-g-14']:
        import open_clip
        pretraining = {
            'ViT-B-32':'laion2b_s34b_b79k',
            'ViT-B-16':'laion2b_s34b_b88k',
            'ViT-L-14':'laion2b_s32b_b82k',
            'ViT-H-14':'laion2b_s32b_b79k',
            'ViT-g-14':'laion2b_s34b_b88k',
            'ViT-bigG-14':'laion2b_s39b_b160k'
        }
        if args.weight_path == '':
            weight_path = os.path.join(args.dataset_path, '..', 'weights', 'open_clip')
        else:
            weight_path = os.path.join(os.getcwd(), args.weight_path)
        os.makedirs(weight_path, exist_ok=True)
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip, pretrained=pretraining[args.clip], cache_dir=weight_path)
        clip_model = clip_model.eval().requires_grad_(False).to(device)
        tokenizer = open_clip.get_tokenizer(args.clip)
        clip_model.tokenizer = tokenizer
    else:
        clip_model, clip_preprocess = clip.load(args.clip, device=device, jit=False)
        clip_model = clip_model.float().eval().requires_grad_(False).to(device)

    print('Done.')
    
    if args.preprocess_type == 'targetpad':
        print('Target pad preprocess pipeline is used.')
        preprocess = data_utils.targetpad_transform(1.25, clip_preprocess.transforms[0].size)
    elif args.preprocess_type == 'clip':
        print('CLIP preprocess pipeline is used.')
        preprocess = clip_preprocess
        
    print(f'Loading BLIP2 {args.blip}... ', end='')


    blip2_processor = Blip2Processor.from_pretrained(args.blip)
    blip_model = None
    if preload_dict['captions'] is None or not os.path.exists(preload_dict['captions']):
        load_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            args.blip,
            device_map='auto' if torch.cuda.is_available() else None,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True
        )

        # Safety alias in case HF internals return nested language_model keys only.
        if hasattr(blip_model, "hf_device_map") and "language_model" not in blip_model.hf_device_map:
            lm_keys = [k for k in blip_model.hf_device_map.keys() if k.startswith("language_model.")]
            if len(lm_keys):
                blip_model.hf_device_map["language_model"] = blip_model.hf_device_map[lm_keys[0]]

        blip_model.eval()
        print('Done.')
    else:
        print(f'Skipped (captions precomputed).')
    
    ### Load Evaluation Datasets.
    target_datasets, query_datasets, pairings = [], [], []
    
    if 'fashioniq' in args.dataset.lower():
        dress_type = args.dataset.split('_')[-1]
        target_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'classic', preprocess))
        query_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'relative', preprocess))
        pairings.append(dress_type)
        compute_results_function = compute_results.fiq
    
    elif args.dataset.lower() == 'toyfashion_iq':
        # Load ToyFashionIQ dataset from a specific caption file
        if args.caption_file is None:
            caption_file = '/workspace/joel/InstructCIR/valid_cap.dress.val.json'
        else:
            caption_file = args.caption_file
        target_datasets.append(datasets.ToyFashionIQDataset(args.dataset_path, caption_file, 'classic', preprocess))
        query_datasets.append(datasets.ToyFashionIQDataset(args.dataset_path, caption_file, 'relative', preprocess))
        pairings.append('toyfashioniq')
        compute_results_function = compute_results.fiq
    
    elif args.dataset.lower() == 'cirr':
        split = 'test1' if args.split == 'test' else args.split
        target_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'classic', preprocess))
        query_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'relative', preprocess))
        compute_results_function = compute_results.cirr
        pairings.append('default')
        
    elif args.dataset.lower() == 'circo':
        target_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'classic', preprocess))
        query_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'relative', preprocess))
        compute_results_function = compute_results.circo
        pairings.append('default')
    
    elif 'genecis' in args.dataset.lower():   
        prop_file = '_'.join(args.dataset.lower().split('_')[1:])
        prop_file = os.path.join(args.dataset_path, 'genecis', prop_file + '.json')
        
        if 'object' in args.dataset.lower():
            datapath = os.path.join(args.dataset_path, 'coco2017', 'val2017')
            genecis_dataset = datasets.COCOValSubset(root_dir=datapath, val_split_path=prop_file, transform=preprocess)                
        elif 'attribute' in args.dataset.lower():            
            datapath = os.path.join(args.dataset_path, 'Visual_Genome', 'VG_All')
            genecis_dataset = datasets.VAWValSubset(image_dir=datapath, val_split_path=prop_file, transform=preprocess)
            
        target_datasets.append(genecis_dataset)
        query_datasets.append(genecis_dataset)
        compute_results_function = compute_results.genecis
        pairings.append('default')
                
    ### Evaluate performances.
    for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
        termcolor.cprint(f'\n------ Evaluating Retrieval Setup: {pairing}', color='yellow', attrs=['bold'])
        
        ### General Input Arguments.
        input_kwargs = {
            'args': args, 'query_dataset': query_dataset, 'target_dataset': target_dataset, 'clip_model': clip_model, 
            'blip_model': blip_model, 'blip_processor': blip2_processor, 'preprocess': preprocess, 'device': device, 'split': args.split,
            'preload_dict': preload_dict,
        }    
        
        ### Compute Target Image Features
        print(f'Extracting target image features using CLIP: {args.clip}.')
        index_features, index_names, index_ranks, aux_data = utils.extract_image_features(
            device, args, target_dataset, clip_model, preload=preload_dict['img_features'])
        index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)
        input_kwargs.update({'index_features': index_features, 'index_names': index_names, 'index_ranks': index_ranks})

        if args.use_hdc and index_features.ndim == 2:
            print(f'Preparing HDlm encoder (HD_DIM={args.HD_DIM}) and index hypervectors.')
            hdc_encoder = HDlm(feature_dim=index_features.shape[-1], HD_DIM=args.HD_DIM, device='cpu')

            if preload_dict['hd_index_features'] is not None and os.path.exists(preload_dict['hd_index_features']):
                print(f'Loading precomputed HD index features from {preload_dict["hd_index_features"]}!')
                hdc_index_features = torch.load(preload_dict['hd_index_features'], map_location='cpu')
            else:
                hdc_index_features = encode_index_features_hdc(index_features.cpu(), hdc_encoder, output_device='cpu')
                if preload_dict['hd_index_features'] is not None:
                    torch.save(hdc_index_features.cpu(), preload_dict['hd_index_features'])

            input_kwargs.update({'hd_encoder': hdc_encoder, 'hdc_index_features': hdc_index_features.cpu()})
        elif args.use_hdc:
            print('Skipping HDC index encoding because index_features is not 2D for this dataset.')

            
        ### Compute Method-specific Query Features.
        # This part can be interchanged with any other method implementation.
        print(f'Generating conditional query predictions (CLIP: {args.clip}, BLIP: {args.blip}).')
        out_dict = utils.generate_predictions(**input_kwargs)
        input_kwargs.update(out_dict)
        
        ### Compute Dataset-specific Retrieval Scores.
        # This part is dataset-specific and declared above.
        print('Computing final retrieval metrics.')
        if args.dataset == 'genecis_focus_attribute':
            aux_data['ref_features'] = torch.nn.functional.normalize(aux_data['ref_features'].float().to(device))
            out_dict['predicted_features'] = torch.nn.functional.normalize(
                (out_dict['predicted_features'].float() + aux_data['ref_features'])/2, dim=-1)

        input_kwargs.update(out_dict)        
        result_metrics = compute_results_function(**input_kwargs)        
        
        # Print metrics.
        print('\n')
        if result_metrics is not None:
            termcolor.cprint(f'Metrics for {args.dataset.upper()} ({args.split})- {pairing}', attrs=['bold'])
            for k, v in result_metrics.items():
                print(f"{pairing}_{k} = {v:.2f}")        
        else:
            termcolor.cprint(f'No explicit metrics available for {args.dataset.upper()} ({args.split}) - {pairing}.', attrs=['bold'])            



if __name__ == '__main__':
    main()
