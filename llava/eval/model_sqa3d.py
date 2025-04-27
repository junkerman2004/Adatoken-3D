import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import argparse
import torch

import time
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_videos, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.video_folder:
        print(f"Video path provided: {args.video_folder}")
        mode = 'video'


    model_name = get_model_name_from_path(args.model_path)
    if args.layer_list is not None:
        pdrop_infer = False  # whether to use pdrop infer
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, pdrop_infer
    )

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    questions=questions["questions"]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    timer=0
    for line in tqdm(questions):
        idx = line["question_id"]
        video_file = line["scene_id"]
        video_path = os.path.join(args.video_folder, video_file)
        qs = line["question"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        videos_dict = process_videos(
            video_path,
            processor['video'],
            mode='random',
            device=model.device,
            text=cur_prompt
        )

        images_tensor = videos_dict['images'].to(model.device, dtype=torch.bfloat16)
        depths_tensor = videos_dict['depths'].to(model.device, dtype=torch.bfloat16)
        poses_tensor = videos_dict['poses'].to(model.device, dtype=torch.bfloat16)
        intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch.bfloat16)
        start_time = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                depths=depths_tensor,
                poses=poses_tensor,
                intrinsics=intrinsics_tensor,
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=512,
                use_cache=True,
            )
        end_time = time.time()
        a_time=end_time-start_time
        print("任务用时",a_time)
        timer+=a_time
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    print(f"Inference time: {int(timer):.3f} seconds")
    print(f"save time: {int(timer/911):.3f} seconds")

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava3d-v1.5-7b-task-v3-tuning")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="/home/zk/llava_3D/LLaVA-3D-Demo-Data/scannet/scannet")
    parser.add_argument("--question-file", type=str, default="playground/data/annotations/llava3d_sqa3d_val_question.json")
    parser.add_argument("--answers-file", type=str, default="./home/zk/llava_3D/sqa_3d/balanced/llava3d_sqa3d_val_answer_pred.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--layer_list", type=str, default="[8, 16, 24]")
    parser.add_argument("--image_token_ratio_list", type=str, default="[0.8,0.6,0.4]")
    args = parser.parse_args()

    eval_model(args)
