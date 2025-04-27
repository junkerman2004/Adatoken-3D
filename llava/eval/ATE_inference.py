# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# %%
import argparse
import torch
import numpy as np
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
import csv
from llava.constants import (IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN,DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN,IMAGE_PLACEHOLDER,)
from llava.mm_utils import (process_images,process_videos,tokenizer_special_token,get_model_name_from_path,)
import requests
from PIL import Image
from io import BytesIO
import re
import json
from tqdm import tqdm
from transformers import TextStreamer
from datasets import load_from_disk, load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
from sklearn.preprocessing import MinMaxScaler

def calculate_information_contribution(output_attentions,layers,image_token_id,prompt_id,total_id,q=0.3,k=0.7):
    #Calculate intra_modal_information
    intra_modal_information = []
    inter_modal_information = []
    image_token_id=int(image_token_id/10)
    prompt_id=int(prompt_id/10)
    total_id=int(total_id/10)
    for i in output_attentions:
        for j in range(layers):
            print("Layer: ", j)
            intra_modal_vlaues=0
            inter_modal_values=0
            averaged_attention = torch.mean(i[0][j],axis=1)[0].float()  # Shape: (n_tokens, n_tokens)
            averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0),10,stride=10).squeeze(0).squeeze(0)
            for x_id in range(total_id):
                for y_id in range(total_id):
                    if image_token_id<=x_id<prompt_id and image_token_id<=y_id<prompt_id and x_id<y_id:
                        intra_modal_vlaues+= averaged_attention[x_id, y_id]*100
                    elif x_id<image_token_id and image_token_id<=y_id<prompt_id:
                        inter_modal_values+=k*averaged_attention[x_id, y_id]*100
                    elif image_token_id<x_id<prompt_id and y_id>=prompt_id:
                        inter_modal_values+=q*averaged_attention[x_id, y_id]*100
            inter_modal_information.append(inter_modal_values)
            intra_modal_information.append(intra_modal_vlaues)


    return intra_modal_information,inter_modal_information
    #Calculate inter_modal_information
def calculate_EMA_information(intra_modal,inter_modal,layers,alpha=0.4,beta=0.3, gamma=0.3):
    #Calculate EMA_information
    multimodal_information=intra_modal[0]
    contribution=[]
    for i in range(1,layers):
        information_contribution=0
        multimodal_information+=intra_modal[i]*0.73
        information_contribution=intra_modal[i]*alpha+inter_modal[i]*beta+gamma*multimodal_information
        contribution.append(information_contribution)
    contribution = np.array([t.cpu().numpy() for t in contribution])

    contribution_2d = contribution.reshape(-1, 1)
    # 归一化（以 Min-Max 为例）
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(contribution_2d)
    contribution=normalized.flatten()
    print(contribution[0]*100,contribution[1]*100,contribution[2]*100, contribution[3]*100,contribution[4]*100,contribution[5]*100,contribution[6]*100,contribution[7]*100)




    return contribution


def exp_decay(x, A, B, C, D):
    return A * np.exp(-B * (x - C)) + D

def calculate_derivatives(data):
    return np.gradient(data)


def fit_function(x, y, a):
    # 确保 x 从 1 到 32
    x = np.arange(1, len(x) + 1)
    # 计算数据的导数
    derivatives = calculate_derivatives(y)


    def objective(params):
        A, B, C, D = params
        fitted_y = exp_decay(x, A, B, C, D)
        # 计算拟合值与原始值的误差
        error_values = np.sum((fitted_y - y) ** 2)
        # 计算拟合函数导数与原始数据导数的误差
        fitted_derivatives = np.gradient(fitted_y)
        error_derivatives = np.sum((fitted_derivatives - derivatives) ** 2)
        # 结合两个误差
        return error_values + error_derivatives


    initial_guess = [100, 0.1, 1, 0]


    def constraint(params):
        A, B, C, D = params
        return [
            exp_decay(1, A, B, C, D) - 100,
            exp_decay(32, A, B, C, D) - a
        ]

    from scipy.optimize import minimize
    result = minimize(objective, initial_guess, constraints={'type': 'eq', 'fun': constraint})
    params = result.x
    return params



def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def save_attention_to_json(file_path, attention_data):
    """
    将注意力权重保存为 JSON 文件。

    :param file_path: 输出文件路径
    :param attention_data: 注意力权重数据
    """
    attention_list = []
    for i ,att in enumerate(attention_data):
        print(f"Layer {i} attention:")
        if isinstance(att, tuple):  # 如果是 tuple，则展开
            layer_attention = []

            for a in att:
                if isinstance(a, torch.Tensor):  # 只处理张量

                    layer_attention.append(a.float().cpu().numpy().tolist())
                else:  # 忽略其他类型
                    print("还需更改")
                    continue
            attention_list.append(layer_attention)
        elif isinstance(att, torch.Tensor):  # 如果是单个张量
            attention_list.append(att.numpy().tolist())

    # 保存为 JSON 文件
    with open(file_path, 'w') as f:
        json.dump(attention_list, f)
def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image
def str_to_list(value):
    return [int(item.strip()) for item in value.split(',')]

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args):
    # Model
    with torch.no_grad():
        disable_torch_init()

        outputs = []
        outputs_attention = []
        torch_dtype = torch.float32
        if args.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif args.precision == "fp16":
            torch_dtype = torch.half
        if isinstance(args, torch.Tensor):
            args = args.cuda(2)
        mode = None

        if args.video_path:
            print(f"Video path provided: {args.video_path}")
            mode = 'video'
        if args.image_file:
            print(f"Image file provided: {args.image_file}")
            mode = 'imagejj'

        model_name = get_model_name_from_path(args.model_path)
        if args.layer_list is not None:
            Adatoken_infer = False  # whether to use pdrop infer
        tokenizer, model, processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name, Adatoken_infer
        )

        model_class_name = type(model).__name__
        if model_class_name == "LlavaLlamaForCausalLM_PDrop":
            model.model.layer_list = eval(args.layer_list)
            model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
            model.model.image_token_ratio_list.insert(0, 1.0)

        total_layers = model.config.num_hidden_layers


        qs = args.query

        matches = re.search(r"\[([^\]]+)\]", qs)
        if matches:
            coord_list = [float(x) for x in matches.group(1).split(',')]
            coord_list = [round(coord, 3) for coord in coord_list[:3]]
            qs = re.sub(r"\[([^\]]+)\]", "<boxes>", qs)
            clicks = torch.tensor([coord_list])
        else:
            clicks = torch.zeros((0,3))

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "3D" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print("prom:",prompt)

        if mode == 'image':
            image_files = image_parser(args)
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                processor['image'],
                model.config
            ).to(model.device, dtype=torch_dtype)
            depths_tensor = None
            poses_tensor = None
            intrinsics_tensor = None
            clicks_tensor = None

        if mode == 'video':
            videos_dict = process_videos(
                args.video_path,
                processor['video'],
                mode='random',
                device=model.device,
                text=args.query
            )

            images_tensor = videos_dict['images'].to(model.device, dtype=torch_dtype)
            depths_tensor = videos_dict['depths'].to(model.device, dtype=torch_dtype)
            poses_tensor = videos_dict['poses'].to(model.device, dtype=torch_dtype)
            intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch_dtype)
            clicks_tensor = clicks.to(model.device, dtype=torch.bfloat16)

        input_ids = (
            tokenizer_special_token(prompt, tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        input_ids=input_ids.to(model.device)


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                depths=depths_tensor,
                poses=poses_tensor,
                attention_mask=None,
                intrinsics=intrinsics_tensor,
                clicks=clicks_tensor,
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,

                )


        print(output_ids.keys())

        output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:],
                                  skip_spectial_tokens=True).strip().replace("</s>", "")
        outputs.append(output)

        output_attention=output_ids["attentions"]
        #save_attention_to_json('output_attention.json', output_attention)
        outputs_attention.append(output_attention)

    return outputs, outputs_attention,total_layers




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-path", type=str, help="Path to the video file")
    group.add_argument("--image-file", type=str, help="Path to the image file")

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--output-path",type=str,default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--layer_list", type=str, default="[4,8,12,16,18,20,22,24,26,28]")
    parser.add_argument("--image_token_ratio_list", type=str, default="[0.9,0.8,0.75,0.7,0.6,0.55,0.5,0.45,0.4,0.25]")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    torch.cuda.empty_cache()  # 释放未被占用的显存
    model_output_ori,outputs_attention,total_layers = eval_model(args)
    output_path = args.output_path
    a,b=calculate_information_contribution(outputs_attention, total_layers,37,1686,1720)
    y=calculate_EMA_information(a,b,total_layers)
    print(y.shape)
    a = 10  # 定义 (32, a) 中的 a
    x = np.arange(2, 33)
    params = fit_function(x, y, a)


    fitted_y = exp_decay(x, *params)

    for i in range(len(x)):
         print(f"x = {x[i]}, fitted_y = {fitted_y[i]}")
