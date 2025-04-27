# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "5"
# %%
import argparse
import torch
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import argparse
import os
import csv
import torch
import numpy as np

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    process_videos,
    tokenizer_special_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
from PIL import Image
import json
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
from datasets import load_from_disk, load_dataset
import torch
import json
from tqdm import tqdm
import re

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm


def visualize_attention(multihead_attention, output_path="atten_map_1.png", title="Layer 5"):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()  # Shape: (n_tokens, n_tokens)

    # pooling the attention scores  with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20,
                                                        stride=20).squeeze(0).squeeze(0)

    cmap = plt.cm.get_cmap("coolwarm ")
    plt.figure(figsize=(5, 5), dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    # set the x and y ticks to 20x of the original

    ax = sns.heatmap(averaged_attention,
                     cmap=cmap,  # custom color map
                     norm=log_norm,  #
                     # cbar_kws={'label': 'Attention score'},
                     )

    # remove the x and y ticks

    # replace the x and y ticks with string

    x_ticks = [str(i * 20) for i in range(0, averaged_attention.shape[0])]
    y_ticks = [str(i * 20) for i in range(0, averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0, averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0, averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # change the x tinks font size
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)

    # make y label vertical
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    plt.title(title)
    # tight layout
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)

    return top_five_attentions, averaged_attention

def visual_information(multihead_attention, output_path="atten_map_1.png", title="Layer 5"):
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()  # Shape: (n_tokens, n_tokens)

    # 池化操作
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0),10,
                                                        stride=10).squeeze(0).squeeze(0)

    # 计算信息传导值
    information_conduction = torch.sum(averaged_attention, dim=0)
    calculation_conduction = information_conduction.numpy()
    # 确保数据为一行
    information_conduction = information_conduction.unsqueeze(0).numpy()
    arr = information_conduction[0]

    # 计算前两个数的平均值
    first_two_avg = (arr[0] + arr[1]) / 2

    # 计算后两个数的平均值
    last_two_avg = (arr[-1] + arr[-2]) / 2

    # 提取中间的数值
    middle_values = arr[2:-2]

    # 计算中间数值的平均值
    middle_avg = np.mean(middle_values)

    # 设置颜色映射和归一化
    cmap = plt.cm.get_cmap("coolwarm")
    # 使用线性归一化，这里用 LogNorm 可能不合适，因为是柱形图


    # 创建画布
    plt.figure(figsize=(10, 2))  # 调整画布高度以适应一行数据

    # 准备绘制柱形图的数据和标签
    averages = [first_two_avg, middle_avg, last_two_avg]
    labels = ['前两个平均', '中间平均', '后两个平均']
    log_norm = LogNorm(vmin=0.001, vmax=averaged_attention.max())
    # 根据数值生成对应的颜色
    original_length = len(arr)
    print('original_length', original_length)
    target_length = 30

    # 使用线性插值将86个点压缩到30个
    x_original = np.linspace(0, 1, original_length)
    x_target = np.linspace(0, 1, target_length)
    arr = np.interp(x_target, x_original, information_conduction[0])

    # 2. 创建高质量画布
    plt.figure(figsize=(15, 4), dpi=300)  # 更宽的画布适应更多柱子

    # 3. 自定义颜色映射
    cmap = plt.get_cmap("coolwarm")


    # 4. 绘制热力柱状图
    bars = plt.bar(range(target_length), arr,
                   color=cmap(log_norm(arr)),  # 根据值映射颜色
                   edgecolor='black', linewidth=0.5)

    # 5. 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=log_norm)
    sm.set_array([])
    # cbar = plt.colorbar(sm,pad=0.01)
    # cbar.set_label('Information Contribution', fontsize=10)

    # 6. 优化坐标轴
    plt.xticks(range(target_length), [f'{i + 1}' for i in range(target_length)],
               rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Token Position', fontsize=10)
    plt.ylabel('Contribution Score', fontsize=10)

    # 7. 添加标题并保存
    plt.title(f"{title} (Compressed {original_length}→{target_length})", fontsize=12)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    return first_two_avg, middle_avg, last_two_avg
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
            pdrop_infer = False  # whether to use pdrop infer
        tokenizer, model, processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name, pdrop_infer
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
    parser.add_argument("--layer_list", type=str, default="[4,8,12,16,18,20,]")
    parser.add_argument("--image_token_ratio_list", type=str, default="[0.75,0.5,0.4,0.3,0.25,0.15]")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    model_output_ori,outputs_attention,total_layers = eval_model(args)
    output_path = args.output_path

    try:
        os.mkdir(output_path)
    except:
        pass

    try:
        os.mkdir(output_path + "/attn_maps")
        os.mkdir(output_path + "/iformation_maps")
    except:
        pass

    with open(output_path + "/output.json", "w") as f:
        # json dumps
        json.dump({"prompt": args.query, "image": args.video_path, "output": model_output_ori}, f, indent=4)

    # draw attention maps
    with open('data_attention.csv', mode='w', newline='', encoding='utf-8') as file:
        # 创建一个 csv.writer 对象
        writer = csv.writer(file)
        # 写入多行数据

    for i in outputs_attention:
        System_contribution=[]
        Spatial_contribution=[]
        Prompt_contribution=[]


        for j in range(0, total_layers):
            top5_attention, average_attentions = visualize_attention(i[0][j].cpu(),output_path=output_path + "/attn_maps/atten_map_" + str(j) + ".png", title="Spatial attention" + str(j + 1))
