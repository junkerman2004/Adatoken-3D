import os
import json
import re
import numpy as np
# 定义文件夹路径
test_scene={"scene0011_00","scene0015_00","scene0019_00","scene0025_00","scene0030_00","scene0046_00","scene0050_00","scene0063_00","scene0064_00",
            "scene0077_00","scene0084_00","scene0153_00","scene0169_00","scene0196_00","scene0300_00","scene0406_00","scene0435_00","scene0593_00",
            "scene0616_00","scene0645_00","scene0678_00","scene0702_00"}
for name in test_scene:
    base_path = f"/home/Dataset/isaacsim/scannet/V2.0/scans/{name}"
    scene_name=os.path.split(base_path)[-1]
    # 初始化 JSON 数据结构
    data = {
        f"scannet/{scene_name}": {}
    }
    pose_path=os.path.join(base_path,"pose")
    intrinsic_path=os.path.join(base_path,"intrinsic")
    matrix_path=os.path.join(base_path,scene_name+".txt")
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(pose_path):
        if file_name.endswith(".txt"):  # 只处理 .txt 文件
            # 构建完整的文件路径
            txt_file_path = os.path.join(pose_path, file_name)
            match = re.match(r'(\d+)\.(txt)', file_name)
            file_number = int(match.group(1))
            number = match.group(1)  # 提取数字部分
            extension = match.group(2)  # 提取文件扩展名
            if int(number) % 10==0:
                formatted_number = f"{int(number):05d}"  # 使用 f-string 格式化
                new_filename = f"{formatted_number}.{extension}"  # 构建新文件名
                # 读取 .txt 文件中的 4x4 矩阵
                with open(txt_file_path, "r") as f:
                    lines = f.readlines()
                    pose = []
                    for line in lines:
                        # 将每行转换为浮点数列表
                        row = list(map(float, line.strip().split()))
                        pose.append(row)

                # 构建对应的图像文件名（将 .txt 替换为 .jpg 和 .png）
                image_name = f"{formatted_number}.jpg"

                depth_name =f"{formatted_number}.png"

                # 添加到 JSON 数据结构中
                data[f"scannet/{scene_name}"][f"scannet/posed_images/{scene_name}/{image_name}"] = {
                    "pose": pose,
                    "depth": f"scannet/posed_images/{scene_name}/{depth_name}"


                     }
            else:
                continue
    lines = open(matrix_path).readlines()
    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break

    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    axis_align_matrix=axis_align_matrix.tolist()
    print("axis_align_matrix=",axis_align_matrix)
    data[f"scannet/{scene_name}"]["axis_align_matrix"] = axis_align_matrix
    for file_name in os.listdir(intrinsic_path):
        intrinsic_name=os.path.join(intrinsic_path,file_name)
        real_name=file_name.split(".")[0]
        print(real_name)
        if real_name=="intrinsic_depth":
            with open(intrinsic_name, "r") as f:
                lines = f.readlines()
                depth_intrinsic= []
                for line in lines:
                    # 将每行转换为浮点数列表
                    row = list(map(float, line.strip().split()))
                    depth_intrinsic.append(row)

            data[f"scannet/{scene_name}"]["depth_intrinsic"] = depth_intrinsic
        if real_name == "intrinsic_color":
            with open(intrinsic_name, "r") as f:
                lines = f.readlines()
                intrinsic = []
                for line in lines:
                    # 将每行转换为浮点数列表
                    row = list(map(float, line.strip().split()))
                    intrinsic.append(row)

            data[f"scannet/{scene_name}"]["intrinsic"] = intrinsic
        else:
            continue

    # 将数据保存为 JSON 文件
    output_json_path = f"/home/Dataset/isaacsim/scannet/V2.0/scans/{scene_name}.json"
    with open(output_json_path, "w") as f:
        json.dump(data, f,indent=0)





