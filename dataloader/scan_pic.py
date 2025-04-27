import os
import shutil
import re
# 定义要处理的根目录
test_scene={"scene0011_00","scene0015_00","scene0019_00","scene0025_00","scene0030_00","scene0046_00","scene0050_00","scene0064_00",
            "scene0084_00","scene0153_00","scene0169_00","scene0300_00","scene0406_00","scene0435_00","scene0593_00",
            "scene0616_00","scene0645_00","scene0678_00","scene0702_00"}
for name in test_scene:
    root_dir = f'/home/Dataset/isaacsim/scannet/V2.0/scans/{name}'
    # 定义要挑选的文件名

    # 定义保存文件的新文件夹
    output_dir = f'/home/zk/llava_3D/LLaVA-3D-Demo-Data/scannet/scannet/posed_images/{name}'
    output_dir2 = f'/home/zk/llava_3D/LLaVA-3D-Demo-Data/scannet/scannet/{name}'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    # 遍历color和depth文件夹
    for folder in ['color', 'depth']:
        folder_path = os.path.join(root_dir, folder)

        # 遍历文件夹中的文件
        for filename in os.listdir(folder_path):
            match=re.match(r'(\d+)\.(jpg|png)', filename)
            file_number = int(match.group(1))
            number = match.group(1)  # 提取数字部分
            extension = match.group(2)  # 提取文件扩展名
            formatted_number = f"{int(number):05d}"  # 使用 f-string 格式化
            new_filename = f"{formatted_number}.{extension}"  # 构建新文件名
            # 如果文件名在目标文件中
            if file_number % 10==0:
                # 构建源文件路径和目标文件路径
                src_file = os.path.join(folder_path, filename)
                dst_file = os.path.join(output_dir, new_filename)
                ds_file=os.path.join(output_dir2, new_filename)

                # 复制文件到新文件夹
                shutil.copy(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")
                shutil.copy(src_file, ds_file)
                print(f"Copied {src_file} to {ds_file}")


