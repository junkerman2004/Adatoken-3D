import json
test_scene={"scene0011_00","scene0015_00","scene0019_00","scene0025_00","scene0030_00","scene0046_00","scene0050_00","scene0063_00","scene0064_00",
            "scene0077_00","scene0084_00","scene0153_00","scene0169_00","scene0196_00","scene0300_00","scene0406_00","scene0435_00","scene0593_00",
            "scene0616_00","scene0645_00","scene0678_00","scene0702_00"}
for name in test_scene:
    # 定义两个 JSON 文件的路径
    file1_path = "/home/zk/llava_3D/playground/data/annotations/embodiedscan_infos_full.json"  # 第一个 JSON 文件
    file2_path = f"/home/Dataset/isaacsim/scannet/V2.0/scans/{name}.json"  # 第二个 JSON 文件
    output_path = "/home/zk/llava_3D/playground/data/annotations/embodiedscan_infos_full.json"  # 合并后的输出文件

    # 读取第一个 JSON 文件
    with open(file1_path, "r") as f1:
        data1 = json.load(f1)

    # 读取第二个 JSON 文件
    with open(file2_path, "r") as f2:
        data2 = json.load(f2)

    # 合并两个 JSON 文件的内容
    # 假设两个文件的结构都是 {"scannet/scene0191_00": {...}}
    for scene_key in data2:
        print(scene_key)
        if scene_key in data1:
            # 如果场景键已存在，合并内部的内容
            data1[scene_key].update(data2[scene_key])
        else:
            # 如果场景键不存在，直接添加
            data1[scene_key] = data2[scene_key]

    # 将合并后的内容保存到新的 JSON 文件
    with open(output_path, "w") as f_out:
        json.dump(data1, f_out)  # indent=4 用于美化输出，可以去掉

    print(f"合并后的 JSON 文件已保存为 {output_path}")