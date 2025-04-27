import json
test_scene=[
    "scene0011_00", "scene0015_00", "scene0019_00", "scene0025_00", "scene0030_00", "scene0046_00", "scene0050_00",
    "scene0064_00",
    "scene0084_00", "scene0153_00", "scene0169_00", "scene0300_00", "scene0406_00", "scene0435_00", "scene0593_00",
    "scene0616_00", "scene0645_00", "scene0678_00", "scene0702_00"
]

json_data=json.load(open("/home/zk/llava_3D/ScanQA/ScanQA_v1.0_val (1).json", 'r'))
# 筛选 questions 列表
filtered_questions = [q for q in json_data if q["scene_id"] in test_scene]

# 更新 JSON 数据中的 questions 列表
json_data = filtered_questions

output_file = "/home/zk/llava_3D/ScanQA/llava-3d-scanqa_val_question.json"  # 新文件的名称
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print(f"数据已成功写入文件: {output_file}")