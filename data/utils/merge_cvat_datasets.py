"""
功能：合并两个 CVAT 数据集（图片 + 标注 XML + 问题 TXT）

主要操作：
1. 合并两份图片，重命名为 dashboard_0001.png 依次递增。
2. 合并两份 XML 文件，仅更新 <image name="..."> 对应的新图片名。
3. 拼接两份 questions.txt，保持空行和原内容，第二份的问题序号在第一份之后递增

使用：
python merge_cvat_datasets.py \
    --images1 path/to/dataset1/images \
    --xml1 path/to/dataset1/annotations.xml \
    --txt1 path/to/dataset1/questions.txt \
    --images2 path/to/dataset2/images \
    --xml2 path/to/dataset2/annotations.xml \
    --txt2 path/to/dataset2/questions.txt \
    --output_dir path/to/output

输出：
merged_dataset/
 ├─ images/
 ├─ merged_annotations.xml
 └─ merged_questions.txt
"""
import os
import shutil
import xml.etree.ElementTree as ET
import re

def merge_cvat_datasets(dataset1_path, dataset2_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    images_out = os.path.join(output_path, "images")
    os.makedirs(images_out, exist_ok=True)

    xml1 = os.path.join(dataset1_path, "annotations.xml")
    xml2 = os.path.join(dataset2_path, "annotations.xml")
    txt1 = os.path.join(dataset1_path, "questions.txt")
    txt2 = os.path.join(dataset2_path, "questions.txt")
    images1 = os.path.join(dataset1_path, "images")
    images2 = os.path.join(dataset2_path, "images")

    # === Step 1: 合并图片 ===
    all_images = []
    counter = 1
    for img_dir in [images1, images2]:
        img_files = sorted(os.listdir(img_dir))
        for img_file in img_files:
            old_path = os.path.join(img_dir, img_file)
            if not os.path.isfile(old_path):
                continue
            new_name = f"dashboard_{counter:04d}.png"
            shutil.copy2(old_path, os.path.join(images_out, new_name))
            all_images.append(new_name)
            counter += 1

    # === Step 2: 合并 XML ===
    tree1 = ET.parse(xml1)
    tree2 = ET.parse(xml2)
    root1, root2 = tree1.getroot(), tree2.getroot()

    imgs1 = root1.findall("image")
    imgs2 = root2.findall("image")

    offset = len(imgs1)
    for i, node in enumerate(imgs2):
        node.set("id", str(offset + i))

    all_nodes = imgs1 + imgs2
    for i, node in enumerate(all_nodes):
        node.set("name", all_images[i])

    for n in root1.findall("image"):
        root1.remove(n)
    for node in all_nodes:
        root1.append(node)

    merged_xml_path = os.path.join(output_path, "merged_annotations.xml")
    tree1.write(merged_xml_path, encoding="utf-8")

    # === Step 3: 合并 questions.txt ===
    def find_last_image_index(txt_path):
        """找到最后一个问题的图片编号（如9.3 -> 返回9）"""
        last_idx = 0
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"(\d+)\.\d+", line.strip())
                if match:
                    last_idx = int(match.group(1))
        return last_idx

    last_index_1 = find_last_image_index(txt1)
    current_image_idx = last_index_1

    merged_txt_path = os.path.join(output_path, "merged_questions.txt")
    with open(merged_txt_path, "w", encoding="utf-8") as out:
        # 写入 dataset1 原始内容
        with open(txt1, "r", encoding="utf-8") as f1:
            out.write(f1.read().rstrip() + "\n\n")

        # 写入 dataset2，编号递增
        with open(txt2, "r", encoding="utf-8") as f2:
            prev_img = None
            q_idx = 0
            for line in f2:
                line = line.strip()
                if not line:
                    out.write("\n")
                    continue
                match = re.match(r"(\d+)\.(\d+)\s+(.*)", line)
                if match:
                    img_idx, qnum, content = match.groups()
                    if img_idx != prev_img:
                        current_image_idx += 1
                        q_idx = 1
                        prev_img = img_idx
                    else:
                        q_idx += 1
                    out.write(f"{current_image_idx}.{q_idx} {content}\n")
                else:
                    out.write(line + "\n")

    print("✅ 合并完成！")
    print(f"输出目录: {output_path}")
    print(f"共合并 {len(all_images)} 张图片。")
    print(f"新的 question 从 {last_index_1+1}.1 开始递增。")

# 示例用法
if __name__ == "__main__":
    merge_cvat_datasets(
        dataset1_path="./datasetA",
        dataset2_path="./datasetB",
        output_path="./merged_dataset"
    )