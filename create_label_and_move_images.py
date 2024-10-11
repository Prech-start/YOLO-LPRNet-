import os
import shutil
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


data_base_path = f'./datasets/CCPD2019/'

task = f'test'

split_filename = task + '.txt'


def flatten(l):
    for el in l:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def convert_to_yolo_format(image_path, label):
    # 打开图像以获取宽度和高度
    image = Image.open(image_path)
    width, height = image.size
    label = [int(i) for i in label]

    # 从label中获取边界框坐标
    x_min, y_min, x_max, y_max = label

    # 计算边界框中心点、宽度和高度
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # 归一化中心点、宽度和高度
    x_center_norm = x_center / width
    y_center_norm = y_center / height
    bbox_width_norm = bbox_width / width
    bbox_height_norm = bbox_height / height

    # 返回 YOLO 格式的 label（class_id x_center y_center width height）
    return [str(x_center_norm), str(y_center_norm), str(bbox_width_norm), str(bbox_height_norm)]


def draw_bbox(image_path, bbox):
    bbox = [int(i) for i in bbox]
    # 打开图像文件
    image = Image.open(image_path)

    # 创建一个绘图对象
    draw = ImageDraw.Draw(image)

    # 在图像上绘制边界框
    draw.rectangle(bbox, outline="red", width=3)

    # 使用 matplotlib 显示图像
    plt.imshow(image)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()

    # 保存绘制了bbox的新图像
    output_image_path = 'output_image_with_bbox.jpg'
    image.save(output_image_path)


with open(os.path.join(data_base_path, split_filename), 'rb+') as f:
    line = f.readline().decode().strip()
    while line:
        # print(line)
        license_plate_number = str(line).split('-')[-3]

        bounding_box_coordinates = [c.split('&') for c in
                                    str(line).split('-')[2].split('_')]
        # 归一化坐标
        bounding_box_coordinates = [i for i in flatten(bounding_box_coordinates)]

        # draw_bbox(os.path.join(data_base_path, 'raw', line), bounding_box_coordinates)

        # move_file
        raw_path = os.path.join(data_base_path, 'raw', line)
        target_folder = os.path.join(data_base_path, task, 'images', raw_path.split('/')[-1])

        if not os.path.exists(target_folder):
            shutil.copy(raw_path, target_folder)
        # normal
        label = convert_to_yolo_format(raw_path, bounding_box_coordinates)
        label = ['0'] + label
        label = [j + ' ' for j in label]
        label_filename = str(line).split('/')[-1].replace('.jpg', '.txt')
        save_path = os.path.join(data_base_path, task, 'labels', label_filename)
        with open(save_path, 'w') as f_:
            f_.writelines(label)

        line = f.readline().decode().strip()
