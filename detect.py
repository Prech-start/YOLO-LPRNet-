import cv2
import numpy as np
import os


# 定义透视变换函数
def four_point_transform(image, pts):
    # 获取一致顺序的点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算新图像的宽度和高度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 构建目标点集
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # 返回变换后的图像
    return warped


# 定义排序点的函数
def order_points(pts):
    # 初始化一个列表，用于存储排序后的坐标
    pts_ordered = np.zeros((4, 2), dtype='float32')
    # 计算每个点的坐标之和
    s = pts.sum(axis=1)
    pts_ordered[0] = pts[np.argmin(s)]
    pts_ordered[2] = pts[np.argmax(s)]
    # 根据x坐标对点进行排序
    diff = np.diff(pts, axis=1)
    pts_ordered[1] = pts[np.argmin(diff)]
    pts_ordered[3] = pts[np.argmax(diff)]
    # 返回排序后的坐标
    return pts_ordered


task = 'test'

# 源文件夹路径
src_folder = r'E:\code\yolov8_OCR_car\datasets\CCPD2019\{}\images'.format(task)
# 目标文件夹路径
dst_folder = r'E:\code\yolov8_OCR_car\datasets\CCPD2019\{}\targets'.format(task)

# 遍历源文件夹中的所有文件
for filename in os.listdir(src_folder):
    # 只处理图片文件
    if filename.endswith('.jpg'):
        # 读取图片
        image_path = os.path.join(src_folder, filename)
        # print(filename)
        image = cv2.imread(image_path)

        # 从文件名中提取四个顶点的坐标
        pts = np.float32([[int(x) for x in filename.split('-')[3].split('_')[0].split('&')],
                          [int(x) for x in filename.split('-')[3].split('_')[1].split('&')],
                          [int(x) for x in filename.split('-')[3].split('_')[2].split('&')],
                          [int(x) for x in filename.split('-')[3].split('_')[3].split('&')]])
        # print(pts)

        # 进行透视变换
        warped_image=  four_point_transform(image, pts)

        # 保存变换后的图片
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        dst_path = os.path.join(dst_folder, filename)
        cv2.imwrite(dst_path, warped_image)
