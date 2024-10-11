from ultralytics import YOLO


if __name__ == '__main__':
    # 加载YOLOv8模型
    model = YOLO("models/v5/yolov5n.yaml")

    # 初始化OCR模型
    # reader = easyocr.Reader(['ch_sim', 'en'])

    # 读取图像
    # image = cv2.imread("car.png")

    model.train(data='dataset.yml', epochs=3, device='cuda', batch=4, workers=2)
    metrics = model.val()  # 在验证集上评估模型性能
    results = model("car.png")  # 对图像进行预测
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

    # 使用YOLOv8检测车牌
    # results = model(image)

    # 遍历检测到的车牌
    # for result in results[0].boxes.xyxy:
    #     x1, y1, x2, y2 = map(int, result)
    #
    #     # 提取车牌区域
    #     license_plate = image[y1:y2, x1:x2]
    #
    #     # 使用OCR识别车牌号
    #     ocr_result = reader.readtext(license_plate)
    #
    #     # 输出识别结果
    #     for detection in ocr_result:
    #         text = detection[1]
    #         print(f"Detected License Plate Text: {text}")
    #
    # # 显示检测结果
    # annotated_image = results[0].plot()
    # cv2.imshow("Detected License Plates", annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
