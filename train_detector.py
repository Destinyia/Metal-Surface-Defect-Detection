from ultralytics_ import YOLO

if __name__ == '__main__':
    # 加载模型
    # model = YOLO("./models/yolov8m-p2.yaml")  # 从头开始构建新模型
    # model = YOLO("weights/yolov8m.pt")  # 加载预训练模型（建议用于训练）
    # model = YOLO("weights/yolov5m6u.pt")  # 加载预训练模型（建议用于训练）
    model = YOLO('yolov8m.pt')
    model.train(data='./datasets/mix_8cls.yaml', 
            warmup_epochs=1, imgsz=640, batch=16, epochs=80, device='cuda', workers=0,
            hsv_h=0.06, hsv_s=0.4, hsv_v=0.5,
            flipud=0.2, label_smoothing=0.1, mosaic=0.9, scale=0.4, degrees=15, mixup=0.1, close_mosaic=10, 
            name='YOLOv8m_mix_1')  # 训练模型