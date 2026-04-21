from ultralytics import YOLO
from paths import PREDICT_DIR, BEST_SEG_MODEL, TEST_IMAGES_DIR

# 加载训练好的模型
model = YOLO(BEST_SEG_MODEL)

# 对 test 文件夹中的图片进行推理
results = model.predict(
    source=TEST_IMAGES_DIR,  # 测试图片路径
    task="segment",   # 分割任务
    save=True,        # 保存结果图
    project=PREDICT_DIR,  # 保存总目录
    name="test_infer_771",  # 本次结果文件夹名
    imgsz=640,        # 推理尺寸
    conf=0.25         # 置信度阈值
)

print("推理完成，结果已保存。")