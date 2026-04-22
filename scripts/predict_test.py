from ultralytics import YOLO
from paths import PREDICT_DIR, BEST_SEG_MODEL, TEST_IMAGES_DIR
from pathlib import Path
import cv2

# 加载模型
model = YOLO(BEST_SEG_MODEL)

# 推理
results = model.predict(
    source=TEST_IMAGES_DIR,
    task="segment",          # 建议用 segment；也可以直接删掉这一行
    imgsz=640,
    conf=0.406,
    retina_masks=True,       # 更清晰的 mask
    save=False               # 不用默认保存，后面自己画
)

# 保存目录
save_dir = Path(PREDICT_DIR) / "seg_dataset_all_pro_random__aug_e150_b16_mask_overlay"
save_dir.mkdir(parents=True, exist_ok=True)

# 逐张绘制并保存
for i, r in enumerate(results):
    plotted = r.plot(
        font_size=10,   # 字小一点，可改成 8 / 9 / 10
        line_width=2,   # 框线细一点
        labels=True,
        boxes=True,
        masks=True,     # 显示 mask 覆盖
        conf=True
    )

    out_path = save_dir / f"{Path(r.path).stem}_overlay.jpg"
    cv2.imwrite(str(out_path), plotted)

print("推理完成，带 mask 覆盖的结果已保存。")