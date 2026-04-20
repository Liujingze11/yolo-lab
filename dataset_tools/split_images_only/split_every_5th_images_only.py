import os
import shutil

train_dir = "请输入你的训练集地址"   # 训练集地址 
val_dir = "请输入你的验证集目标地址"   # 测试集目标地址

os.makedirs(val_dir, exist_ok=True)

images = [f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")]
images.sort(key=lambda x: int(os.path.splitext(x)[0]))

for img in images:
    num = int(os.path.splitext(img)[0])
    if num % 5 == 0:
        src = os.path.join(train_dir, img)
        dst = os.path.join(val_dir, img)
        shutil.move(src, dst)

print("已将 5 的倍数编号图片移动到 val")