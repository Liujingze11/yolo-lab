from dataclasses import dataclass
import os
from paths import DATA_YAML, MODEL_FILE, RESULTS_DIR, LOG_DIR

@dataclass
class TrainConfig:
    
    data_yaml: str = DATA_YAML # data.yaml 配置文件路径
    model_file: str = MODEL_FILE # 初始加载的模型权重路径（如 yolov8n.pt、best.pt、last.pt）
    results_dir: str = RESULTS_DIR   # 所有实验结果保存的根目录
    experiment_name: str = "seg_dataset771_random_e150"    # 当前实验名称
    epochs: int = 150   # 训练轮数
    imgsz: int = 640    # 输入图片尺寸
    batch: int = 8  # 每批次训练图片数量
    device: int = 0 # 使用的设备，0 表示第1块 GPU
    log_dir: str = LOG_DIR   # 日志保存目录

    @property
    def save_dir(self) -> str:
        """
        本次实验结果的保存目录
        """
        return os.path.join(self.results_dir, self.experiment_name)

    @property
    def last_pt(self) -> str:
        """
        本次训练中断训练权重文件 last.pt 的路径
        """
        return os.path.join(self.save_dir, "weights", "last.pt")

    @property
    def best_pt(self) -> str:
        """
        本次实验最佳权重文件 best.pt 的路径
        """
        return os.path.join(self.save_dir, "weights", "best.pt")
    