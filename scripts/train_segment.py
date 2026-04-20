import os
import yaml
from ultralytics import YOLO
from config import TrainConfig
from train_logger import append_train_log, append_full_val_log

# =========================
# 训练配置对象
# 统训练时需要使用的路径、模型和超参数
# =========================
CONFIG = TrainConfig()

# =========================
# 工具函数
# =========================
def ask_confirm_train(mode, pt_path, config):
    print("\n------------------------------")
    print(f"即将执行：{mode}")
    print(f"当前使用的 PT 文件：{pt_path}")
    print(f"数据配置文件      ：{config.data_yaml}")
    print(f"结果保存目录      ：{config.results_dir}")
    print(f"实验名称          ：{config.experiment_name}")
    print("------------------------------")

    confirm = input("请确认是否继续？输入 y 继续，其他任意键取消：").strip().lower()
    if confirm != "y":
        print("\n已取消本次训练。")
        return False
    return True


def list_experiments(results_dir):
    if not os.path.exists(results_dir):
        print(f"\n结果目录不存在：{results_dir}")
        return []

    folders = []
    for name in os.listdir(results_dir):
        full_path = os.path.join(results_dir, name)
        if os.path.isdir(full_path):
            folders.append(name)

    folders.sort()
    return folders



# =========================
# 数据集与验证集处理
# =========================
def get_class_names_from_data_yaml(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})

    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    else:
        return {}
    



def get_val_labels_dir(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root_path = data.get("path", "")
    val_path = data.get("val", "")

    if not val_path:
        return None

    # 如果 val 是相对路径，且 yaml 中配置了 path，则先拼完整路径
    if root_path and not os.path.isabs(val_path):
        val_path = os.path.join(root_path, val_path)

    val_path = os.path.normpath(val_path)

    # 常见情况：.../images/val -> .../labels/val
    parts = val_path.split(os.sep)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return os.path.normpath(os.sep.join(parts))

    # 兜底方案
    parent_dir = os.path.dirname(os.path.dirname(val_path))
    val_name = os.path.basename(val_path)
    return os.path.join(parent_dir, "labels", val_name)




def count_val_label_stats(config):
    val_labels_dir = get_val_labels_dir(config.data_yaml)
    if not val_labels_dir or not os.path.exists(val_labels_dir):
        print(f"\n未找到 val 标签目录：{val_labels_dir}")
        return {}, {}

    class_names = get_class_names_from_data_yaml(config.data_yaml)

    class_image_counts = {}
    class_instance_counts = {}

    for class_id, class_name in class_names.items():
        class_image_counts[class_name] = 0
        class_instance_counts[class_name] = 0

    for file_name in os.listdir(val_labels_dir):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(val_labels_dir, file_name)
        appeared_in_this_image = set()

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 1:
                continue

            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue

            class_name = class_names.get(class_id, f"class_{class_id}")
            class_instance_counts[class_name] = class_instance_counts.get(class_name, 0) + 1
            appeared_in_this_image.add(class_name)

        for class_name in appeared_in_this_image:
            class_image_counts[class_name] = class_image_counts.get(class_name, 0) + 1

    return class_image_counts, class_instance_counts


def get_val_metrics(best_pt_path, config):
    model = YOLO(best_pt_path)

    metrics = model.val(
        data=config.data_yaml,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device
    )

    return metrics


def log_validation_result(config, mode, notes=""):
    if not os.path.exists(config.best_pt):
        print(f"\n未找到 best.pt，无法记录验证结果：{config.best_pt}")
        return

    try:
        metrics = get_val_metrics(config.best_pt, config)
        class_image_counts, class_instance_counts = count_val_label_stats(config)

        append_full_val_log(
            config=config,
            mode=mode,
            metrics=metrics,
            class_image_counts=class_image_counts,
            class_instance_counts=class_instance_counts,
            notes=notes
        )
        print("\n验证结果已记录到日志。")

    except Exception as e:
        print(f"\n记录验证结果失败：{e}")


# =========================
# 训练流程
# =========================
def start_new_training(config):
    print("\n开始新训练...")

    if not ask_confirm_train("开始新训练", config.model_file, config):
        return
    
    append_train_log(config, mode="new_train", status="started", notes="开始新训练")

    try:
        model = YOLO(config.model_file)
        model.train(
            data=config.data_yaml,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            project=config.results_dir,
            name=config.experiment_name
        )

        append_train_log(config, mode="new_train", status="finished", notes="训练完成")
        log_validation_result(config, mode="new_train", notes="训练完成后的验证结果")

    except Exception as e:
        append_train_log(config, mode="new_train", status="failed", notes=str(e))
        print(f"\n训练失败：{e}")


def resume_training(config):
    if not os.path.exists(config.last_pt):
        print(f"\n没有找到上次中断训练的权重文件：{config.last_pt}")

        choice = input("是否改为开启新的训练？输入 y 继续，其他任意键取消：").strip().lower()
        if choice == "y":
            start_new_training(config)
        else:
            print("已取消操作。")
        return

    if not ask_confirm_train("继续上次训练", config.last_pt, config):
        return

    append_train_log(config, mode="resume_train", status="started", notes="继续上次训练")

    try:
        model = YOLO(config.last_pt)
        model.train(resume=True)

        append_train_log(config, mode="resume_train", status="finished", notes="继续训练完成")
        log_validation_result(config, mode="resume_train", notes="继续训练后的验证结果")

    except Exception as e:
        append_train_log(config, mode="resume_train", status="failed", notes=str(e))
        print(f"\n继续训练失败：{e}")


def train_from_previous_best(config):
    folders = list_experiments(config.results_dir)

    if not folders:
        print("\n没有找到任何历史实验文件夹。")
        return

    print("\n检测到以下历史实验：")
    for i, folder in enumerate(folders, 1):
        print(f"{i} - {folder}")

    choice = input("请选择要作为基础模型的实验编号：").strip()

    if not choice.isdigit():
        print("输入无效，已取消。")
        return

    idx = int(choice) - 1
    if idx < 0 or idx >= len(folders):
        print("编号超出范围，已取消。")
        return

    selected_exp = folders[idx]
    selected_best_pt = os.path.join(config.results_dir, selected_exp, "weights", "best.pt")

    if not os.path.exists(selected_best_pt):
        print(f"\n该实验下没有找到 best.pt：{selected_best_pt}")
        return

    print(f"\n你选择的实验是：{selected_exp}")

    if not ask_confirm_train("基于历史 best.pt 开启新训练", selected_best_pt, config):
        return

    append_train_log(
        config,
        mode="train_from_best",
        status="started",
        notes=f"基于历史实验 {selected_exp} 的 best.pt 开始训练"
    )

    try:
        model = YOLO(selected_best_pt)
        model.train(
            data=config.data_yaml,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            project=config.results_dir,
            name=config.experiment_name
        )

        append_train_log(
            config,
            mode="train_from_best",
            status="finished",
            notes=f"基于历史实验 {selected_exp} 的训练完成"
        )

        log_validation_result(
            config,
            mode="train_from_best",
            notes=f"基于历史实验 {selected_exp} 的验证结果"
        )

    except Exception as e:
        append_train_log(
            config,
            mode="train_from_best",
            status="failed",
            notes=str(e)
        )
        print(f"\n训练失败：{e}")


# =========================
# 主程序入口
# =========================
def main():
    print("请选择训练方式：")
    print("1 - 开启一个新的训练")
    print("2 - 继续上次中断的训练")
    print("3 - 基于历史实验的 best.pt 再次训练")
    choice = input("请输入 1、2 或 3，直接回车退出\n").strip()

    if choice == "1":
        start_new_training(CONFIG)
    elif choice == "2":
        resume_training(CONFIG)
    elif choice == "3":
        train_from_previous_best(CONFIG)
    else:
        print("输入无效，程序已退出。")


if __name__ == "__main__":
    main()