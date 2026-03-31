import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

def get_device():
    if torch.cuda.is_available():
        print("Używane urządzenie: GPU (cuda)")
        return "cuda"
    print("Używane urządzenie: CPU")
    return "cpu"

def train_yolo():
    device = get_device()
    model = YOLO("yolo12n.pt")

    results = model.train(
        data=r"C:\Users\wikto\Desktop\Nowy folder\nocne\nocne\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        patience=20,
        plots=False
    )

    print("\n=== TRENING ZAKOŃCZONY ===\n")
    return model

def plot_training_metrics(results_csv="runs/detect/train/results.csv", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(results_csv):
        print("Brak pliku:", results_csv)
        return

    df = pd.read_csv(results_csv)

    train_loss = df["train/box_loss"]
    val_loss = df["val/box_loss"]

    recall = df["metrics/recall(B)"]
    precision = df["metrics/precision(B)"]
    map50 = df["metrics/mAP50(B)"]
    map5095 = df["metrics/mAP50-95(B)"]

    epochs = df["epoch"]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Strata: trening vs walidacja")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_train_val.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, recall, label="Recall (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall (walidacja)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/recall_val.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, map50, label="mAP50 (val)")
    plt.xlabel("Epoch")
    plt.ylabel("mAP50")
    plt.title("mAP50 (walidacja)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/mAP50_val.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, map5095, label="mAP50-95 (val)")
    plt.xlabel("Epoch")
    plt.ylabel("mAP50-95")
    plt.title("mAP50-95 (walidacja)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/mAP50-95_val.png")
    plt.show()
    plt.close()

    print("Wygenerowano wykresy metryk w folderze:", output_dir)


def plot_pr_curve(results_csv="runs/detect/train/results.csv", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(results_csv)

    recall = df["metrics/recall(B)"]
    precision = df["metrics/precision(B)"]

    plt.figure(figsize=(7, 7))
    plt.plot(recall, precision, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)
    plt.savefig(f"{output_dir}/precision_recall_curve.png")
    plt.show()
    plt.close()

    print("Zapisano Precision-Recall Curve:", f"{output_dir}/precision_recall_curve.png")


def load_trained_models(run_dir="runs/detect/train"):
    best_path = os.path.join(run_dir, "weights", "best.pt")
    last_path = os.path.join(run_dir, "weights", "last.pt")

    models = {}

    if os.path.exists(best_path):
        print(f"Ładowanie modelu BEST: {best_path}")
        models["best"] = YOLO(best_path)
    else:
        print("Brak pliku best.pt")

    if os.path.exists(last_path):
        print(f"Ładowanie modelu LAST: {last_path}")
        models["last"] = YOLO(last_path)
    else:
        print("Brak pliku last.pt")

    return models


def run_inference_on_test(models, test_dir="test_images"):
    if not os.path.exists(test_dir):
        print(f"Folder testowy '{test_dir}' nie istnieje")
        return

    os.makedirs(test_dir, exist_ok=True)

    for name, model in models.items():
        print(f"\n=== Wykrywanie ({name}.pt) na folderze: {test_dir} ===")

        model.predict(
            source=test_dir,
            imgsz=640,
            save=True,
            save_txt=False,
            conf=0.25,
        )

        print(f"Wyniki zapisane w runs/detect/predict* dla modelu: {name}")

if __name__ == "__main__":
    model = train_yolo()

    results_csv = "runs/detect/train/results.csv"

    plot_training_metrics(results_csv)
    plot_pr_curve(results_csv)

    models = load_trained_models()

    run_inference_on_test(models, test_dir="C:/Users/wikto/Desktop/Nowy folder/DaneGotowe/test/images")

    print("\nWszystkie wykresy zapisane w folderze: plots/")
    print("Modele dostępne pod kluczami: models['best'], models['last']")
    print("Wykrycia zapisane w runs/detect/predict*/\n")
