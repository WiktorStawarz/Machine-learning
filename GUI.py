import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class YOLOComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv12 – porównanie modeli")
        self.root.geometry("1200x760")

        self.best_model = None
        self.last_model = None
        self.test_dir = ""
        self.single_image = None
        self.images = []
        self.img_index = 0
        self.results_best = None
        self.results_last = None
        self.data_yaml = ""

        self.create_widgets()

    def create_widgets(self):
        top = tk.Frame(self.root)
        top.pack(pady=10)

        tk.Button(top, text="Załaduj pierwszy model", command=self.load_best).grid(row=0, column=0, padx=5)
        tk.Button(top, text="Załaduj drugi model", command=self.load_last).grid(row=0, column=1, padx=5)
        tk.Button(top, text="Zbiór testowy", command=self.select_test).grid(row=0, column=2, padx=5)
        tk.Button(top, text="Pojedynczy obraz", command=self.select_single_image).grid(row=0, column=3, padx=5)
        tk.Button(top, text="Wybierz DATA.YAML", command=self.select_data_yaml).grid(row=0, column=4, padx=5)

        self.yaml_label = tk.Label(top, text="DATA.YAML: nie wybrano", fg="gray")
        self.yaml_label.grid(row=1, column=0, columnspan=5)

        tk.Label(top, text=f"Urządzenie: {DEVICE.upper()}").grid(row=2, column=0, columnspan=5)

        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=10)

        stats_frame = tk.Frame(self.root)
        stats_frame.pack(pady=10)

        self.stats_best = tk.Label(stats_frame, text="pierwszy model\n-", justify="left", width=45, anchor="nw")
        self.stats_best.grid(row=0, column=0, padx=10)

        self.stats_last = tk.Label(stats_frame, text="drugi model\n-", justify="left", width=45, anchor="nw")
        self.stats_last.grid(row=0, column=1, padx=10)

        bottom = tk.Frame(self.root)
        bottom.pack(pady=10)

        tk.Button(bottom, text="Testuj pierwszy model", command=lambda: self.run_test("pierwszy model")).grid(row=0, column=0, padx=5)
        tk.Button(bottom, text="Testuj drugi model", command=lambda: self.run_test("drugi model")).grid(row=0, column=1, padx=5)
        tk.Button(bottom, text="Poprzedni", command=self.prev_img).grid(row=0, column=2, padx=5)
        tk.Button(bottom, text="Następny", command=self.next_img).grid(row=0, column=3, padx=5)

    def select_data_yaml(self):
        path = filedialog.askopenfilename(filetypes=[("YAML file", "*.yaml")])
        if path:
            self.data_yaml = path
            self.yaml_label.config(text=f"DATA.YAML: {os.path.basename(path)}", fg="black")

    def load_best(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO model", "*.pt")])
        if path:
            self.best_model = YOLO(path)
            messagebox.showinfo("OK", "Załadowano pierwszy model")

    def load_last(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO model", "*.pt")])
        if path:
            self.last_model = YOLO(path)
            messagebox.showinfo("OK", "Załadowano drugi model")

    def select_test(self):
        self.test_dir = filedialog.askdirectory()
        if self.test_dir:
            self.images = [
                os.path.join(self.test_dir, f)
                for f in os.listdir(self.test_dir)
                if f.lower().endswith((".jpg", ".png"))
            ]
            self.img_index = 0
            self.single_image = None
            self.show_image(self.images[0])

    def select_single_image(self):
        self.single_image = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png")]
        )
        if self.single_image:
            self.images = [self.single_image]
            self.img_index = 0
            self.test_dir = ""
            self.show_image(self.single_image)

    def evaluate_model(self, model):
        if not self.data_yaml:
            messagebox.showwarning("Uwaga", "Nie wybrano pliku DATA.YAML")
            return None

        metrics = model.val(
            data=self.data_yaml,
            split="test",
            device=DEVICE,
            imgsz=640,
            verbose=False
        )
        return metrics

    def run_test(self, mode):
        model = self.best_model if mode == "pierwszy model" else self.last_model
        if model is None:
            messagebox.showwarning("Uwaga", f"Nie załadowano {mode}.pt")
            return

        source = self.single_image if self.single_image else self.test_dir
        if not source:
            messagebox.showwarning("Uwaga", "Brak danych do testu")
            return

        start = time.time()
        results = model.predict(
            source=source,
            imgsz=640,
            device=DEVICE,
            conf=0.25,
            save=True,
            name=f"predict_{mode}"
        )
        elapsed = time.time() - start
        fps = len(results) / elapsed

        text = f"{mode.upper()}\nCzas: {elapsed:.2f}s | FPS: {fps:.2f}"

        if not self.single_image:
            m = self.evaluate_model(model)
            if m:
                text += (
                    f"\nmAP50: {m.box.map50:.3f}"
                    f"\nmAP50-95: {m.box.map:.3f}"
                    f"\nPrecision: {m.box.mp:.3f}"
                    f"\nRecall: {m.box.mr:.3f}"
                )

        if mode == "pierwszy model":
            self.stats_best.config(text=text)
            self.results_best = results
        else:
            self.stats_last.config(text=text)
            self.results_last = results

        self.show_image(results[self.img_index].plot())

    def show_image(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (720, 480))

        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        self.img_label.imgtk = imgtk
        self.img_label.configure(image=imgtk)

    def next_img(self):
        if len(self.images) > 1:
            self.img_index = (self.img_index + 1) % len(self.images)
            self.update_view()

    def prev_img(self):
        if len(self.images) > 1:
            self.img_index = (self.img_index - 1) % len(self.images)
            self.update_view()

    def update_view(self):
        if self.results_best:
            self.show_image(self.results_best[self.img_index].plot())
        elif self.results_last:
            self.show_image(self.results_last[self.img_index].plot())

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOComparisonGUI(root)
    root.mainloop()
