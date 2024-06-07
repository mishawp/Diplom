import os
import customtkinter as ctk
from src.utils import start_training
import threading
import sys

PROJECT_ROOT = os.getenv("project_root")


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(ctk.END, string)

    def flush(self):
        pass  # Метод flush нужен для совместимости, но мы его не используем


class TrainingPage(ctk.CTkFrame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.training_thread = None

        self.label = ctk.CTkLabel(
            self, text="Обучение модели", fg_color="lightblue", corner_radius=6
        )
        self.label.grid(row=0, column=0, padx=20, pady=20, sticky="ew", columnspan=4)

        self.button_back = ctk.CTkButton(
            self,
            text="Назад",
            command=self.prev_page,
        )
        self.button_back.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.button_start = ctk.CTkButton(
            self, text="Обучить модель", command=self.start
        )
        self.button_start.grid(
            row=2, column=0, padx=20, pady=20, sticky="ew", columnspan=4
        )

        self.label_status = ctk.CTkTextbox(self, width=800, height=400)
        self.label_status.grid(
            row=3, column=0, padx=20, pady=20, sticky="ew", columnspan=4
        )
        # self.label_status.configure(state="disabled")

    def set_args(self, **kwargs):
        """Принимает с предыдущей страницы данные

        kwargs:
            model_class (type[Module]): класс нейронной сети
            dimension (int): 1 или 2 (2D или 1D нейросеть будет обучаться)
            model_name (str): имя, которая под которой будет сохранена модель
            dataset_name (str): название датасета, на которой будет обучаться модель
            parameters (dict):
                (epochs, batch_size, learning_rate, l2_decay, optimizer: str = ["adam", "sgd"], device: str = ["cuda", "cpu", "mps"])
            model_parameters (dict): Параметры для конструктора модели. Default {}
        """
        self.kwargs = kwargs
        self.keep_stdout = sys.stdout
        self.keep_stderr = sys.stderr
        sys.stdout = RedirectText(self.label_status)
        sys.stderr = RedirectText(self.label_status)

    def start(self):
        for frame_name, frame in self.controller.frames.items():
            if frame_name != "TrainingPage" and frame_name != "ResultsPage":
                frame.destroy()
                self.controller.frames[frame_name] = None

        self.button_back.destroy()
        self.button_start.destroy()
        if self.training_thread is None or not self.training_thread.is_alive():
            print("Запуск обучения...\n")
            self.training_thread = threading.Thread(
                target=start_training, kwargs=self.kwargs
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            self.check_training_thread()

    def check_training_thread(self):
        if self.training_thread.is_alive():
            # Если поток все еще активен, проверяем его снова через 100 мс
            self.after(100, self.check_training_thread)
        else:
            # Если поток завершился, обновляем статус
            print("Обучение завершено!\n")
            sys.stdout = self.keep_stdout
            sys.stderr = self.keep_stderr
            self.button_next = ctk.CTkButton(
                self,
                text="Результаты",
                command=self.next_page,
            )
            self.button_next.grid(row=4, column=3, padx=20, pady=20, sticky="ew")

    def prev_page(self):
        sys.stdout = self.keep_stdout
        sys.stderr = self.keep_stderr
        self.controller.show_frame("ParametersPage")

    def next_page(self):
        self.controller.frames["ResultsPage"].set_args(
            model_name=self.kwargs["model_name"]
        )
        self.controller.show_frame("ResultsPage")
