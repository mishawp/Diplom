import os
import customtkinter as ctk
from pathlib import Path

PROJECT_ROOT = os.getenv("project_root")


class TrainingPage(ctk.CTkFrame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.label = ctk.CTkLabel(
            self, text="Обучение модели", fg_color="gray30", corner_radius=6
        )
        self.label.grid(row=0, column=0, padx=20, pady=20, sticky="ew", columnspan=4)

        self.button_back = ctk.CTkButton(
            self,
            text="Назад",
            command=lambda: controller.show_frame("ParametersPage"),
        )
        self.button_back.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.label = ctk.CTkButton(self, text="Обучить модель", command=self.start)
        self.label.grid(row=2, column=0, padx=20, pady=20, sticky="ew", columnspan=4)

    def set_args(self, **kwargs):
        """Принимает с предыдущей страницы данные

        Args:
            model (nn.Module): модель нейронной сети
            dimension (int): размерность входных данных
            model_name (str): имя, которая будет дана модели
            dataset_name (str): название датасета
            params_nn (dict[str: value]): параметры нейронной сети в виде словаря
            params_other (dict[str: value]): параметры обучения в виде словаря
        """
        print(kwargs)
        pass

    def start(self):
        for frame_name, frame in self.controller.frames.items():
            if frame_name != "TrainingPage":
                frame.destroy()
                del frame
        self.button_back.destroy()
        del self.button_back
        pass
