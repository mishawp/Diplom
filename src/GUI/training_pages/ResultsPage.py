import os
import customtkinter as ctk
from pathlib import Path

PROJECT_ROOT = os.getenv("project_root")


class ResultsPage(ctk.CTkFrame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.training_thread = None

        self.button_start = ctk.CTkLabel(
            self, text="Обучение модели", fg_color="gray30", corner_radius=6
        )
        self.button_start.grid(
            row=0, column=0, padx=20, pady=20, sticky="ew", columnspan=4
        )

        self.button_back = ctk.CTkButton(
            self,
            text="Назад",
            command=self.prev_page,
        )
        self.button_back.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

    def set_args(self, **kwargs):
        """Принимает с предыдущей страницы данные

        kwargs:
        model_name (str): имя, под которой сохранена модель
        """
        self.model_name = kwargs["model_name"]
        pass
