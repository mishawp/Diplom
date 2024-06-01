import os
import customtkinter as ctk
from pathlib import Path

PROJECT_ROOT = os.getenv("project_root")


class HyperparametersPage(ctk.CTkFrame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.label = ctk.CTkLabel(
            self, text="Гиперпараметры", fg_color="gray30", corner_radius=6
        )
        self.label.grid(row=0, column=0, padx=20, pady=20, sticky="ew", columnspan=4)

        self.button_back = ctk.CTkButton(
            self,
            text="Назад",
            command=self.prev_page,
        )
        self.button_back.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

    def init_hyperparameters(self, hyperparameters):
        """Динамическое создание полей для ввода параметров нейронной сети

        Args:
            hyperparameters (NNCreator): В каждом файл существует класс NNCreator,
            где NN - архитектура нейронной сети
        """
        self.labels_params = []
        self.entrys_params = []
        for i, (param, param_type) in enumerate(
            hyperparameters.params.items(), start=2
        ):
            label = ctk.CTkLabel(self, text=param)
            label.grid(row=i, column=0, padx=(20, 10), pady=10, sticky="ew")
            entry = ctk.CTkEntry(self)
            entry.grid(row=i, column=1, padx=(10, 20), pady=10, sticky="ew")

            self.labels_params.append(label)
            self.entrys_params.append(entry)

    def prev_page(self):
        for label, entry in zip(self.labels_params, self.entrys_params):
            label.destroy()
            entry.destroy()
        self.labels_params = []
        self.entrys_params = []

        self.controller.show_frame("NNSelectionPage")
