import os
import customtkinter as ctk
import importlib
from pathlib import Path

PROJECT_ROOT = os.getenv("project_root")


class NNSelectionPage(ctk.CTkFrame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.label = ctk.CTkLabel(
            self, text="Нейронные сети", fg_color="gray30", corner_radius=6
        )
        self.label.grid(row=0, column=0, padx=20, pady=20, sticky="ew", columnspan=4)

        self.button_back = ctk.CTkButton(
            self,
            text="Назад",
            command=lambda: controller.show_frame("DataProcessingPage"),
        )
        self.button_back.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.var_nn = ctk.StringVar()

        self.frame_nn1d = NNFrame(self, 1)
        self.frame_nn1d.grid(
            row=2, column=0, padx=20, pady=10, sticky="ew", columnspan=2
        )

        self.frame_nn2d = NNFrame(self, 2)
        self.frame_nn2d.grid(
            row=2, column=2, padx=20, pady=10, sticky="ew", columnspan=2
        )

        self.button_next_page = ctk.CTkButton(
            self,
            text="Настройка гиперпараметров",
            command=self.next_page,
        )
        self.button_next_page.grid(row=3, column=3, padx=20, pady=(10, 20), sticky="ew")

    def next_page(self):
        var_nn_text = self.var_nn.get()
        if var_nn_text == "":
            return
        path_nn = var_nn_text.split("/")
        dimension = int(path_nn[1][2])  # nn1d или nn2d -> 1 или 2
        path_module = ".".join(path_nn[:-1])  # извлекаем src.nn1d или src.nn2d
        nn_name = path_nn[-1][:-3]  # удаляем суффикс .py
        creator = getattr(importlib.import_module(path_module), f"{nn_name}Creator")()

        # передаем выбранную NN (точнее, NN Creator ) в HyperparametersPage, т.е. в следующую страницу
        page = self.controller.frames["HyperparametersPage"]
        page.init_hyperparameters(creator, dimension)
        self.controller.show_frame(page.__class__.__name__)


class NNFrame(ctk.CTkFrame):
    def __init__(self, parent, dimension):
        super().__init__(parent)
        self.grid_columnconfigure(0, weight=1)
        self.label = ctk.CTkLabel(
            self,
            text=f"{dimension}D Нейронные Сети",
            fg_color="gray30",
            corner_radius=6,
        )
        self.label.grid(row=0, column=0, padx=10, pady=20, sticky="ew")

        path_nns = Path(PROJECT_ROOT, "src", f"nn{dimension}d")
        self.radiobuttons = []
        for i, nn in enumerate(path_nns.iterdir(), start=1):
            if not nn.is_file() or nn.name == "__init__.py":
                continue

            radiobutton = ctk.CTkRadioButton(
                self,
                text=nn.stem,
                variable=parent.var_nn,
                value="/".join(nn.parts[-3:]),  # src/nn1(2)d/nn_creator.py
            )
            radiobutton.grid(row=i, column=0, padx=10, pady=10, sticky="ew")
            self.radiobuttons.append(radiobutton)
