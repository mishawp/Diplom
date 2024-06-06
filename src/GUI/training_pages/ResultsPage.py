import os
import customtkinter as ctk
from pathlib import Path

PROJECT_ROOT = os.getenv("project_root")


class ResultsPage(ctk.CTkFrame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.button_start = ctk.CTkLabel(
            self, text="Результаты обучения", fg_color="gray30", corner_radius=6
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

        self.frame_files = FilesFrame(self)
        self.frame_files.grid(
            row=2, column=0, padx=20, pady=20, sticky="ew", columnspan=4
        )

        self.button_finish = ctk.CTkButton(
            self,
            text="Закрыть",
            command=self.controller.destroy,
        )
        self.button_finish.grid(row=3, column=1, padx=20, pady=10, sticky="ew")

    def set_args(self, **kwargs):
        """Принимает с предыдущей страницы данные

        kwargs:
        model_name (str): имя, под которой сохранена модель
        """
        self.model_name = kwargs["model_name"]
        self.frame_files.init(self.model_name)

    def prev_page(self):
        for widget in self.frame_files.winfo_children():
            widget.destroy()
        self.controller.show_frame("TrainingPage")


class FilesFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

    def init(self, model_name: str):
        self.model_name = model_name
        path_reports = Path(PROJECT_ROOT, "reports", self.model_name)
        self.labels_files = []
        self.buttons_open = []
        for i, path in enumerate(path_reports.iterdir()):
            label = ctk.CTkLabel(self, text=path.stem)
            label.grid(row=i, column=0, padx=20, pady=10, sticky="ew")
            button = ctk.CTkButton(
                self, text="Открыть", command=lambda path=path: os.startfile(path)
            )
            button.grid(row=i, column=1, padx=20, pady=10, sticky="ew")
            self.labels_files.append(label)
            self.buttons_open.append(button)
