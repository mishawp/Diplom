import customtkinter as ctk
import tkinter as tk
import os
import threading
from pathlib import Path
from src.data import fetch_ecg_data, make_dataset

PROJECT_ROOT = os.getenv("project_root")

# Для доступа к важным виджетам из объектов. Плохое решение (
global_widgets = {"frequency": None, "leads": None}


class DataProcessingPage(ctk.CTkFrame):
    def __init__(self, controller):
        # 8 row 4 col
        super().__init__(controller)
        self.controller = controller

        self.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8), weight=1)

        self.label = ctk.CTkLabel(
            self, text="Предобработка данных", fg_color="gray30", corner_radius=6
        )
        self.label.grid(row=0, column=0, padx=20, pady=20, sticky="ew", columnspan=4)

        # self.button_back = ctk.CTkButton(
        #     self, text="Назад", command=lambda: controller.show_frame("MainMenu")
        # )
        # self.button_back.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.window_download = None
        self.button_download = ctk.CTkButton(
            self, text="Скачать PTB-XL", command=self.download_data
        )
        self.button_download.grid(
            row=2, column=0, padx=20, pady=10, sticky="ew", columnspan=4
        )

        self.frame_frequency = FrequencyFrame(self)
        self.frame_frequency.grid(row=3, column=3, padx=20, pady=10, sticky="ew")

        self.frame_leads = LeadsFrame(self)
        self.frame_leads.grid(
            row=4, column=0, padx=20, pady=10, sticky="ew", columnspan=4
        )

        self.frame_processing_1d = ProcessingFrame(self, "1D обработка", 1)
        self.frame_processing_1d.grid(
            row=5, column=0, padx=20, pady=10, sticky="ew", columnspan=2
        )

        self.frame_processing_2d = ProcessingFrame(self, "2D обработка", 2)
        self.frame_processing_2d.grid(
            row=5, column=2, padx=20, pady=10, sticky="ew", columnspan=2
        )

        self.button_next_page = ctk.CTkButton(
            self,
            text="Выбрать нейронную сеть",
            command=lambda: controller.show_frame("NNSelectionPage"),
        )
        self.button_next_page.grid(row=6, column=3, padx=20, pady=(10, 20), sticky="ew")

    def download_data(self):
        if self.window_download is None or not self.window_download.winfo_exists():
            self.window_download = DownloadWindow(self)
        else:
            self.window_download.focus()


class DownloadWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Загрузить данные")
        self.geometry("600x300")
        self.grid_columnconfigure(0, weight=1)

        # Создаем фрейм для размещения элементов
        self.frame = ctk.CTkFrame(self, width=300, height=300)
        self.frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Поле ввода с текстом по умолчанию
        self.label_url = ctk.CTkLabel(self.frame, text="URL")
        self.label_url.grid(row=0, column=0, padx=10, pady=0, sticky="w")

        self.entry_url = ctk.CTkEntry(self.frame)
        self.entry_url.insert(
            0,
            r"https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
        )
        self.entry_url.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.label_name = ctk.CTkLabel(self.frame, text="Название датасета")
        self.label_name.grid(row=2, column=0, padx=10, pady=0, sticky="w")

        self.entry_name = ctk.CTkEntry(self.frame)
        self.entry_name.insert(0, "ptbxl")
        self.entry_name.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        # Кнопка
        self.button = ctk.CTkButton(
            self.frame, text="Скачать", command=self.download_data
        )
        self.button.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        self.label_status = ctk.CTkLabel(self.frame, text="")
        self.label_status.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

        self.label_download_progress = ctk.CTkLabel(self.frame, text="")
        self.label_download_progress.grid(
            row=6, column=0, padx=10, pady=10, sticky="ew"
        )

    def download_data(self):
        # Здесь можно добавить логику для загрузки данных
        url = self.entry_url.get()

        path_data_raw = Path(PROJECT_ROOT, "data", "raw")
        path_data_raw.mkdir(parents=True, exist_ok=True)
        path_data_raw = Path(path_data_raw, self.entry_name.get())

        if path_data_raw.exists():
            self.label_status.configure(
                text=f"Датасет уже существует {path_data_raw}", text_color="red"
            )
        else:
            self.label_status.configure(text="")
            path_data_raw.mkdir(parents=True, exist_ok=True)
            thread = threading.Thread(
                target=fetch_ecg_data, args=(url, path_data_raw, self.progress_bar)
            )
            thread.start()

    def progress_bar(self, current, total, width=100):
        self.label_download_progress.configure(
            text=f"Загрузка: {(current/total*100):.2f}%"
        )


class FrequencyFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.grid_columnconfigure((0, 1), weight=1)
        self.title = ctk.CTkLabel(
            self, text="Частота дискретизации", fg_color="gray30", corner_radius=6
        )
        self.title.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.variable = ctk.StringVar(value="100")

        self.radiobutton100fs = ctk.CTkRadioButton(
            self, text="100 Гц", variable=self.variable, value="100"
        )
        self.radiobutton100fs.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.radiobutton500fs = ctk.CTkRadioButton(
            self, text="500 Гц", variable=self.variable, value="500"
        )
        self.radiobutton500fs.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    def get(self):
        return self.variable.get()

    def set(self, value):
        self.variable.set(value)


class LeadsFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.grid_columnconfigure(tuple(range(12)), weight=1)
        self.title = ctk.CTkLabel(
            self,
            text="Отведения, используемые при обучении",
            fg_color="gray30",
            corner_radius=6,
        )
        self.title.grid(row=0, column=0, columnspan=12, padx=10, pady=10, sticky="ew")
        self.leads = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        self.checkboxes = []

        for i, lead in enumerate(self.leads):
            checkbox = ctk.CTkCheckBox(self, text=lead)
            checkbox.grid(row=1, column=i, padx=(10, 0), pady=5, sticky="ew")
            self.checkboxes.append(checkbox)

    def get(self):
        checked_checkboxes = []
        for checkbox in self.checkboxes:
            if checkbox.get() == 1:
                checked_checkboxes.append(checkbox.cget("text"))
        return checked_checkboxes


class ProcessingFrame(ctk.CTkFrame):
    def __init__(self, parent, title: str, dimension: int):
        super().__init__(parent)
        self.grid_columnconfigure(0, weight=1)
        self.title = ctk.CTkLabel(self, text=title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.dimension = dimension
        self.mainframe_frame_frequency = parent.frame_frequency
        self.mainframe_frame_leads = parent.frame_leads

        self.path_processing1d = Path(PROJECT_ROOT, "src", "data", "processing1d.py")
        self.path_processing2d = Path(PROJECT_ROOT, "src", "data", "processing2d.py")
        # self.path_processed1d_data = Path(PROJECT_ROOT, "data", "processed", "1D")
        # self.path_processed2d_data = Path(PROJECT_ROOT, "data", "processed", "2D")

        self.processing_file_button = ctk.CTkButton(
            self, text="Файл для обработки данных", command=self.processing_file
        )
        self.processing_file_button.grid(
            row=1, column=0, padx=10, pady=10, sticky="ew", columnspan=4
        )

        self.process_window = None

        self.process_button = ctk.CTkButton(
            self, text="Обработать данные", command=self.process
        )
        self.process_button.grid(
            row=2, column=0, padx=10, pady=10, sticky="ew", columnspan=4
        )

    def processing_file(self):
        """Открывает файл с функциями для обработки данных"""
        path = self.__getattribute__(f"path_processing{self.dimension}d")
        if not path.exists():
            path.touch()
        os.startfile(path)

    def process(self):
        if self.process_window is None or not self.process_window.winfo_exists():
            self.process_window = ProcessWindow(self)
        else:
            self.process_window.focus()


class ProcessWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Обработка данных")
        self.geometry("600x400")
        self.grid_columnconfigure(0, weight=1)

        self.dimension = parent.dimension
        self.mainframe_frame_frequency = parent.mainframe_frame_frequency
        self.mainframe_frame_leads = parent.mainframe_frame_leads

        # Создаем фрейм для размещения элементов
        self.frame = ctk.CTkFrame(self, width=300, height=300)
        self.frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.frame_dataset = DatasetFrame(self.frame)
        self.frame_dataset.grid(row=0, column=0, padx=10, pady=20, sticky="we")

        self.label_name = ctk.CTkLabel(self.frame, text="Сохранить предобработку как:")
        self.label_name.grid(row=1, column=0, padx=10, pady=0, sticky="w")

        self.entry_name = ctk.CTkEntry(self.frame)
        self.entry_name.insert(0, "my_data")
        self.entry_name.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        # Кнопка
        self.button = ctk.CTkButton(self.frame, text="Обработать", command=self.process)
        self.button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.label_status = ctk.CTkLabel(self.frame, text="")
        self.label_status.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        self.label_process_progress = ctk.CTkLabel(self.frame, text="")
        self.label_process_progress.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

    def process(self):
        if not self.mainframe_frame_leads.get():
            self.label_status.configure(text="Отведения не выбраны", text_color="red")
            return
        if not self.frame_dataset.get():
            self.label_status.configure(text="Датасет не выбран", text_color="red")
            return
        if not self.frame_dataset.get():
            self.label_status.configure(text="Недопустимое имя", text_color="red")
            return
        path_data_processed = Path(
            PROJECT_ROOT,
            "data",
            "processed",
            f"{self.dimension}D",
            self.entry_name.get(),
        )

        if path_data_processed.exists():
            self.label_status.configure(
                text=f"Датасет уже существует: {path_data_processed}", text_color="red"
            )
        else:
            self.label_status.configure(text="")
            path_data_processed.mkdir(parents=True, exist_ok=True)
            thread = threading.Thread(
                target=make_dataset,
                args=(
                    self.frame_dataset.get(),
                    self.entry_name.get(),
                    self.dimension,
                    self.mainframe_frame_frequency.get(),
                    self.mainframe_frame_leads.get(),
                    self.progress_bar,
                ),
            )
            thread.start()

    def progress_bar(self, current, total):
        self.label_process_progress.configure(
            text=f"Обработка: {(current/total*100):.2f}%"
        )


class DatasetFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.path_data_raw = Path(PROJECT_ROOT, "data", "raw")
        self.raw_datasets = tuple(self.path_data_raw.iterdir())

        self.grid_columnconfigure(tuple(range(len(self.raw_datasets))), weight=1)

        self.title = ctk.CTkLabel(
            self, text="Датасет", fg_color="gray30", corner_radius=6
        )
        self.title.grid(
            row=0,
            column=0,
            columnspan=len(self.raw_datasets),
            padx=10,
            pady=10,
            sticky="ew",
        )
        self.variable = ctk.StringVar(value="")

        self.radiobuttons = []
        for i, dataset in enumerate(self.raw_datasets):
            radiobutton = ctk.CTkRadioButton(
                self, text=dataset.name, variable=self.variable, value=dataset.name
            )
            radiobutton.grid(row=1, column=i, padx=5, pady=5, sticky="ew")
            self.radiobuttons.append(radiobutton)

    def get(self):
        return self.variable.get()

    def set(self, value):
        self.variable.set(value)
