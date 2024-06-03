import os
import importlib
import customtkinter as ctk
from pathlib import Path

PROJECT_ROOT = os.getenv("project_root")


class HyperparametersPage(ctk.CTkFrame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        self.path_model = Path(PROJECT_ROOT, "models")
        self.path_reports = Path(PROJECT_ROOT, "reports")

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

        self.frame_dataset = DatasetFrame(self)  # см. в init_hyperparameters
        self.frame_dataset.grid(
            row=3, column=0, padx=20, pady=20, sticky="ew", columnspan=4
        )

        self.frame_parameters = ParametersFrame(self)
        self.frame_parameters.grid(
            row=2, column=1, padx=20, pady=20, sticky="ew", columnspan=2
        )

        self.label_name = ctk.CTkLabel(self, text="Назовите модель: ")
        self.label_name.grid(row=4, column=2, padx=20, pady=10, sticky="ew")

        self.entry_name = ctk.CTkEntry(self)
        self.entry_name.grid(row=4, column=3, padx=20, pady=10, sticky="ew")

        self.label_name_status = ctk.CTkLabel(self, text="")
        self.label_name_status.grid(row=5, column=3, padx=20, pady=10, sticky="ew")

        self.button_next_page = ctk.CTkButton(
            self,
            text="Обучить модель",
            command=self.next_page,
        )
        self.button_next_page.grid(row=6, column=3, padx=20, pady=(10, 20), sticky="ew")

    def set_args(self, **kwargs):
        """Получение параметров из предыдущей страницы

        Args:
            path_nn (str): путь к нейронной сети
            dimension (int): 1 или 2 (2D или 1D нейросеть будет обучаться)
        """

        self.frame_dataset.init(kwargs["dimension"])

    def prev_page(self):
        self.frame_dataset.destroy()
        self.controller.show_frame("NNSelectionPage")

    def next_page(self):
        dataset_name = self.frame_dataset.get()
        if dataset_name == "":
            return

        model_class = importlib.import_module(self.path_nn)

        self.controller.frames["TrainingPage"].set_model(
            model_class, self.dimension, model_class.__name__, dataset_name
        )
        self.controller.show_frame("TrainingPage")


class NNFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

    def init(self, path_nn: Path | str):
        """Динамическое создание фрейма

        Args:
            path_nn (Path | str): путь к реализации класса нейронной сети
        """
        pass


class ParametersFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.grid_columnconfigure(0, weight=1)
        self.label_parameters = {}.fromkeys(
            ("epochs", "batch_size", "learning_rate", "l2_decay", "optimizer")
        )
        self.entry_parameters = {}.fromkeys(
            ("epochs", "batch_size", "learning_rate", "l2_decay", "optimizer")
        )

        self.title = ctk.CTkLabel(
            self, text="Параметры", fg_color="gray30", corner_radius=6
        )
        self.title.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        for i, param in enumerate(self.parameters, start=1):
            label = ctk.CTkLabel(self, text=param)
            label.grid(row=i, column=0, padx=(20, 10), pady=20, sticky="ew")

            if param == "optimizer":
                self.var_optimizer = ctk.StringVar(value="adam")
                entry = ctk.CTkOptionMenu(
                    self,
                    values=["adam", "sgd"],
                    variable=self.var_optimizer,
                )
                entry.grid(row=i, column=1, padx=(10, 20), pady=20, sticky="ew")
            else:
                entry = ctk.CTkEntry(self)
                entry.grid(row=i, column=1, padx=(10, 20), pady=20, sticky="ew")

            self.label_parameters[param] = label
            self.entry_parameters[param] = entry

        self.default_color = self.entry_parameters["batch_size"].cget("fg_color")

    def check_parameters(self):
        for entry in self.entry_parameters.value():
            if isinstance(entry, ctk.CTkEntry):
                entry.configure(fg_color=self.default_color)

        for param, entry in self.entry_parameters.items():
            value = entry.get()
            if param == "epochs" or param == "batch_size":
                if not value.isdigit():
                    entry.configure(fg_color="red")
                    return False
            elif param == "learning_rate" or param == "l2_decay":
                try:
                    value = float(value)
                except ValueError:
                    entry.configure(fg_color="red")
                    return False

                if not (0 <= value <= 1):
                    entry.configure(fg_color="red")
                    return False

    def get_parameters(self):
        return self.entry_parameters


class DatasetFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

    def init(self, dimension):
        self.dimension = dimension
        self.path_data_processed = Path(
            PROJECT_ROOT, "data", "processed", f"{self.dimension}D"
        )
        self.processed_datasets = tuple(self.path_data_processed.iterdir())

        self.grid_columnconfigure(tuple(range(len(self.processed_datasets))), weight=1)

        self.title = ctk.CTkLabel(
            self, text="Датасет", fg_color="gray30", corner_radius=6
        )
        self.title.grid(
            row=0,
            column=0,
            columnspan=len(self.processed_datasets),
            padx=10,
            pady=10,
            sticky="ew",
        )

        self.variable = ctk.StringVar(value="")
        self.radiobuttons = []
        for i, dataset in enumerate(self.processed_datasets):
            radiobutton = ctk.CTkRadioButton(
                self, text=dataset.name, variable=self.variable, value=dataset.name
            )
            radiobutton.grid(row=1, column=i, padx=5, pady=5, sticky="ew")
            self.radiobuttons.append(radiobutton)

    def get_dataset(self):
        return self.variable
