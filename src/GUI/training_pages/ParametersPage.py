import os
import importlib
import customtkinter as ctk
from pathlib import Path

PROJECT_ROOT = os.getenv("project_root")


class ParametersPage(ctk.CTkFrame):
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

        self.frame_parameters = ParametersFrame(self)
        self.frame_parameters.grid(
            row=2, column=0, padx=20, pady=20, sticky="ew", columnspan=2
        )

        self.frame_nn = NNFrame(self)
        self.frame_nn.grid(
            row=2, column=2, padx=20, pady=20, sticky="nsew", columnspan=2
        )

        self.frame_dataset = DatasetFrame(self)  # см. в init_hyperparameters
        self.frame_dataset.grid(
            row=3, column=0, padx=20, pady=20, sticky="ew", columnspan=4
        )

        self.frame_model_naming = ModelNamingFrame(self)
        self.frame_model_naming.grid(
            row=4, column=2, padx=20, pady=20, sticky="ew", columnspan=2
        )

        self.button_next_page = ctk.CTkButton(
            self,
            text="Обучить модель",
            command=self.next_page,
        )
        self.button_next_page.grid(row=5, column=3, padx=20, pady=(10, 20), sticky="ew")

    def set_args(self, **kwargs):
        """Получение параметров из предыдущей страницы

        kwargs:
            path_nn (str): путь к нейронной сети
            dimension (int): 1 или 2 (2D или 1D нейросеть будет обучаться)
        """
        self.kwargs = kwargs
        self.frame_nn.init(kwargs["path_nn"])
        self.frame_dataset.init(kwargs["dimension"])

    def prev_page(self):
        for widget in self.frame_dataset.winfo_children():
            widget.destroy()
        self.controller.show_frame("NNSelectionPage")

    def next_page(self):
        self.frame_model_naming.reset_color()
        self.frame_parameters.reset_color()

        if not self.frame_parameters.check_parameters():
            return
        if not self.frame_model_naming.check_name():
            return

        model_name = self.frame_model_naming.get()
        parameters = self.frame_parameters.get()

        dataset_name = self.frame_dataset.get()
        if dataset_name == "":
            return

        path_nn = self.frame_nn.get()
        # src.nn1(2)d
        module = ".".join(path_nn.parts[-3:-1])
        model_class = getattr(importlib.import_module(module), path_nn.stem)

        self.controller.frames["TrainingPage"].set_args(
            model_class=model_class,
            dimension=self.kwargs["dimension"],
            model_name=model_name,
            dataset_name=dataset_name,
            parameters=parameters,
        )
        self.controller.show_frame("TrainingPage")


class ParametersFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.grid_columnconfigure(0, weight=1)
        self.parameters = {}.fromkeys(
            ("epochs", "batch_size", "learning_rate", "l2_decay", "optimizer", "device")
        )
        self.label_parameters = {}.fromkeys(self.parameters.keys())
        self.entry_parameters = {}.fromkeys(self.parameters.keys())

        self.title = ctk.CTkLabel(
            self, text="Параметры", fg_color="gray30", corner_radius=6
        )
        self.title.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

        for i, param in enumerate(self.entry_parameters, start=1):
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
            elif param == "device":
                self.var_device = ctk.StringVar(value="cuda")
                entry = ctk.CTkOptionMenu(
                    self,
                    values=["cuda", "cpu", "mps"],
                    variable=self.var_device,
                )
                entry.grid(row=i, column=1, padx=(10, 20), pady=20, sticky="ew")
            else:
                entry = ctk.CTkEntry(self)
                entry.grid(row=i, column=1, padx=(10, 20), pady=20, sticky="ew")

            self.label_parameters[param] = label
            self.entry_parameters[param] = entry

        self.default_color = self.entry_parameters["batch_size"].cget("fg_color")

    def check_parameters(self):
        for param, entry in self.entry_parameters.items():
            value = entry.get()
            if param == "epochs" or param == "batch_size":
                if not value.isdigit():
                    entry.configure(fg_color="red")
                    return False
                else:
                    self.parameters[param] = int(value)
            elif param == "learning_rate" or param == "l2_decay":
                try:
                    value = float(value)
                except ValueError:
                    entry.configure(fg_color="red")
                    return False

                if not (0 <= value <= 1):
                    entry.configure(fg_color="red")
                    return False
                else:
                    self.parameters[param] = float(value)
        self.parameters["optimizer"] = self.var_optimizer.get()
        self.parameters["device"] = self.var_device.get()
        return True

    def reset_color(self):
        for entry in self.entry_parameters.values():
            if isinstance(entry, ctk.CTkEntry):
                entry.configure(fg_color=self.default_color)

    def get(self):
        return self.parameters


class NNFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

    def init(self, path_nn: Path | str):
        """Динамическое создание фрейма

        Args:
            path_nn (Path | str): путь к реализации класса нейронной сети
        """
        self.path_nn = Path(path_nn)
        self.label = ctk.CTkLabel(
            self, text=self.path_nn.stem, fg_color="gray30", corner_radius=6
        )
        self.label.grid(row=0, column=0, padx=20, pady=20, sticky="ew", columnspan=2)

        self.button_open_nn = ctk.CTkButton(
            self, text="Открыть код", command=lambda: os.startfile(self.path_nn)
        )
        self.button_open_nn.grid(
            row=1, column=0, padx=20, pady=20, sticky="ew", columnspan=2
        )

    def get(self):
        return self.path_nn


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

    def get(self):
        return self.variable.get()


class ModelNamingFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.path_model = Path(PROJECT_ROOT, "models")
        self.path_reports = Path(PROJECT_ROOT, "reports")

        self.label_name = ctk.CTkLabel(self, text="Назовите модель: ")
        self.label_name.grid(row=0, column=2, padx=20, pady=10, sticky="ew")

        self.entry_name = ctk.CTkEntry(self)
        self.entry_name.grid(row=0, column=3, padx=20, pady=10, sticky="ew")

        self.label_name_status = ctk.CTkLabel(self, text="")
        self.label_name_status.grid(row=1, column=3, padx=20, pady=10, sticky="ew")

        self.default_color = self.entry_name.cget("fg_color")

    def check_name(self):
        name = self.entry_name.get()

        if name == "":
            self.label_name_status.configure(text="Название не может быть пустым")
            self.entry_name.configure(fg_color="red")
            return False
        if name in self.path_model.iterdir():
            self.label_name_status.configure(
                text="Модель с таким именем уже существует"
            )
            self.entry_name.configure(fg_color="red")
            return False
        if name in self.path_reports.iterdir():
            self.label_name_status.configure(
                text="Модель с таким именем уже существует"
            )
            self.entry_name.configure(fg_color="red")
            return False

        return True

    def reset_color(self):
        self.label_name_status.configure(text="")
        self.entry_name.configure(fg_color=self.default_color)

    def get(self):
        return self.entry_name.get()
