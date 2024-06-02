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

        self.frame_nn_params = ctk.CTkFrame(self)
        self.frame_nn_params.grid(
            row=2, column=0, padx=20, pady=20, sticky="ew", columnspan=2
        )

        self.frame_other_params = ctk.CTkFrame(self)
        self.frame_other_params.grid(
            row=2, column=2, padx=20, pady=20, sticky="ew", columnspan=2
        )
        self.init_other_params()
        self.default_color = self.entry_epochs.cget("fg_color")

        self.button_next_page = ctk.CTkButton(
            self,
            text="Настройка гиперпараметров",
            command=self.next_page,
        )
        self.button_next_page.grid(row=3, column=3, padx=20, pady=(10, 20), sticky="ew")

    def init_hyperparameters(self, NNCreator, dimension):
        """Динамическое создание полей для ввода параметров нейронной сети

        Args:
            hyperparameters (NNCreator): В каждом файле,
            в котором определяется нейронная сеть, должен существовать класс NNCreator,
            где NN - архитектура нейронной сети
        """
        self.dimension = dimension
        self.NNCreator = NNCreator
        self.labels_params = []
        self.entrys_params = {}
        for i, (param, param_type) in enumerate(NNCreator.params.items()):
            label = ctk.CTkLabel(self.frame_nn_params, text=param)
            label.grid(row=i, column=0, padx=(20, 10), pady=20, sticky="ew")
            entry = ctk.CTkEntry(self.frame_nn_params)
            entry.grid(row=i, column=1, padx=(10, 20), pady=20, sticky="ew")

            self.labels_params.append(label)
            self.entrys_params[param] = entry

    def init_other_params(self):
        """Знаю х*йня. Я за*бался"""
        self.label_epochs = ctk.CTkLabel(self.frame_other_params, text="epochs")
        self.label_epochs.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="ew")
        self.entry_epochs = ctk.CTkEntry(self.frame_other_params)
        self.entry_epochs.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="ew")

        self.label_batch_size = ctk.CTkLabel(self.frame_other_params, text="batch_size")
        self.label_batch_size.grid(row=1, column=0, padx=(20, 10), pady=20, sticky="ew")
        self.entry_batch_size = ctk.CTkEntry(self.frame_other_params)
        self.entry_batch_size.grid(row=1, column=1, padx=(10, 20), pady=20, sticky="ew")

        self.label_learning_rate = ctk.CTkLabel(
            self.frame_other_params, text="learning_rate"
        )
        self.label_learning_rate.grid(
            row=2, column=0, padx=(20, 10), pady=20, sticky="ew"
        )
        self.entry_learning_rate = ctk.CTkEntry(self.frame_other_params)
        self.entry_learning_rate.grid(
            row=2, column=1, padx=(10, 20), pady=20, sticky="ew"
        )

        self.label_l2_decay = ctk.CTkLabel(self.frame_other_params, text="l2_decay")
        self.label_l2_decay.grid(row=3, column=0, padx=(20, 10), pady=20, sticky="ew")
        self.entry_l2_decay = ctk.CTkEntry(self.frame_other_params)
        self.entry_l2_decay.grid(row=3, column=1, padx=(10, 20), pady=20, sticky="ew")

        self.optimizer = ctk.StringVar(value="adam")
        self.label_optimizer = ctk.CTkLabel(self.frame_other_params, text="optimizer")
        self.label_optimizer.grid(row=4, column=0, padx=(20, 10), pady=20, sticky="ew")
        self.optionmenu_optimizer = ctk.CTkOptionMenu(
            self.frame_other_params,
            values=["adam", "sgd"],
            variable=self.optimizer,
        )
        self.optionmenu_optimizer.grid(
            row=4, column=1, padx=(10, 20), pady=20, sticky="ew"
        )

    def next_page(self):
        """Нет проверки на положительность числа"""
        params_other = {}

        params_other["epochs"] = self.entry_epochs.get()
        params_other["batch_size"] = self.entry_batch_size.get()
        params_other["learning_rate"] = self.entry_learning_rate.get()
        params_other["l2_decay"] = self.entry_l2_decay.get()
        params_other["optimizer"] = self.optimizer.get()

        try:
            tmp = "epochs"
            params_other[tmp] = int(params_other[tmp])
            getattr(self, f"entry_{tmp}").configure(fg_color=self.default_color)
            tmp = "batch_size"
            params_other[tmp] = int(params_other[tmp])
            getattr(self, f"entry_{tmp}").configure(fg_color=self.default_color)
            tmp = "learning_rate"
            params_other[tmp] = float(params_other[tmp])
            getattr(self, f"entry_{tmp}").configure(fg_color=self.default_color)
            tmp = "l2_decay"
            params_other[tmp] = float(params_other[tmp])
            getattr(self, f"entry_{tmp}").configure(fg_color=self.default_color)
        except ValueError:
            getattr(self, f"entry_{tmp}").configure(fg_color="red")
            return

        for entrys in self.entrys_params.values():
            entrys.configure(fg_color=self.default_color)

        params_nn = {param: entry.get() for param, entry in self.entrys_params.items()}
        NN = self.NNCreator.create(**params_nn)

        if isinstance(NN, str):
            self.entrys_params[NN].configure(fg_color="red")
            return

        self.controller.show_frame("NNTrainingPage")

    def prev_page(self):
        for label, entry in zip(self.labels_params, self.entrys_params.values()):
            label.destroy()
            entry.destroy()
        del self.labels_params
        del self.entrys_params

        self.controller.show_frame("NNSelectionPage")
