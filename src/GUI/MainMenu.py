import customtkinter as ctk
import tkinter as tk


class MainMenu(ctk.CTkFrame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller

        self.button_train = ctk.CTkButton(
            self,
            text="Обучить нейронную сеть",
            command=lambda: controller.show_frame("DataProcessingPage"),
        )
        self.button_train.grid(row=0, column=0, padx=30, pady=30, sticky="ew")

        self.button_about = ctk.CTkButton(
            self,
            text="Диагностировать",
            command=lambda: controller.show_frame("PredictPage"),
        )
        self.button_about.grid(row=1, column=0, padx=30, pady=(10, 30), sticky="ew")

        # self.configure(border_width=1, border_color="red")
