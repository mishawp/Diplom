import customtkinter as ctk
import tkinter as tk


class PredictPage(ctk.CTkFrame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller

        label = ctk.CTkLabel(self, text="Обучить нейронную сеть")
        label.pack(pady=10, padx=10)

        back_button = ctk.CTkButton(
            self, text="Назад", command=lambda: controller.show_frame("MainMenu")
        )
        back_button.pack(pady=10, padx=10)
