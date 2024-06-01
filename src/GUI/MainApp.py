import customtkinter as ctk
import tkinter as tk
from .MainMenu import MainMenu
from .training_pages import DataProcessingPage, NNSelectionPage, HyperparametersPage
from .predict_pages import PredictPage


class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ECG Classification")
        self.geometry("1600x800")
        self.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (
            MainMenu,
            DataProcessingPage,
            PredictPage,
            NNSelectionPage,
            HyperparametersPage,
        ):
            page_name = F.__name__
            frame = F(controller=self)
            self.frames[page_name] = frame
            frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.show_frame("MainMenu")

    def show_frame(self, page_name):
        for frame in self.frames.values():
            frame.place_forget()  # Скрыть все фреймы
        frame = self.frames[page_name]
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Отобразить нужный фрейм
        frame.tkraise()
