import customtkinter as ctk
import tkinter as tk
from .training_pages import (
    DataProcessingPage,
    NNSelectionPage,
    ParametersPage,
    TrainingPage,
    ResultsPage,
)


class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ECG Classification")
        self.geometry("1400x1000")
        self.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (
            DataProcessingPage,
            NNSelectionPage,
            ParametersPage,
            TrainingPage,
            ResultsPage,
        ):
            page_name = F.__name__
            frame = F(controller=self)
            self.frames[page_name] = frame
            frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # self.show_frame("MainMenu")
        self.show_frame("DataProcessingPage")

    def show_frame(self, page_name):
        for frame in self.frames.values():
            if frame is not None:
                frame.place_forget()  # Скрыть все фреймы
        frame = self.frames[page_name]
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Отобразить нужный фрейм
        frame.tkraise()
