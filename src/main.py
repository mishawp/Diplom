from dotenv import load_dotenv
from GUI import MainApp

if __name__ == "__main__":
    load_dotenv()
    app = MainApp()
    app.mainloop()
