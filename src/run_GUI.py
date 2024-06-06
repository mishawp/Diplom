from dotenv import load_dotenv

load_dotenv()
from GUI import MainApp

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
