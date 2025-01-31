import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from file_manager import get_list_images
from config import IMAGES_DIR

class FrontEnd(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VHS Application")
        self.geometry("800x600")
        self.configure(bg = "#F0F0F0")
        self._create_frames()
        self._load_images_list()
    
    def _create_frames(self):
        # FRAME 1
        self.frame_files = tk.Frame(self, width = 200, bg = "lightgray")
        self.frame_files.pack(side = "left", fill = "y", padx = 5, pady = 5)
        self.label_title = tk.Label(self.frame_files, text="Images", font=("Arial", 12, "bold"), bg="lightgray")
        self.label_title.pack(pady=5)
        self.listbox = tk.Listbox(self.frame_files, height = 30)
        self.listbox.pack(fill = "both", expand = True, padx = 10, pady = 10)
        self.listbox.bind("<<ListboxSelect>>", self.on_image_select)

        # FRAME 2
        self.frame_image = tk.Frame(self, bg = "white")
        self.frame_image.pack(side = "left", fill = "both", expand = True,  padx = 5, pady = 5)
        self.label_image = tk.Label(self.frame_image, text = "Select an image", bg = "white")
        self.label_image.pack(pady = 20)

        # FRAME 3
        self.frame_info = tk.Frame(self, width = 250, bg = "lightgray")

        self.frame_info.pack(fill = "y", side = "right", padx = 5, pady = 5)
        self.info_text = tk.Label(self.frame_info, text = "Informations", font = ("Arial", 12, "bold"), bg = "lightgray")
        self.info_text.pack(pady = 10)
        self.no_info_text = tk.Label(self.frame_info, text = "No image selected", wraplength = 200, bg = "lightgray")
        self.no_info_text.pack(pady = 10)
        self.b1 = ttk.Button(self.frame_info, text = "AI Prediction VHS")
        self.b1.pack(pady = 20)
        self.b2 = ttk.Button(self.frame_info, text = "Download the report")
        self.b2.pack(pady = 30)
    
    def _load_images_list(self):
        images = get_list_images()
        for img in images:
            self.listbox.insert("end", img)
    
    def on_image_select(self, event):
        selection = self.listbox.curselection()
        if selection:
            img_name = self.listbox.get(selection[0])
            img_path = os.path.join(IMAGES_DIR, img_name)
            self.selected_image = img_name
            self.display_img(img_path)
            self.no_info_text.config(text = f"File: {img_name} \n Size: {os.path.getsize(img_path)//1024} KB ")
    
    def display_img(self, path):
        img = Image.open(path)
        img = img.resize((400,400), Image.Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(img)
        self.label_image.config(image = self.img_tk, text = "")



if __name__ == "__main__":
    app = FrontEnd()
    app.mainloop()

