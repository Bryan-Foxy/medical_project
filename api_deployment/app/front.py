import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import requests
from file_manager import get_list_images
from config import Config

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

class FrontEnd(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VHS Application")
        self.geometry("900x600")
        self.configure(bg="#F0F0F0")
        self.selected_image = None
        self.vhs_results = None
        self.predicted_label = None
        self._create_frames()
        self._load_images_list()

    def _create_frames(self):
        # Left Frame: File List
        self.frame_files = tk.Frame(self, width=200, bg="lightgray")
        self.frame_files.pack(side="left", fill="y", padx=5, pady=5)
        self.label_title = tk.Label(self.frame_files, text="Images", font=("Arial", 12, "bold"), bg="lightgray")
        self.label_title.pack(pady=5)
        self.listbox = tk.Listbox(self.frame_files, height=30)
        self.listbox.pack(fill="both", expand=True, padx=10, pady=10)
        self.listbox.bind("<<ListboxSelect>>", self.on_image_select)

        # Center Frame: Image Preview
        self.frame_image = tk.Frame(self, bg="white")
        self.frame_image.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.label_image = tk.Label(self.frame_image, text="Select an image", bg="white")
        self.label_image.pack(pady=20)

        # Right Frame: Info & Actions
        self.frame_info = tk.Frame(self, width=250, bg="lightgray")
        self.frame_info.pack(fill="y", side="right", padx=5, pady=5)
        self.info_text = tk.Label(self.frame_info, text="Informations", font=("Arial", 12, "bold"), bg="lightgray")
        self.info_text.pack(pady=10)
        self.no_info_text = tk.Label(self.frame_info, text="No image selected", wraplength=200, bg="lightgray")
        self.no_info_text.pack(pady=10)
        
        # Buttons
        self.b1 = ttk.Button(self.frame_info, text="AI Prediction VHS", command=self.prediction)
        self.b1.pack(pady=20)
        self.b2 = ttk.Button(self.frame_info, text="Download the report", command=self.download_report)
        self.b2.pack(pady=30)

    def _load_images_list(self):
        """Loads images from directory into listbox."""
        images = get_list_images()
        for img in images:
            self.listbox.insert("end", img)

    def on_image_select(self, event):
        """Displays the selected image and updates info text."""
        selection = self.listbox.curselection()
        if selection:
            img_name = self.listbox.get(selection[0])
            img_path = os.path.join(Config.IMAGES_DIR, img_name)
            self.selected_image = img_path
            self.display_img(img_path)
            self.no_info_text.config(text=f"File: {img_name}\nSize: {os.path.getsize(img_path)//1024} KB")

    def display_img(self, path):
        """Displays an image in the UI."""
        img = Image.open(path)
        img = img.resize((400, 400), Image.Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(img)
        self.label_image.config(image=self.img_tk, text="")

    def prediction(self):
        """Sends image to API for VHS analysis."""
        if not self.selected_image:
            messagebox.showerror("Error", "No image selected.")
            return
        
        url = f"{Config.API_URL}/vhs"
        with open(self.selected_image, "rb") as img_file:
            response = requests.post(url, files={"file": img_file})
        
        if response.status_code == 200:
            self.vhs_results = response.json()

            # Results formatted for better readability
            major, minor = self.vhs_results['major'], self.vhs_results['minor']
            major_text = f"Major Diameter: {major[0]} mm (Vertebrae: {major[1]})"
            minor_text = f"Minor Diameter: {minor[0]} mm (Vertebrae: {minor[1]})"
            vhs_text = f"VHS Score: {self.vhs_results['vhs_score']:.1f}"

            result_text = f"{major_text}\n{minor_text}\n{vhs_text}"
            self.no_info_text.config(text=result_text)

            # Load the predicted image (with the drawn measurements)
            predicted_img_path = self.vhs_results['output_path']
            self.display_predicted_img(predicted_img_path)

            messagebox.showinfo("Success", "Prediction complete.")
        else:
            messagebox.showerror("Error", "Prediction failed.")

    def display_predicted_img(self, predicted_img_path):
        """Displays the predicted image."""
        img = Image.open(predicted_img_path)
        img = img.resize((400, 400), Image.Resampling.LANCZOS)
        self.predicted_img_tk = ImageTk.PhotoImage(img)

        # If there is already a label for the predicted image, destroy it before creating a new one
        if self.predicted_label:
            self.predicted_label.destroy()

        # Create a new label and pack it
        self.predicted_label = tk.Label(self.frame_image, image=self.predicted_img_tk)
        self.predicted_label.pack(pady=10)

    def download_report(self):
        """Requests and downloads the VHS report as a PDF."""
        if not self.vhs_results:
            messagebox.showerror("Error", "No VHS data available.")
            return

        url = f"{Config.API_URL}/report"
        response = requests.post(url, json=self.vhs_results)

        if response.status_code == 200:
            pdf_path = os.path.join(Config.IMAGES_OUTPUT, "VHS_Report.pdf")
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            messagebox.showinfo("Success", f"Report downloaded:\n{pdf_path}")
        else:
            messagebox.showerror("Error", "Failed to generate report.")

if __name__ == "__main__":
    app = FrontEnd()
    app.mainloop()