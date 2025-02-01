import os
import ollama
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors

# Diagnostic Class
class Diagnostic:
    def __init__(self, major, minor, vhs_score):
        self.major = major
        self.minor = minor
        self.vhs_score = vhs_score
        self.prompt = f"""
        You are an expert assistant in a veterinary clinic. Your task is to interpret the Vertebral Heart Size (VHS) results for animals, mainly dogs and cats.
        Based on the major and minor heart diameters and the VHS score, provide an initial interpretation to assist veterinarians.
        
        Major Diameter: {major}
        Minor Diameter: {minor}
        VHS Score: {vhs_score}
        The major diameter and the minor are tuples containing (value in millimeters, transposition to the vertebrae)
        
        Please return only the expert interpretation in a clear and professional manner.
        """

    def _diagnostize(self):
        """Generates a diagnostic interpretation using Ollama LLM."""
        response = ollama.chat(model="monotykamary/medichat-llama3:latest", messages=[
            {"role": "user", "content": self.prompt},
        ])
        return response['message']['content']

    def generate_report(self, img_path, predicted_path):
        """Generates a professional PDF report with the original and predicted images side by side."""
        pdf_path = f"{os.path.splitext(img_path)[0]}.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # Report Content
        elements = [
            Paragraph("Veterinary Medical Report", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"<b>Patient Name:</b> Unknown", styles["Normal"]),
            Paragraph(f"<b>Species:</b> Dog/Cat", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>VHS Measurements:</b>", styles["Heading2"]),
            Paragraph(f"- Major Diameter: {self.major} mm", styles["Normal"]),
            Paragraph(f"- Minor Diameter: {self.minor} mm", styles["Normal"]),
            Paragraph(f"- VHS Score: {self.vhs_score}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>Interpretation:</b>", styles["Heading2"]),
            Paragraph(self._diagnostize(), styles["Normal"]),
            Spacer(1, 12),
            Paragraph("<b>Images:</b>", styles["Heading2"]),
            Spacer(1, 12),
        ]

        # Load images
        img_original = Image(img_path, width=200, height=200)
        img_predicted = Image(predicted_path, width=200, height=200)

        # Align images side by side
        image_table = Table([[img_original, img_predicted]])
        image_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(image_table)
        elements.append(Spacer(1, 12))

        # Build PDF
        doc.build(elements)
        return pdf_path