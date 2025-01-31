import sys
import os
import ollama
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

class Diagnostic:
    def __init__(self):
        self.PROMPT = """
        You are an assistant 
        """
    def diagnostize(self):
        return 0


class GenerateReport:
    def __init__(self):
        return 0


