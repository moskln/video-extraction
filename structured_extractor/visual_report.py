# structured_extractor/visual_report.py

from fpdf import FPDF
from PIL import Image
import os
import textwrap

def generate_visual_report(image_paths, explanations, output_pdf="static/visual_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)

    if os.path.exists("DejaVuSans.ttf"):
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", "", 12)
    else:
        pdf.set_font("Arial", "", 12)

    for img_path, explanation in zip(image_paths, explanations):
        pdf.add_page()

        try:
            # Resize image to fit width
            img = Image.open(img_path)
            img.thumbnail((170, 170))
            temp_path = f"{img_path}_thumb.jpg"
            img.save(temp_path)
            pdf.image(temp_path, w=100)
            os.remove(temp_path)
        except Exception as e:
            print(f"[WARN] Failed to embed image {img_path}: {e}")

        wrapped = "\n".join(textwrap.fill(line, width=100) for line in explanation.split("\n"))
        pdf.ln(5)
        pdf.multi_cell(0, 10, wrapped)

    pdf.output(output_pdf)
    print(f"[✅] Visual report saved to: {output_pdf}")
