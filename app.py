# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from datetime import datetime
from flask import render_template
import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from explainable_ai_code import visualize
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_dic= ["Benign","Melignant"]



#from model_predict  import pred_leaf_disease
from model_predict2 import detector_model
# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Skin Cancer Detection'
    return render_template('index.html', title=title)

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Skin Cancer Detection'
    
    if request.method == 'POST':
        file = request.files.get('file')

        img = Image.open(file)
        img.save('output.png')

        prediction = detector_model("output.png")
        visualize("output.png", 0.15)

        if prediction == "Benign":
            precaution = "Cancer is Not Melanoma"
        else:
            precaution = "Cancer is Melanoma"

        patient_id = request.form.get('patient_id')
        patient_name = request.form.get('patient_name')
        date_of_birth = request.form.get('date')
        gender = request.form.get('gender')

        # Calculate age based on date of birth
        dob_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
        today = datetime.today()
        age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
        
        # Validate age
        if age < 0:
            return "Invalid date of birth. Please enter a valid date."

        print(patient_id, patient_name, age, date_of_birth, gender)
        
        # Set up the canvas and page size
        import os

        # Create a folder with the patient name and ID
        folder_name = f"{patient_name}_{patient_id}"
        os.makedirs(folder_name, exist_ok=True)

        # Set up the canvas and page size
        pdf_file_name = f"{folder_name}/medical_report.pdf"
        c = canvas.Canvas(pdf_file_name, pagesize=letter)

        # Define the patient data
        skin_cancer = prediction

        # Write the report title
        c.setFontSize(16)
        c.drawString(50, 750, "Medical Report")

        # Write the patient information
        c.setFontSize(12)
        c.drawString(50, 700, "Patient ID: " + patient_id)
        c.drawString(50, 680, "Patient Name: " + patient_name)
        c.drawString(50, 660, "Age: " + str(age))  # Convert age to string
        c.drawString(50, 640, "Date: " + date_of_birth)  # Use date_of_birth directly
        c.drawString(50, 620, "Gender: " + gender)
        c.drawString(50, 600, "Skin Cancer Diagnosis: " + skin_cancer)

        # Save the PDF file
        c.save()

        return render_template('disease-result.html', prediction=prediction, precaution=precaution, title=title)

    return render_template('disease.html', title=title)

# render disease prediction result page


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
