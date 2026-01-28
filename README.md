🎗️ Cervical Cancer Detection System
Machine Learning–Based Cervical Cancer Detection
This project is a machine learning–based web application that helps in early detection of cervical cancer using Pap smear images and basic patient details.

It is designed for educational and research purposes.

📌 Overview
Cervical cancer can be prevented if detected early.
This system uses a deep learning model to analyze Pap smear images and combines it with clinical information to predict cancer risk.

The output is shown as:

Low Risk

Moderate Risk

High Risk

A PDF report is also generated for reference.

✨ Features
Upload Pap smear image (JPG / PNG)

Enter basic patient details

Deep learning–based image analysis

Risk classification (Low / Moderate / High)

Automatic PDF report generation

Simple and user-friendly web interface

🛠️ Technologies Used
Python

PyTorch (Deep Learning)

Flask (Web Framework)

HTML & CSS

FPDF (PDF Report Generation)

NumPy & Pillow

📂 Project Structure
pgsql
Copy code
CERVICAL-CANCER-DETECTION-SYSTEM/
│
├── app.py                     → Flask web application
├── train_pytorch_model.py     → Model training script
├── requirements.txt           → Required Python packages
├── README.md                  → Project documentation
│
├── templates/
│   ├── index.html             → Input page
│   └── result.html            → Result page
│
├── uploads/                   → Uploaded images
├── reports/                   → Generated PDF reports
⚙️ Installation & Setup
Step 1: Clone the repository
bash
Copy code
git clone https://github.com/22eg105p61-droid/CARVICAL-CANCER-DETECTION-SYSTEM.git
cd CARVICAL-CANCER-DETECTION-SYSTEM
Step 2: Install dependencies
bash
Copy code
pip install -r requirements.txt
Step 3: Run the application
bash
Copy code
python app.py
Open browser and go to:

arduino
Copy code
http://localhost:5000
🧪 How It Works (Simple Logic)
User enters patient details

User uploads Pap smear image

Image is processed by the trained deep learning model

Model predicts abnormality probability

Clinical details add small risk weight

Final risk level is calculated

Result + PDF report is generated

📊 Risk Levels
Risk Level	Meaning
Low Risk	Normal condition
Moderate Risk	Needs follow-up
High Risk	Immediate medical consultation required

⚠️ Disclaimer
This project is only for educational and research purposes.

❌ Not for real medical diagnosis

❌ Not a replacement for doctors

✅ Used as a screening support tool

Always consult a qualified medical professional.

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgment
PyTorch Community

Flask Framework

Public medical datasets used for learning purposes

