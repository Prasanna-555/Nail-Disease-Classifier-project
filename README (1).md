# 🧠 Nail Disease Classification using Deep Learning

This is a Streamlit-based web application that classifies nail diseases using a trained PyTorch deep learning model. The app supports real-time image upload or webcam input and provides disease prediction along with risk percentages, treatment suggestions, PDF reports, and more.

---

## 🚀 Features

- Upload or capture nail images
- Predict disease class (e.g., Beau’s Lines, Leukonychia, etc.)
- Shows risk percentage
- Displays treatment and precautions
- Downloadable PDF report
- City-based doctor suggestions via Google Maps
- Gemini-powered chatbot assistant
- Multilingual support and voice diagnosis
- Admin dashboard with user history and statistics

---

## 🛠️ Tech Stack

- Python, PyTorch
- Streamlit
- OpenCV
- ReportLab (for PDF)
- Google Maps API
- Gemini API

---

## 🔧 Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
```

---

## 📂 Download Dataset & Model

Since GitHub has a 100MB file size limit, download the dataset and model file separately:

- 🔗 [Download Trained Model (.pth)](https://drive.google.com/file/d/YOUR_MODEL_LINK/view?usp=sharing)
- 🔗 [Download Dataset (Images)](https://drive.google.com/file/d/YOUR_DATASET_LINK/view?usp=sharing)

After downloading:
- Place the `.pth` file inside the root directory
- Place the `dataset/` folder in the project root as well

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📸 Screenshot

url("<img width="1354" height="614" alt="Screenshot 2025-07-23 152426" src="https://github.com/user-attachments/assets/a3e2137f-6d2a-440a-a472-4db855cc63be" /">
)

---

## 📃 License

This project is open source and available under the [MIT License](LICENSE).
