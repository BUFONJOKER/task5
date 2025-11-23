---
title: Task5
emoji: 🏢
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "6.0.0"
app_file: app.py
pinned: false
---

# Task 5: Annual Medical Cost AI System

This repository contains **Task 5** of the DS-ML Internship.  

The project is a **Annual Medical Cost AI System** that estimates annual medical costs based on personal health data. Users can also **upload new datasets**, which automatically retrains the machine learning model to keep predictions accurate and up-to-date.  

When a new dataset is uploaded, it triggers a **GitHub Actions workflow** that:  
1. Installs dependencies  
2. Updates the dataset  
3. Runs `train.py` to update the model  
4. Rebuilds the Gradio app  

## Demo
- **App:** [Annual Medical Cost AI System](https://huggingface.co/spaces/BUFON-JOKER/task5)  
- **Video demo:** [Watch Video](https://github.com/user-attachments/assets/9ad9ead3-4153-41ae-8bbb-8bbc3ee7e5a7)

https://github.com/user-attachments/assets/9ad9ead3-4153-41ae-8bbb-8bbc3ee7e5a7

## Tech Stack
- **Gradio** – Web app interface  
- **Hugging Face Spaces** – Hosting  
- **GitHub Actions** – Automated ML pipeline  
- **Pandas** – Data manipulation  
- **Scikit-learn** – Model building  

## Features
- Input personal health data to predict medical costs  
- Upload new datasets → triggers automated ML pipeline  
- Model retraining and Gradio app rebuild happen automatically  

## Running Locally
To run the app locally:  
```
git clone https://github.com/BUFONJOKER/task5.git
```
```
cd task5
```
```
pip install -r requirements.txt
```
```
python app.py
```

---
