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





https://github.com/user-attachments/assets/9ad9ead3-4153-41ae-8bbb-8bbc3ee7e5a7



Perfect! Here’s a **well-formatted README.md version** for your GitHub repository, including space for the video link:

````markdown
# Task 5: Medical Cost Prediction App

This repository contains **Task 5** of the DS-ML Internship.  

The project is a **Medical Cost Prediction App** that estimates annual medical costs based on personal health data. Users can also **upload new datasets**, which automatically retrains the machine learning model to keep predictions accurate and up-to-date.  

When a new dataset is uploaded, it triggers a **GitHub Actions workflow** that:  
1. Installs dependencies  
2. Updates the dataset  
3. Runs `train.py` to update the model  
4. Rebuilds the Gradio app  

## Demo
- **App:** [Medical Cost Prediction App](https://huggingface.co/spaces/BUFON-JOKER/task5)  
- **Video demo:** [Watch Video](YOUR_VIDEO_LINK_HERE)

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
```bash
git clone https://github.com/BUFONJOKER/task5.git
cd task5
pip install -r requirements.txt
python app.py
````

---

Feel free to **replace `YOUR_VIDEO_LINK_HERE`** with the actual video URL so anyone visiting your repo can see the demo.

```

If you want, I can also **add a nice badge for GitHub Actions** showing the workflow status and make the README look even more professional. Do you want me to do that?
```
