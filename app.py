import gradio as gr
import pandas as pd
import os
from github import Github
import joblib

# --- 1. Train/Load Model (Simplified for Demo) ---
# In a real app, you would load 'model.pkl' from disk
model_scaler_mae = joblib.load("model/model_scaler_mae.pkl")
model = model_scaler_mae["model"]
scaler = model_scaler_mae["scaler"]
mae = model_scaler_mae["mae"]
model_version = model_scaler_mae['model_version']
updated_at = model_scaler_mae['updated_at']

# --- 2. Helper Functions ---
def predict_cost(age, bmi, risk_score, chronic_count, provider_quality):
    new_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'risk_score': [risk_score],
    'chronic_count': [chronic_count],
    'provider_quality': [provider_quality]
    })

    scaled_features = scaler.transform(new_data)
    prediction = model.predict(scaled_features)
    return f"{prediction[0]:.2f} USD (± {mae:.2f} USD)"

def upload_and_trigger_retrain(file_obj):
    if file_obj is None:
        return "⚠️ No file uploaded."
    
    # Get Token from HF Secrets
    token = os.getenv("GH_TOKEN")
    if not token:
        return "❌ Error: GH_TOKEN secret is missing in Hugging Face settings."

    try:
        # Connect to GitHub
        g = Github(token)
        # UPDATE THIS LINE with your details!
        repo = g.get_repo("BUFONJOKER/task5") 
        
        # Read the uploaded csv content
        with open(file_obj.name, "r") as f:
            new_content = f.read()

        # Define path in repo
        file_path = "data/dataset.csv"
        
        # Check if file exists to decide 'create' or 'update'
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(file_path, "Update data via HF Upload CSV Panel", new_content, contents.sha)
            action = "Updated"
        except:
            repo.create_file(file_path, "Create data via HF Upload CSV Panel", new_content)
            action = "Created"
            
        return f"✅ Success! {action} 'data/dataset.csv' on GitHub. \n🚀 Training Pipeline triggered!"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- 3. Build App with Tabs ---
with gr.Blocks(title="Annual Medical Cost Prediction") as app:
    gr.Markdown("# Annual Medical Cost AI System")
    gr.Markdown(f"**Model Version:** {model_version}  \n**Last Updated:** {updated_at} \n**Model MAE:** {mae:.2f} USD")

    gr.Markdown("Predict annual medical costs based on personal health metrics. Use the admin tab to upload new data and retrain the model.")
    gr.Markdown("---")

    with gr.Tab("🔮 Prediction"):
        with gr.Row():
            age = gr.Number(label="Age [0-100]")
            bmi = gr.Number(label="BMI [10-60]")
            risk_score = gr.Number(label="Risk Score [0.0-1.0]")
            chronic_count = gr.Number(label="Chronic Count [0-10]")
            provider_quality = gr.Number(label="Provider Quality [0-10]")
        btn = gr.Button("Predict", variant="primary")
        out = gr.Textbox(label="Annual Medical Cost Prediction")
        btn.click(predict_cost, [age, bmi, risk_score, chronic_count, provider_quality], out)

    with gr.Tab("📂 Upload New Data to Retrain Model"):
        gr.Markdown("### 📂 Upload New Data to Retrain Model")
        gr.Markdown("*Uploading here will push to GitHub and start the training workflow.*")
        
        file_input = gr.File(label="Upload csv Dataset", file_types=[".csv"])
        upload_btn = gr.Button("Upload & Train", variant="stop")
        status_out = gr.Textbox(label="System Logs")
        
        upload_btn.click(upload_and_trigger_retrain, file_input, status_out)

# Add password protection to the whole app (Optional but recommended)
# app.launch(auth=("admin", "pass123")) 
app.launch()