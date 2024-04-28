import os
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
from google.cloud import storage
from google.cloud import aiplatform

import json
import re
import gradio as gr


project_id = "ds-ml-pod"
location = "us-central1"
bucket_name = "sustainable-ai"


# Paths
train_folder = 'documents2/'

aiplatform.init(project=project_id, location=location)
vertexai.init(project=project_id, location=location)

# Read example ip and op pdfs from train folder
def list_train_folder_files(bucket, train_folder):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    train_pdfs = []
    
    blobs = bucket.list_blobs(prefix=train_folder)
    for blob in blobs:
        if blob.name.endswith('.pdf'):
            train_pdfs.append(blob.name)
       
    return train_pdfs


# Prompt

# List PDF files and find corresponding JSON output files
train_pdf_list = list_train_folder_files(bucket_name, train_folder)

# Construct the prompt for each input and its corresponding output JSON
train_prompt = []

# Initial prompt setup
train_prompt.append("""You are a Sustainable AI trained to suggesting the most CO2 efficient Google Cloud Platform technical stack based on user usecase.
Understand and use the following pieces of context to answer the question at the end. Think step-by-step and then answer. Always explain why you are answering the question the way you are.
""")


# Prepare train prompt
for train_pdf in train_pdf_list:
    train_pdf_uri = f"gs://{bucket_name}/{train_pdf}"


    # Append dynamically generated content to the prompt string
    train_prompt += [
        "Context Input Text:  ",
        Part.from_uri(mime_type="application/pdf", uri=train_pdf_uri)
    ]

# -----------------------------

# Gemini 1.5 model

vertexai.init(project=project_id, location=location)
model = GenerativeModel("gemini-1.5-pro-preview-0409")

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.4,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}



# LLM Model
def generate_response(prompt):
    # Placeholder function, replace with actual model call
    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False    
    )
    return responses.text 




    
'''
---------  GRADIO UI -----
'''
import gradio as gr
import socket

def main_gradio(user_prompt):
    train_prompt = "Predefined training prompt or context information here."  # Adjust as needed
    combined_prompt = train_prompt + " " + user_prompt
    output_text = generate_response(combined_prompt)
    return output_text

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        return port

port = find_free_port()
print(f"Using free port: {port}")

def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("### Enter your test prompt")
        user_prompt = gr.Textbox(label="Test Prompt", placeholder="Enter your test prompt here...")
        submit_button = gr.Button("Generate Response")
        output_text = gr.Textbox(label="Output", placeholder="Output will appear here...")

        submit_button.click(fn=main_gradio, inputs=user_prompt, outputs=output_text)

    return demo

if __name__ == "__main__":
    interface = gradio_ui()
    interface.launch(server_name="0.0.0.0", server_port=port)  # Use the dynamically found free port