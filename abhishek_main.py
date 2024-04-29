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
train_folder = 'documents/'

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

train_prompt = []

# train prompt
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

#LLM MODEL
def generate_response(prompt):
    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False    
    )
    return responses.text  

def upload_to_gcs(file_paths):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        blob = bucket.blob(f"pdf_files/{file_name}")
        blob.upload_from_filename(file_path)
        
    return [f"gs://{bucket_name}/pdf_files/{os.path.basename(path)}" for path in file_paths]



    
'''
---------  GRADIO UI -----
'''
import gradio as gr
import socket


def main_gradio(user_prompt, uploaded_files):
    user_prompt_list=[]
    user_prompt_list=user_prompt_list+[user_prompt]
    combined_prompt = train_prompt+ user_prompt_list
    
    if uploaded_files:
        file_paths = [file.name for file in uploaded_files]
        pdf_uris = upload_to_gcs(file_paths)

        for i in pdf_uris:
            combined_prompt = combined_prompt + [Part.from_uri(mime_type="application/pdf", uri=i)]
        
    print(combined_prompt)
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
        gr.Markdown("### Enter your test prompt and upload PDF files")
        with gr.Row():
            user_prompt = gr.Textbox(label="Test Prompt", placeholder="Enter your test prompt here...")
            file_input = gr.UploadButton(label="Upload PDF Files", file_types=["pdf"], file_count="multiple")

        submit_button = gr.Button("Generate Response")
        output_markdown = gr.Markdown(label="Output")

        submit_button.click(fn=main_gradio, inputs=[user_prompt, file_input], outputs=output_markdown)

    return demo

if __name__ == "__main__":
    interface = gradio_ui()
    interface.launch(server_name="0.0.0.0", server_port=port) 