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
    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False    
    )
    return responses



#------------------------------------

test_prompt=["I want to design a PDF to JSON extractor using Gemini PRo. I want to use GCS for storage and vertex ai workbench. Suggest me the most efficient GCP solution stack to build this out"]

combined_prompt = train_prompt+test_prompt
print(combined_prompt)


def main_gradio():
    output_response = generate_response(combined_prompt)
    output_text = output_response.text
    return output_text

# print(output_text)




    
'''
---------  GRADIO UI -----
'''
def handle_action(action, json_filename=""):
    if action == "Generate Response":
        return main_gradio()
    else:
        return "Invalid action."


import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to all interfaces and a free port chosen by the OS
        s.listen(1)
        port = s.getsockname()[1]  # Get the port number
        return port
    
port = find_free_port()
print(port)

# Create your Gradio UI interface
combined_interface = gr.Interface(
fn=handle_action,
inputs=[
    gr.Radio(["Generate Response"], label="Select Action")
],
outputs="text",
title="Sustainable AI"
)

combined_interface.launch(server_name="0.0.0.0", server_port=57997)