import os
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
from google.cloud import storage
from google.cloud import aiplatform

import gradio as gr




project_id = "ds-ml-pod"
location = "us-central1"
bucket_name = "sustainable-ai"


# Paths
train_folder = 'documents/'
input_folder= 'user_files/'
output_folder= 'output_solution/'

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
train_prompt.append("""
You are an AI trained in sustainable technologies, specializing in suggesting the most CO2-efficient Google Cloud Platform (GCP) technical stack according to the user's specific use case. In responding, consider the following steps:

Analyze Requirements: Based on the provided use case, analyze the computational intensity, storage needs, and potential scalability. This analysis will help in selecting the most suitable and eco-friendly GCP services.
Suggest GCP Services: Recommend a set of GCP services that align with the CO2 efficiency goals. Include options for computing services, storage solutions, and any relevant management tools.
Explain Your Choices: For each suggested service, explain why it is considered CO2-efficient in the context of the user’s needs. Discuss any trade-offs and suggest best practices for optimizing resource usage.
Finally, answer the following question based on the information provided. Always support your answer with clear, step-by-step reasoning that explains your choices.
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

def upload_to_gcs(file_paths, folder):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        blob = bucket.blob(f"{folder}{file_name}")
        blob.upload_from_filename(file_path)
        
    return [f"gs://{bucket_name}/{folder}{os.path.basename(path)}" for path in file_paths]

def upload_output_to_gcs(output_text, output_folder):
    # Create a text file from output text
    text_output_path = "output.md"
    with open(text_output_path, "w") as text_file:
        text_file.write(output_text)
    
    # Upload the text file to GCS
    file_uri = upload_to_gcs([text_output_path], output_folder)
    return file_uri[0], text_output_path

    
'''
---------  GRADIO UI -----
'''


def main_gradio(user_prompt, uploaded_files):
    user_prompt_list = [user_prompt]
    combined_prompt = train_prompt + user_prompt_list
    
    if uploaded_files:
        file_paths = [file.name for file in uploaded_files]
        pdf_uris = upload_to_gcs(file_paths, input_folder)

        for uri in pdf_uris:
            combined_prompt += [Part.from_uri(mime_type="application/pdf", uri=uri)]
    
    output_text = generate_response(combined_prompt)
    output_pdf_uri, local_file_path = upload_output_to_gcs(output_text, output_folder)
    download_button = gr.DownloadButton(label=f"Download Report", value=local_file_path, visible=True)

    return output_text, output_pdf_uri, download_button

def download_file():
    # This function will return the path to be downloaded
    return [gr.DownloadButton(label="Download Report", visible=False)]

def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("### Enter your application description and upload PDF files")
        with gr.Row():
            user_prompt = gr.Textbox(label="Project Details", placeholder="Enter your project details here...")
            file_input = gr.UploadButton(label="Upload PDF Files", file_types=["pdf"], file_count="multiple")

        submit_button = gr.Button("Generate Response")
        download_button = gr.DownloadButton(label="Download Report", visible=False)

        output_markdown = gr.Markdown(label="Output")
        output_pdf_link = gr.Markdown()


        submit_button.click(fn=main_gradio, inputs=[user_prompt, file_input], outputs=[output_markdown, output_pdf_link, download_button])
        download_button.click(fn=download_file, inputs=[], outputs=[download_button])
    return demo



port = 8080
if __name__ == "__main__":
    interface = gradio_ui()
    interface.launch(server_name="0.0.0.0", server_port=port) 