# 🌿 Sustainability AI hackathon
This repository contains the codebase for the Google AI hackathon on the topic -"Sustainability AI". It aims to analyze documents and images related to software projects to assess potential improvements to reduce environmental impacts and carbon footprints. It generates comprehensive reports that highlight areas where CO2 emissions can be reduced and suggests actionable steps to enhance sustainability.

## 🌿 Code Files

- **main.py**: Contains the main codebase. 
    - It takes in the user description and uploaded files dynamically
    - Saves them temporarily on Google Cloud Storage
    - Calls the **Gemini 1.5 Pro Preview** model API
    - Passes the prompt and the files (including PDF or Image files) through GCS to Gemini API
    - Generates the output solution

- **Dockerfile**: To containerize the code.
    - Exposed port is 8080

### 🌿 Running the project locally 

- Create a virtual env and install python=3.12.0
- Install required libraries:
```
pip install -r requirements.txt
```
- Once you've completed the above steps, you can run the project locally using the following command:
```
python main.py
```
- Click on the server link generated in the terminal which will redirect you to the UI

### 🌿 Remote Endpoint

Interact with the Gradio UI app which is deployed on Google Cloud Run: https://sustainability-ai-o3no33v6yq-uc.a.run.app