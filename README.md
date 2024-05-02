# Sustainability AI hackathon
This repository contains the code for the AI hackathon topic -"Give a less CO2 emission based gcp solution"

### Setting Up the Virtual Environment

1. Begin by installing the virtual environment package by running:

```
pip install virtualenv
```
2. Create a Python virtual environment at the root directory of the project:

``` 
python<version> -m venv <virtual-environment-name> (e.g python3.8 -m venv env) 
```

3. Activate the virtual environment:

```
source <virtual-environment-name>/bin/activate
```

4. Install required libraries:

```
pip install -r requirements.txt
```

### Google Cloud Platform (GCP) Setup

1. Project name: ds-ml-pod
2. Bucket name: sustainable-ai
    2.1. Inside the ```bucket```, there are 2 folders: <br>
    First is   ```documents``` which contain the training documents 
    Second is ```pdf files```  which are the pdf files being uploaded dynamically through the UI. 



### Running the project locally 

Once you've completed the above steps, you can run the project locally using the following command:

python abhishek_main.py
