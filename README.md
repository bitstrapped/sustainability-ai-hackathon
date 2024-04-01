# Local Setup Guide for Kora
This repository contains the code for setting up and running Kora locally.

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

1. Create a bucket in Google Cloud Storage following the guidelines below:

![Bucket naming convention](/assets/one.png)

> Note the bucket must be name as ```{project-id}.me.embeddings```

2. Inside the ```{project-id}.me.embeddings bucket```, create a folder named ```init_index```. Upload the ```embeddings_0.json``` file located in the Kora folder to this init_index folder.


2.  Create a second bucket in Google Cloud Storage for storing your files : 


![Bucket naming convention](/assets/two.png)

> Note the bucket must be name as ```{project-id}.training.docs```

**Update Project Details in the Code**

3. Open the ```KORA > main.py``` file and modify the project ID and region in the following code:

```
kora = KoraAI("Project id", "Region")

```
Replace "Project id" and "Region" with your GCP project ID and region details.


### Running the project locally 

OOnce you've completed the above steps, you can run the project locally using the following command:


``` uvicorn main:app --host 0.0.0.0 --reload ```

After running the command, access Kora at ```localhost:8000```.

You're all set up! Enjoy using Kora locally.


