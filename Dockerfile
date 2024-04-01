# Use the official Python 3.8.16 image as the base image
FROM python:3.8.16

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
#uninstall pip uninstall pdfminer
RUN pip uninstall pdfminer
RUN pip install pdfminer.six
# Install the Python dependencies
RUN pip install -r requirements.txt
RUN pip install --upgrade pdfminer.six
# Copy the rest of the application code into the container
COPY . .

# Command to run when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]
