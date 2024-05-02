# Use the official Python 3.12.0 image as the base image
FROM python:3.12.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run main.py when the container launches
CMD ["python", "./main.py"]