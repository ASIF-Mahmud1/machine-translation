# Use an official Python runtime as a parent image
FROM python:3.9.6

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


# Set the working directory in the container
WORKDIR /app

# Copy the Pipfile and Pipfile.lock to the working directory
COPY Pipfile Pipfile.lock /app/

# Install pipenv
RUN pip install pipenv

# Install project dependencies
RUN pipenv install --system --deploy

# Copy the rest of the application code to the working directory
COPY . /app/

# Keep the container running by running a long-running process
CMD ["bash", "-c", "while true; do sleep 3600; done"]
