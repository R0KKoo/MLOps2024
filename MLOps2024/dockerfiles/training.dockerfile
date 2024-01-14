## Base image
FROM python:3.11-slim

## Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

## Copying important documents for the image
COPY MLOps2024/requirements.txt MLOps2024/requirements.txt
COPY MLOps2024/pyproject.toml MLOps2024/pyproject.toml
COPY MLOps2024/ MLOps2024/
COPY MLOps2024/data/ MLOps2024/data/

## Setting the working directory in our container and add commands that install the dependencies

# We split the the installation into two steps, such that docker can cache our project dependencies separately from our application code.
# This means that if we change our application code, we do not need to reinstall all the dependencies.
# This is a common strategy for docker images.

# As an alternative you can use RUN make requirements if you have a Makefile that installs the dependencies.
# Just remember to also copy over the Makefile into the docker image.
WORKDIR /MLOps2024/
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

## Finally, we are going to name our training script as the entrypoint for our docker image.
## The entrypoint is the application that we want to run when the image is being executed:

# The "u" here makes sure that any output from our script e.g. any print(...) statements gets redirected to our terminal.
ENTRYPOINT ["python", "-u", "CorruptedMNIST_Classification/train_model.py"]
