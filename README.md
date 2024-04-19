# Grammar-Controlled Educational Dialog Generation
This repository contains the code and some data associated with my Master's Thesis about grammar-controlled dialog generation for pedagogical purposes.

## Structure
Experiments are ordered chronologically in the respective directory. Their purpose is to document the mental process that lead to the scripts in the folder /scripts that are suitable to reproduce essential part of the work.

## Requirements
This project is based on Python 3.11.2 and a collection of common libraries that you can find in `requirements.txt`. It is theoretically possible to run it on CPU only but the performance greatly benefits from training and running models on GPU. For accessing the OpenAI API, you'll need an API key.

## Setup
1. Create a copy of the .env.example file and fill in your configuration values.
2. Create a virtual environment and use pip to install the requirements.
3. You can run the experiments in a Jupyter notebook with a kernel within your virtual environment with the working directory being the same directory.
4. You can run the scripts with your environment activated with the working directory being the same directory.