# Controlling Grammar in Dialogue Response Generation for Language Learning
This repository contains the code and data associated with my Master's Thesis about controlling grammar in dialogue response generation for language learning. The goal is to adapt large language models to use grammar from the English Grammar Profile in responses to preceding dialogues. The experiments also revolve around robust grammar detection and simulating how learners may respond to grammar-controlled chatbot responses.

## Structure
Experiments are ordered chronologically in the respective directory and contain explaining headings and comments. Their purpose is to document the mental process that led to the scripts in the folder `/scripts` that are designed to reproduce essential parts of the work. You may use the file /scripts/run_script.sh to configure batch jobs. `/data` contains input such as the dialog data and some generated data as part of the experiments. `/source` contains code that is used on multiple occasions. `/results` contains plots from the experiments. `/models` exists to contain model checkpoints that are creating while running experiments and scripts. You can find a description of a directory's content in their respective README files.

## Requirements
This project is based on Python 3.11.2 and a collection of common libraries such as Pytorch, Pandas, scikit-learn and seaborn. It is theoretically possible to run it on CPU only but the performance greatly benefits from training and running models on GPU. It is recommended to have at least 2 GPU cores with a memory of 20GB each. For accessing the OpenAI API, you have to obtain an API key and configure it (see below). If you want to use the [EFCAMDat](https://www.lingref.com/cpp/slrf/2012/paper3100.pdf) corpus in experiment 004, you need to obtain the corresponding file from the authors.

## Setup
1. Create a copy of the `.env.example` file named `.env` and fill in your configuration values.
2. Create a virtual environment and use pip to install the requirements from `requirements.txt`.
3. You can run the experiments in a Jupyter notebook with a kernel within your virtual environment with the working directory being the same directory.
4. You can run the scripts with your environment activated with the working directory being the same directory.