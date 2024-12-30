## Explain Extended New Year 2025: GPT in 500 lines of SQL

https://explainextended.com/2023/12/31/happy-new-year-15/

### About

This repository contains the code for the article [Explain Extended New Year 2025: Happy New Year: Diffusion Model image generator in about 700 lines of pure SQL](https://explainextended.com/2024/12/31/happy-new-year-16/).

The SQL code described in the article runs on PostgreSQL 17.2.

To populate the tables, you need to run the script `populate.py`. This script will download the model files from HuggingFace, create database tables, and populate them with the model weights and biases.

### Requirements

* Docker
* Python 3.11 or later

### Installation

1. Build and run the database image:

   ```
   cd docker
   docker compose up -d
   ```

1. Install Python dependencies:

   ```
   pip install -r requirements.txt
   ```

1. Populate the tables:

   ```
   python populate.py --connection_string="postgresql://explainextended2025:MySecretPassword@localhost:5432/explainextended2025"
   ```

   Consider changing the password if your machine is exposed to the Internet.

1. The scripts mentioned in the article are located in the [scripts](./scripts) directory.

### Notebook

The [Jupyter notebook](./butterflies.ipynb) contains an implementation of Denoising Diffusion Probabilistic Model in about 130 lines of PyTorch code. It was used to verify the correctness of the SQL code.

**Enjoy!**