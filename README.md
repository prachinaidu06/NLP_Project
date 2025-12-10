# NLP_Project

This project fine-tunes LLaMA-3.2-1B-Instruct using LoRA to build a production-ready AI system that can generate concise email summaries and write professional email replies based on user instructions

The repository contains everything needed to:
--Prepare training data
--Train a LoRA-based instruction-tuned model
--Evaluate it using ROUGE and BERTScore
--Deploy it using a Streamlit web application over SSH

System Requirements
--Python 3.9 or higher
--CUDA-enabled GPU recommended
--Linux or macOS
--At least 16 GB RAM
--Hugging Face account for model downloads

Create and activate a virtual environment

Install all dependencies from the provided requirements file:
  pip install -r requirements.txt

To run reply.py login to OPEN API and for training Hugging Face API is used 

After logging into OPEN API 
  run reply.py 

Dataset Preparation
  python merge.py

This combines:
--Original email threads
--Human summaries
--GPT-generated synthetic replies

TRAINING
To fine-tune the LLaMA model
  python train.py

MODEL EVALUATION 
  python evaluation.py

INFERENCE
  python inference.py

STREAMLIT APP
  streamlit run app.py

Run the App on a Remote Server via SSH:

  streamlit run app.py
Create a reverse tunnel
  ssh -N -R 8501:localhost:8501 pnaidu2@hopper2.orc.gmu.edu
Then on your local machine:
  ssh -N -L 8501:localhost:8501 pnaidu2@hopper2.orc.gmu.edu
After this open the url:
  http://localhost:8501


