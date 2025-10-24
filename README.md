# AI QA (Question Answering) with NLP Transformers

Scalable lightweight system that answers to questions from the inference corpus. Accelerated performance with lightweight encoder transformers.

The way this system functions is,   
1. RoBERTa base squad2 checkpoint from Huggingface is finetuned on Natural Question (short version) for real-world diverse QA adaptation.   
2. Then the zero-shot model is finetuned on NASA SMD training dataset towards domain adaptation.
3. Fused retriever with BM25 and SimCSE (FAISS) inspired by RAG.   

Benefits of this approach:   
1. Fast adaptation to smaller domain datasets while having the zero-shot baseline answerable capability.    
2. Preprocessing steps to handle noisy unknown dataset to create a quality inference corpus. 
3. Includes UMAP + unsupervised HDBSCAN clustering, and coarse filtering with local inference on Meta LLaMA 3.1 8B (can we swapped as needed).   
4. Deployable backend with Flask API and frontend with React + tailwind.     
5. TensorRT optimization steps for Encoder Transformers.   

Refer [System Architecture](https://github.com/matthew-roche/astro-spacey/blob/main/docs/system-arch.png) for more details on how the infernce works.   

## Getting started guide
Developed on Python version [3.12.5](https://www.python.org/downloads/release/python-3125/)   

Packages used: [PyTorch v2.8 + cuda 12.9](https://github.com/pytorch/pytorch/releases/tag/v2.8.0), [matplotlib](https://pypi.org/project/matplotlib/), [Huggingface transformers v4.57.1](https://pypi.org/project/transformers/4.57.1/), [Huggingface datasets](https://pypi.org/project/datasets/), [Flask](https://pypi.org/project/Flask/), [Flask-smorest for swagger-ui](https://flask-smorest.readthedocs.io/en/latest/)   

### STEP 1
We recommended to use a virtual python environment, this can be done with;
```
python -m venv dl
```

### STEP 2
Then on windows, the virtual env can be activated by 
```
.\dl\Scripts\activate
python.exe -m pip install --upgrade pip
```   

### STEP 3
Afterwards, Install the packages from ```requirements.txt```, can be done using;
```
pip install -r <project-dir>/requirements.txt
```

Then change dir to project dir and run below to install this :
```
pip install -e .
```

### STEP 4
Then to locally download the models and datasets, acquire a read access token from your hugging face account and set it as an enviornment variable with:
```
set HF_TOKEN=yourtoken
```

#### For deployment only:   
We uploaded our finetuned nq_nasa_v1 model to hugging face [quantaRoche/roberta-finetuned-nq-nasa-qa](https://huggingface.co/quantaRoche/roberta-finetuned-nq-nasa-qa) therefore this is referenced in ```models/models.json``` and can be downloaded through the below method.
```
python pysrc/spacey_dev/setup.py --deploy-only=true
```

#### For new inference datasets and deployment:   
Note: To donwload the [Meta LLaMA 3.1 8B Instruct model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) prior permission should be requested from the linked source, this is a gated model.    
But this step is not required unless there are plans to redo the coarse filtering. Therefore in ```models/models.json``` remove the meta llama object to skip downloading it.    
This repository already provides the coarse filtering results in ```/reports``` which is sufficient for using the preprocessing pipeline with the datasets in this project.         

Then run the below step to download the models and datasets, this process can take time depending on the network speed.   
```
python pysrc/spacey_dev/setup.py
```

### STEP 5    
Run the preprocessor to reproduce the data needed for either deploymen or retraining, LLaMA ins't needed here.
```
python pysrc/spacey_dev/preprocessor/pipeline.py
```

### STEP 6
Check if inference is working and that the model answers, Run:
```
python pysrc/spacey_dev/inference/run.py
```

### STEP 7
Create an API_SEC to limit access to backend, use any kind of secret generator or use:
```
python -c "import secrets; print(secrets.token_hex(32));"
```
Then copy the value and set the API_SEC environmental variable which is read by the backend server.py:
```cmd
SET API_SEC=secretkey
```
And add this key to ```/frontend/.env/VITE_API_SEC```   


Then, deploy the backend locally and test the deployment with swagger UI ```http://127.0.0.1:5000/api/docs```
```cmd
python pysrc/spacey_api/server.py
```
Ensure ```/api/health``` returns healthy and ```/api/device``` returns ```cuda```, if it returns ```cpu``` then inference time would be slow. 

Recommendation is to deploy the backend in an [aws g6 instance](https://aws.amazon.com/ec2/instance-types/g6/), like g6f.2xlarge and ssh to check the status. But this can also be deployed on g4 or previous architectures.

We didn't build a bf16 model but we included the tensorrt optimization code which converts the torch model to onnx to trt with assertion. Feel free to check it out in ```/pysrc/spacey_dev/optimization/```   


### STEP 8
Frontend can be deployed anywhere needed, we did the deployment on vercel hobby plan and used their [service-side functions for proxy](https://github.com/matthew-roche/astro-spacey/tree/main/frontend/api). Below are the steps for vercel deployment, remember to sign up and create an account there.   
Then locally in the frontend root path, run below to install the packages:
```cmd
npm install
```
Then install [vercel cli](https://vercel.com/docs/cli?package-manager=npm) and run:
```cmd
vercel
```
Which will require account authentication and then deploys the preview version.

Because our backend was locally deployed, we used [cloudflared tunnel v10.0](https://github.com/cloudflare/cloudflared/releases/tag/2025.10.0) and a public exposed aws s3 general purpose bucket to dynamically set the backend url without redeploying the frontend.

To follow the same steps, In AWS General purpose S3 bucket, add a file with this json:
```json
{
    "BACKEND_BASE":"https://yourbackendurl"
}
```
And in frontend project, set the s3 json file url in .env for ```CONFIG_URL```. The frontend will cache this on initial load [/frontent/lib/config.ts](https://github.com/matthew-roche/astro-spacey/blob/main/frontend/lib/config.ts). 


## Our setup

Our local config is;
| Component     | Specification |
| ------------- | ------------- |
| CPU | i5 12600K |
| RAM | 32GB DDR4 |
| Storage | nvme ssd |
| GPU | 1x RTX 4080 super |
| GPU Architecture | Ada Lovelace |

**Task:** Question Answering   
**Language:** Engligh   
**Local setup:** 1x RTX 4080 super  
**Evaluation Metric:** Squad_v2   
**Evaluation Dataset:** [nasa smd qa validation split](https://huggingface.co/datasets/nasa-impact/nasa-smd-qa-benchmark)

### Hyperparameters

#### Finetune (1) Hyperparameters (NQ Dataset) zero-shot:
train_batch_size = 16   
val_batch_size = 8   
n_epochs = 2   
base_LM_model = "roberta-base"   
max_seq_len = 384   
doc_stride=128    
optimizer=adamW   
last_layer_learning_rate=5e-6   
qa_head_learning_rate=3e-5   
release_model= "roberta-finetuned-nq"   
gradient_checkpointing=True   

#### Finetune (2) Hyperparameters (Nasa train):
train_batch_size = 16   
val_batch_size = 8   
n_epochs = 5   
base_LM_model = "roberta-finetuned-nq"   
max_seq_len = 384   
doc_stride=128   
optimizer=adamW   
layer_learning_rate=1e-6   
qa_head_learning_rate=1e-5   
release_model= "roberta-finetuned-nq-nasa"   
gradient_checkpointing=True   

### Performance   
"exact": 66.0,   
"f1": 79.86357273703948,   
"total": 50,   
"HasAns_exact": 53.333333333333336,   
"HasAns_f1": 76.43928789506579,   
"HasAns_total": 30,   
"NoAns_exact": 85.0,   
"NoAns_f1": 85.0,   
"NoAns_total": 20   

## Contributors
| Name     | Area |
| ------------- | ------------- |
| Sugam | Unsupervised data clustering |
| Yuxuan | Evaluation Metric reviewer |
| Moe | Team communications |
| Kritika | Benchmark setup |

## Acknowledgements
- [hugging face deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
- [hugging face cjlovering Natural Questions (short) Dataset](https://huggingface.co/datasets/cjlovering/natural-questions-short)
- [hugging face NASA SMD Dataset](https://huggingface.co/datasets/nasa-impact/nasa-smd-qa-benchmark)
- [Huggingface Trainer Documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- Developed as a part of Westcliff MSCS AIT500 course 2025 Fall 1 session 2. We thank our professor Desmond for the continous guidance.
