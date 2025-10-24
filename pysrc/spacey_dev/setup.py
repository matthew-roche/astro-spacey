from transformers import AutoModelForQuestionAnswering, AutoModel, AutoModelForCausalLM, AutoTokenizer
import json, os, requests, zipfile, argparse
from spacey_util.add_path import model_list_file_path, model_path, data_list_path, data_raw_path, data_post_process
from datasets import load_dataset

DATA_CHUNK_SIZE = 8000
HF_TOKEN=os.getenv('HF_TOKEN') # hugging face read token

def download(deployment_only: bool = False):
    # download the models needed for the project, source hugging face
    with open(model_list_file_path(), "r", encoding="utf-8") as f:
        model_list = json.load(f)
        for model in model_list['models']:
            model_name = model['repo'].rstrip()

            model_for_deploy = bool(model['deployment'] == "True")
            if deployment_only and not model_for_deploy:
                print("Skipping model: ", model_name)
                continue # skip based on mode

            model_name_sanitize = model_name.replace("/","-")
            model_save = model_path() / model_name_sanitize

            print("Downloading model: ", model_name)

            if model['type'] == "AutoModelForQuestionAnswering":
                hf_model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=HF_TOKEN)
            elif model['type'] == "AutoModel":
                hf_model = AutoModel.from_pretrained(model_name, token=HF_TOKEN)
            elif model['type'] == "AutoModelForCausalLM":
                hf_model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
            else:
                print("Type not handled: ", model['type'])

            # download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

            hf_model.save_pretrained(model_save)
            tokenizer.save_pretrained(model_save)

    print(f"Downloaded the models to {model_path()}")

    # download the datasets
    with open(data_list_path(), "r", encoding="utf-8") as f:
        ds_list = json.load(f)

        for ds in ds_list['datasets']:
            repo_name = ds['repo']
            source = ds['source']
            repo_save_name = ds['name']

            print("Downloading Dataset: ", repo_name)

            if source == "HF":
                ds = load_dataset(repo_name)
                ds.save_to_disk(data_raw_path() / repo_save_name)
            elif source == "Kaggle":
                repo_url = ds['url']
                response = requests.get(repo_url, stream=True) # handle large datasets with mem efficiency
                response.raise_for_status()
                # save locally
                with open(data_raw_path() / repo_save_name, "wb") as f:
                    for chunk in response.iter_content(chunk_size=DATA_CHUNK_SIZE):
                        f.write(chunk)

                # kaggle spacenews is zipped
                with zipfile.ZipFile(data_raw_path() / repo_save_name, "r") as zip_ref:
                    zip_ref.extractall(data_raw_path())
            else:
                print("Source not handled: ", source)

    print(f"Downloaded the datasets to {data_raw_path()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-only", required=False, type=bool, default=False)
    args = parser.parse_args()

    if args.deploy_only:
        print(f"Setting up for deployment only..!")
    else:
        print(f"Setting up entire project...")

    download(args.deploy_only)