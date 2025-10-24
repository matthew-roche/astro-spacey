import sys, os
from pathlib import Path

def project_path():
    script_dir = Path(__file__).resolve()
    project_dir = script_dir.parents[2]
    return project_dir

def add_project_path():
    """
    add proj dir ro path for import file resolution
    """
    project_dir = project_path()
    sys.path.append(str(project_dir))

def data_path():
    project_dir = project_path()
    return project_dir / "data"

def data_raw_path():
    project_dir = project_path()
    return project_dir / "data" / "raw"

def data_processed_path():
    project_dir = project_path()
    path_ = project_dir / "data" / "processed"
    if not os.path.exists(path_):
        os.makedirs(path_)
    return path_

def data_post_process():
    project_dir = project_path()
    path_ = project_dir / "data" / "out"
    if not os.path.exists(path_):
        os.makedirs(path_)
    return path_

def report_path():
    project_dir = project_path()
    path_ = project_dir / "reports"
    if not os.path.exists(path_):
        os.makedirs(path_)
    return path_

def model_path():
    project_dir = project_path()
    return project_dir / "models"

def model_output_path():
    project_dir = project_path()
    return project_dir / "finetuned_models"

def fine_tuned_model_path():
    project_dir = project_path()
    return project_dir / "finetuned_models"

def onnx_model_path():
    project_dir = project_path()
    path_ = project_dir / "onnx"
    if not os.path.exists(path_):
        os.makedirs(path_)
    return path_

def trt_model_path():
    project_dir = project_path()
    path_ = project_dir / "tensorrt"
    if not os.path.exists(path_):
        os.makedirs(path_)
    return path_

def model_list_file_path():
    project_dir = project_path()
    return project_dir / "models" / "models.json"

def data_list_path():
    project_dir = project_path()
    return project_dir / "data" / "datasets.json"