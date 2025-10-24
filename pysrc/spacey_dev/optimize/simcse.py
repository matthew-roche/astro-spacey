import torch, numpy as np, time
from torch import nn
import onnxruntime as ort
from spacey_util.add_path import model_path, onnx_model_path, trt_model_path
from spacey_dev.util.helper import assert_embedding_close
from transformers import AutoModel, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import tensorrt as trt

EXPORT_ONNX = False
VALIDATE_ONNX = True
EXPORT_TRT = False
VALIDATE_TRT = True

TRT_OPTIMIZE_LEVEL = 5 
OPSET_VERSION = 20 # ONNX opset version compatible with TensorRT

class SimCSEWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=False, return_dict=True)
        last = out.last_hidden_state                      # [B, T, H]
        pool = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None \
               else torch.zeros(last.size(0), last.size(-1), device=last.device, dtype=last.dtype)
        return last, pool, attention_mask

@torch.no_grad()
def encode_query(model, tokenizer, q: str, max_len=128):
    t = tokenizer(q, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
    out = model(**t, output_hidden_states=True, return_dict=True)
    if out.pooler_output is not None:
        v = out.pooler_output
    else:
        last = out.last_hidden_state
        mask = t["attention_mask"].unsqueeze(-1)
        v = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
    v = torch.nn.functional.normalize(v, p=2, dim=1)
    return v.cpu().numpy().astype("float32")  # [1, D]

def export_onnx(model, inputs, save_onnx_dir):
    torch.onnx.export(
        model,  # PyTorch model
        (inputs["input_ids"], inputs["attention_mask"]),  # Model inputs
        save_onnx_dir,  # Output ONNX file
        input_names=["input_ids", "attention_mask"],  # Input layer names
        output_names=["last_hidden_state", "pooler_output", "attention_mask_out"],  # Output layer names
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch", 1: "seq"},
            "pooler_output": {0: "batch"},
            "attention_mask_out": {0: "batch", 1: "seq"},
        },  # Dynamic axes for flexible input sizes
        opset_version=OPSET_VERSION,  
    )

def onnx_inference(ort_session, inputs):
    onnx_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy()
    }

    outputs = ort_session.run(None, onnx_inputs)

    last_hidden_state = outputs[0]
    pooler_output = outputs[1]
    attention_mask = outputs[2]

    normalized = pooler_output / np.linalg.norm(pooler_output, ord=2, axis=1, keepdims=True)
    return  normalized.astype(np.float32)


if __name__ == "__main__":
    saved_m_dir = model_path() / "princeton-nlp-sup-simcse-roberta-large"
    saved_t_dir = model_path() / "princeton-nlp-sup-simcse-roberta-large-token"
    save_onnx_dir = onnx_model_path() / "princeton-nlp-sup-simcse-roberta-lg.onnx"
    save_engine_dir = trt_model_path() /  f"./princeton-nlp-sup-simcse-roberta-lg-{TRT_OPTIMIZE_LEVEL}.trt"

    transformer_model = AutoModel.from_pretrained(saved_m_dir)
    wrapped_model = SimCSEWrapper(transformer_model).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(saved_t_dir)

    context_list = []

    question = "What's on Mercury craters ?"

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # print(len(inputs['input_ids'][0]))
    # exit(0)

    #  step 1 inference on model
    last, pool, attention_mask = wrapped_model(**inputs)
    wrapped_embeddings = torch.nn.functional.normalize(pool, p=2, dim=1).detach().cpu().numpy().astype("float32")

    start_time = time.perf_counter()
    expected_embeddings = encode_query(transformer_model, tokenizer, question, 512)
    end_time = time.perf_counter()
    print(f"Torch Inference time: {end_time - start_time:.6f} seconds")

    print("Expected Embeddings: ", expected_embeddings)

    assert wrapped_embeddings[0][0] == expected_embeddings[0][0]
    
    # step 2
    if EXPORT_ONNX:
        export_onnx(wrapped_model, inputs, save_onnx_dir)
    
    transformer_model = None # unalloc mem
    wrapped_model = None


    # step 3 - onnx
    if VALIDATE_ONNX:
        ort_session = ort.InferenceSession(save_onnx_dir)

        start_time = time.perf_counter()
        onnx_embeddings = onnx_inference(ort_session, inputs)

        end_time = time.perf_counter()
        print(f"Onnx Inference time: {end_time - start_time:.6f} seconds")

        assert_embedding_close(expected_embeddings, onnx_embeddings)

        # ensure export was successful
        # assert expected_embeddings[0][0] == onnx_embeddings[0][0] 
        print("Validated Onnx!")
    
    # step 4 - tensor rt engine
    if EXPORT_TRT:
        # TensorRT Logger
        logger = trt.Logger(trt.Logger.WARNING)

        # TensorRT Engine Builder
        with trt.Builder(logger) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, logger) as parser:
            # Load ONNX model
            with open(save_onnx_dir, "rb") as model_file:
                parser.parse(model_file.read())
            
            
            # Configure the builder
            config = builder.create_builder_config()
            # config.flags |= trt.BuilderFlag.CUDA_GRAPH
            # Set memory pool limits
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB workspace memory
            config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY, 1 << 28)  # 256 MB for tactic memory
            #config.flags = trt.BuilderFlag.FP16  # Enable FP16 precision
            config.builder_optimization_level = TRT_OPTIMIZE_LEVEL  # Set optimization level (0-5)

            # Ensure FP16 is disabled
            config.clear_flag(trt.BuilderFlag.FP16)

            # Ensure INT8 is disabled
            config.clear_flag(trt.BuilderFlag.INT8)

            profile = builder.create_optimization_profile()

            # Define input shapes (name must match the ONNX input names)
            profile.set_shape("input_ids", min=(1, 1), opt=(1, 128), max=(8, 512))  # Batch x Sequence
            profile.set_shape("attention_mask", min=(1, 1), opt=(1, 128), max=(8, 512))
            config.add_optimization_profile(profile)

            # Build the engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build the TensorRT engine")

            # Save the engine
            with open(save_engine_dir, "wb") as f:
                f.write(serialized_engine)

        print(f"Engine saved to {save_engine_dir}")

    if VALIDATE_TRT:
        from spacey.optimize.tensorrt_inference import engine_inference, init_cuda_context
        init_cuda_context()
        
        with open(save_engine_dir, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        inputs = tokenizer(
            question,
            return_tensors="np", 
            truncation=True,
            max_length=512
        ).to(device)
        
        seq_len = inputs["input_ids"].shape[1]
        
        start_time = time.perf_counter()
        last, pool, atten = engine_inference(engine, inputs, batch_size=1, seq_len=seq_len, classify=False)
        end_time = time.perf_counter()
        print(f"TensorRT Inference time: {end_time - start_time:.6f} seconds")

        print(pool)

        normalized = [pool] / np.linalg.norm([pool], ord=2, axis=1, keepdims=True)
        normalized.astype(np.float32)


        assert_embedding_close(expected_embeddings, normalized)

        # print("TensorRT Engine Answer: ", answer)

        # assert answer == expected_answer

        print("Validated Tensorrt!")

# Result: not worth the gain