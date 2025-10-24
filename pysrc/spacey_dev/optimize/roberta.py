import torch, numpy as np, time
import onnxruntime as ort
from spacey_util.add_path import model_path, onnx_model_path, trt_model_path
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import tensorrt as trt

EXPORT_ONNX = True
VALIDATE_ONNX = True
EXPORT_TRT = True
VALIDATE_TRT = True

TRT_OPTIMIZE_LEVEL = 5 
OPSET_VERSION = 20 # ONNX opset version compatible with TensorRT

def export_onnx(model, inputs, save_onnx_dir):
    torch.onnx.export(
        model,  # PyTorch model
        (inputs["input_ids"], inputs["attention_mask"]),  # Model inputs
        save_onnx_dir,  # Output ONNX file
        input_names=["input_ids", "attention_mask"],  # Input layer names
        output_names=["output"],  # Output layer name
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"}
        },  # Dynamic axes for flexible input sizes
        opset_version=OPSET_VERSION,  
        external_data=False
    )

def onnx_inference(ort_session, inputs, tokenizer):
    onnx_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy()
    }

    outputs = ort_session.run(None, onnx_inputs)

    start_logits = outputs[0]
    end_logits = outputs[1]
    start_index = np.argmax(start_logits, axis=1)[0]
    end_index = np.argmax(end_logits, axis=1)[0]

    return tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1])


if __name__ == "__main__":
    # saved_m_dir = fine_tuned_model_path() / "roberta-base-squad2-nq-nasa/checkpoint-90"
    # saved_t_dir = fine_tuned_model_path() / "roberta-base-squad2-nq-nasa/checkpoint-90"
    roberta_nq_nasa = model_path() / "quantaRoche-roberta-finetuned-nq-nasa-qa"
    save_onnx_dir = onnx_model_path() / "roberta-base-squad2-nq-nasa-cp90.onnx"
    save_engine_dir = trt_model_path() /  f"./roberta-base-squad2-nq-nasa-cp90-{TRT_OPTIMIZE_LEVEL}.trt"

    transformer_model = AutoModelForQuestionAnswering.from_pretrained(roberta_nq_nasa)
    tokenizer = AutoTokenizer.from_pretrained(roberta_nq_nasa)

    context_list = [
        "The spacecraft carries a suite of instruments designed to study the planet's upper atmosphere to better understand how the planet's atmosphere and climate have changed over time.",
        "The ozone layer on Venus sits 100 kilometers above the planet's surface, which is roughly four times higher in the atmosphere compared with Earth, and is also a hundred to a thousand times less dense.",
        "MAVEN's goal is to explore Mars' upper atmosphere, sorting out what role the escape of gas from the atmosphere to space has played in altering the planet's climate throughout its history.",
        "While the planet is known for its hot, dense atmosphere that contains sulfuric acid, conditions are more hospitable in the upper atmosphere where temperatures and pressures are lower.",
        "One possibility for the phosphine found in Venus's atmosphere is that life in the upper atmosphere may be creating it, but scientists involved with the discovery said there was no definitive proof yet of life there.",
        "SAM can detect organics in martian soil, and it will sniff the red planet's atmosphere for methane, which may be a sign of life as organisms here on Earth are known to generate the gas.",
        '"An SEP event like this typically occurs every couple weeks. Once all the instruments are turned on, we expect to also be able to track the response of the upper atmosphere to them."',
        "Volcanoes spew gases that can change the composition of the atmosphere and affect the amount of sunlight that hits a planet's surface, she said.",
        "Ground controllers are now moving MAVEN into its lower, science orbit in order to take more observations of the planet's upper atmosphere and find out how some of it might be escaping into outer space.",
        "While they work well for predicting where a Mars explorer should land, they fail to accurately describe the complex atmosphere on the planet, Chevrier said.",
        "WASHINGTON — NASA's Mars Atmosphere and Volatile Evolution (MAVEN) spacecraft, which began orbiting Mars in late September to probe the planet's thin atmosphere and help scientists understand what caused the planet to change from a warm, wet world to the cold and dry one it is today, has already beamed back some important new data."
    ]

    QA_input = {
        'question': "Which planet has a higher and upper atmosphere ?",
        'context': " ".join(context_list)
    }
    inputs = tokenizer(
        QA_input['question'],
        QA_input['context'], 
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # print(len(inputs['input_ids'][0]))
    # exit(0)

    #  step 1 inference on model
    transformer_model.to(device)

    start_time = time.perf_counter()
    outputs = transformer_model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    end_time = time.perf_counter()
    print(f"Torch Inference time: {end_time - start_time:.6f} seconds")

    expected_answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1])
    print("Expected Answer:", expected_answer)
    
    # step 2
    if EXPORT_ONNX:
        export_onnx(transformer_model, inputs, save_onnx_dir)
    
    transformer_model = None # unalloc mem


    # step 3 - onnx
    if VALIDATE_ONNX:
        ort_session = ort.InferenceSession(save_onnx_dir)

        start_time = time.perf_counter()
        onnx_answer = onnx_inference(ort_session, inputs, tokenizer)
        end_time = time.perf_counter()
        print(f"Onnx Inference time: {end_time - start_time:.6f} seconds")

        print("Onnx Answer: ", onnx_answer)

        # ensure export was successful
        assert expected_answer == onnx_answer 
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
            profile.set_shape("input_ids", min=(1, 32), opt=(1, 128), max=(8, 512))  # Batch x Sequence
            profile.set_shape("attention_mask", min=(1, 32), opt=(1, 128), max=(8, 512))
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
        from spacey_dev.optimize.tensorrt_inference import engine_inference, init_cuda_context
        init_cuda_context()
        
        with open(save_engine_dir, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        inputs = tokenizer(
            QA_input['question'],
            QA_input['context'], 
            return_tensors="np", 
            truncation=True,
            max_length=512
        ).to(device)
        
        seq_len = inputs["input_ids"].shape[1]
        
        start_time = time.perf_counter()
        start_index, end_index = engine_inference(engine, inputs, batch_size=1, seq_len=seq_len)
        end_time = time.perf_counter()
        print(f"TensorRT Inference time: {end_time - start_time:.6f} seconds")

        answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1])

        print("TensorRT Engine Answer: ", answer)

        assert answer == expected_answer

        print("Validated Tensorrt!")

