import tensorrt as trt
import pycuda.driver as cuda_driver
import numpy as np
import time
# import pycuda.autoinit

def init_cuda_context():
    # Initialize CUDA
    cuda_driver.init()
    from pycuda.tools import make_default_context
    global active_context

    active_context = make_default_context()
    device = active_context.get_device()

    import atexit
    def _finish_up():
        global active_context
        active_context.pop()
        active_context = None

        from pycuda.tools import clear_context_caches

        clear_context_caches()
    atexit.register(_finish_up)

# init_cuda_context()



def engine_inference(engine, inputs_data, batch_size, seq_len, classify:bool=True):
    context = engine.create_execution_context()
    context.set_input_shape("input_ids", (batch_size, seq_len))
    context.set_input_shape("attention_mask", (batch_size, seq_len))

    # prepare mem alloc
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_driver.Stream()
    for binding in engine:
        #print(f"Binding: {binding}, Is Input: {engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT}")

        shape = context.get_tensor_shape(binding)  # Updated API
        dtype = trt.nptype(engine.get_tensor_dtype(binding))

        #print(shape)

        size = trt.volume(shape)  # Calculate total elements in the shape
        host_mem = cuda_driver.pagelocked_empty(size, dtype)  # Host memory
        device_mem = cuda_driver.mem_alloc(host_mem.nbytes)   # Device memory

        bindings.append(int(device_mem))

        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})

    np.copyto(inputs[0]["host"], inputs_data["input_ids"].flatten())
    np.copyto(inputs[1]["host"], inputs_data["attention_mask"].flatten())

    cuda_driver.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    cuda_driver.memcpy_htod_async(inputs[1]["device"], inputs[1]["host"], stream)

    for i, binding in enumerate(engine):
        context.set_tensor_address(binding, bindings[i])
        # print(f"Tensor Name: {binding}, Address Set: {bindings[i]}")

    try:
        context.execute_async_v3(stream_handle=stream.handle)
        
    except Exception as e:
        print(f"Execution Error: {e}")

    # Transfer predictions back to host
    for output in outputs:
        cuda_driver.memcpy_dtoh_async(output["host"], output["device"], stream)
    
    stream.synchronize()

    if classify:
        start_logits = outputs[0]['host']
        end_logits = outputs[1]["host"]

        start_index = np.argmax(start_logits)
        end_index = np.argmax(end_logits)

        return start_index, end_index
    
    else:
        last_hidden_state = outputs[0]["host"]
        pooler_output = outputs[1]["host"]
        attention_mask = outputs[2]["host"]

        return last_hidden_state, pooler_output, attention_mask