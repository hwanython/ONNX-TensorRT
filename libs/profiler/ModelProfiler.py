import torch
import time


# 데이터 타입 크기 확인 함수 (바이트 단위)
def get_dtype_size(dtype):
    if dtype == torch.float32:
        return 4
    elif dtype == torch.float64:
        return 8
    elif dtype == torch.int32:
        return 4
    elif dtype == torch.int64:
        return 8
    else:
        raise ValueError("Unsupported data type")


# 모델 매개변수 메모리 계산
def calculate_model_memory(model):
    total_params = 0
    total_memory = 0

    for param in model.parameters():
        param_size = param.numel()  # 매개변수의 총 요소 수
        param_memory = param_size * get_dtype_size(param.dtype)  # 메모리 사용량 (바이트)
        total_params += param_size
        total_memory += param_memory

    return total_params, total_memory


# 중간 텐서 메모리 계산 (예제 입력 데이터 사용)
def calculate_tensor_memory(input_data, model):
    total_memory = 0
    hooks = []

    def hook_fn(module, input, output):
        nonlocal total_memory
        if isinstance(input, tuple):
            for i in input:
                if isinstance(i, torch.Tensor):
                    total_memory += i.numel() * get_dtype_size(i.dtype)
        elif isinstance(input, torch.Tensor):
            total_memory += input.numel() * get_dtype_size(input.dtype)

        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    total_memory += o.numel() * get_dtype_size(o.dtype)
        elif isinstance(output, torch.Tensor):
            total_memory += output.numel() * get_dtype_size(output.dtype)

    # 모든 레이어에 hook 추가
    for layer in model.children():
        hooks.append(layer.register_forward_hook(hook_fn))

    t = time.time()
    # 모델 예측
    with torch.no_grad():
        _ = model(input_data)
    pred = time.time() - t

    # hook 제거
    for hook in hooks:
        hook.remove()

    return total_memory, pred


def monitoring(model, input_shape, device='cpu', repeats=5):
    # 모델 메모리 계산 및 출력
    total_params, model_memory = calculate_model_memory(model)
    print(f"Total Parameters: {total_params}")
    print(f"Model Parameters Memory Usage: {model_memory / 1024 ** 2:.2f} MB")

    # 입력 데이터 생성
    input_data = torch.randn(*input_shape).to(device)

    # N회 반복하여 중간 텐서 메모리 계산
    tensor_memory_list = []
    preds = []

    for _ in range(repeats):
        tensor_memory, pred = calculate_tensor_memory(input_data, model)
        tensor_memory_list.append(tensor_memory)
        preds.append(pred)

    # 평균 메모리 사용량 계산
    average_tensor_memory = sum(tensor_memory_list) / len(tensor_memory_list)
    average_time = sum(preds) / len(preds)

    # 중간 텐서 메모리 계산 및 출력
    print(f"Average Intermediate Tensors Memory Usage: {average_tensor_memory / 1024 ** 2:.2f} MB")

    # 총 메모리 사용량 출력
    total_memory = model_memory + average_tensor_memory
    print(f"Average Total Memory Usage: {total_memory / 1024 ** 2:.2f} MB")
    print(f"Average Time: {average_time:.2f}")
    
    f_total_memory = total_memory / 1024 ** 2
    f_average_time = average_time

    return f_total_memory, f_average_time