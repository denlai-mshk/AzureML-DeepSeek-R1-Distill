# 🚀 Step-by-Step Guide to Deploying DeepSeek Distill Model on Azure ML

## 1. Background: Why Azure ML Over Azure VM?

When deploying machine learning models, especially large language models like DeepSeek Distill, choosing the right infrastructure is crucial. While Azure Virtual Machines (VMs) offer flexibility, **Azure Machine Learning (Azure ML)** provides a **managed, scalable, and production-ready environment** for ML workflows.

### Benefits of Azure ML:
- **Managed Endpoints**: Easily deploy and manage models with autoscaling and traffic splitting.
- **Experiment Tracking**: Built-in tools for tracking runs, metrics, and artifacts.
- **Integrated MLOps**: Seamless integration with CI/CD pipelines.
- **Cost Efficiency**: Pay only for what you use with managed compute and storage.

## 2. Requirements: Request GPU Quota on Azure ML
Before deploying, ensure your Azure ML workspace has access to GPU resources:

1. Go to your Azure ML workspace in the Azure Portal.
2. Navigate to **Usage + quotas**.
3. Request a quota increase for GPU-enabled compute (e.g., `Standard_NC4as_T4_v3`).
4. Wait for approval (usually within 24 hours).

## 3. High-Level Workflow

### Step 1: Start a Compute Instance
- Launch a **CPU-based compute instance** (e.g., 4 vCPUs, 16 GB RAM).
- This instance is only for running the notebook, not for inference.

### Step 2: Import Notebook and Assets

- Upload your deployment notebook and any required asset files (e.g., `env.yaml`, `score.py`, `deployment_config.json`).
- Modify the notebook parameters:
  - `subscription_id`
  - `resource_group`
  - `workspace_name`
  - `model_name`
  - `endpoint_name`

## 4. Execute the Notebook

The notebook will guide you through the following steps:

### 4.1 Download the Model from Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-instruct")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-instruct")
```

### 4.2 Register the Model in Azure ML

```python
from azure.ai.ml.entities import Model

registered_model = Model(
    path="./deepseek_model",
    name="deepseek-7b-distill",
    description="DeepSeek 7B Distill Model",
    type="custom_model"
)
ml_client.models.create_or_update(registered_model)
```

### 4.3 Set Up Managed Endpoint and Deployment
```python
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

endpoint = ManagedOnlineEndpoint(name="deepseek-endpoint", auth_mode="key")
deployment = ManagedOnlineDeployment(
    name="blue",
    model=registered_model,
    instance_type="Standard_NC4as_T4_v3",
    instance_count=1
)
ml_client.online_endpoints.begin_create_or_update(endpoint)
ml_client.online_deployments.begin_create_or_update(deployment)
```
### 4.4 Test the Model Inference
```python
response = ml_client.online_endpoints.invoke(
    endpoint_name="deepseek-endpoint",
    request_file="sample_request.json"
)
print(response)
```

## 5. How Much VRAM is Needed for a 7B Model

Before deploying a large language model like DeepSeek 7B, it's important to estimate the VRAM requirements to avoid running into memory issues.

### Estimating VRAM from Model Binary Size

A smart way to estimate the VRAM needed is by checking the total size of the model's binary files (e.g., TensorFlow or PyTorch `.bin` files). Here's how:

- Suppose the model directory contains **7 binary files**, each approximately **2 GB** in size.
- The total size is **14 GB**.
- If the model uses **8-bit precision** (1 byte = 8 bits), then **14 GB of binary files** will roughly require **14–15 GB of VRAM**.
- If the model uses **16-bit precision** (e.g., BF16 or FP16), then you need **double the VRAM**, i.e., around **28–30 GB**.

### Additional Memory Considerations

Besides the model weights, you also need VRAM for:

- **Middleware**: Libraries like `vLLM`, `Transformers`, or `Ollama` consume additional memory.
- **Inference Overhead**: Memory usage increases with the number of concurrent requests and batch sizes.

### Rule of Thumb

- **8-bit model with 14 GB weights** → ~15 GB VRAM
- **16-bit model with 14 GB weights** → ~30 GB VRAM
- Add **2–4 GB** extra for middleware and inference buffer

### Always plan for a buffer to ensure smooth operation under load

I attempted to deploy the DeepSeek R1 Distilled Qwen 7B model using the Standard_NC4as_T4_v3 instance, which provides 16 GB of VRAM. During testing, the deployment process progressed smoothly until the final loading stage, where it encountered a VRAM allocation error. The system reported that only 15.75 GB of VRAM was available, while the model required slightly more than 16 GB. This suggests that even minimal overhead or background processes can impact deployment success when operating near the memory threshold.