# 🚀 Step-by-Step Guide to Deploying DeepSeek Distill Model on Azure ML

## Background: Why Azure ML Over Azure VM?

When deploying machine learning models, especially large language models like DeepSeek Distill, choosing the right infrastructure is crucial. While Azure Virtual Machines (VMs) offer flexibility, **Azure Machine Learning (Azure ML)** provides a **managed, scalable, and production-ready environment** for ML workflows.

### Benefits of Azure ML:
- **Managed Endpoints**: Easily deploy and manage models with autoscaling and traffic splitting.
- **Experiment Tracking**: Built-in tools for tracking runs, metrics, and artifacts.
- **Integrated MLOps**: Seamless integration with CI/CD pipelines.
- **Cost Efficiency**: Pay only for what you use with managed compute and storage.

## Requirements: Request GPU Quota on Azure ML
Before deploying, ensure your Azure ML workspace has access to GPU resources:

1. Go to your Azure ML workspace in the Azure Portal.
2. Navigate to **Usage + quotas**.
3. Request a quota increase for GPU-enabled compute (e.g., `Standard_NC4as_T4_v3`).
4. Wait for approval (usually within 24 hours).

## Workflow
### Step 0: Provison Azure ML Workspace
- Follow this [tutorial](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?view=azureml-api-2) to create Azure ML Workspace 

### Step 1: Start a Compute Instance
- Launch a **CPU-based compute instance** (e.g., 4 vCPUs, 16 GB RAM).
- This instance is only for running the notebook, not for inference.

### Step 2: Import Notebook and Assets

- Upload the deployment notebook [azml_deepseekr1.ipynb](azml_deepseekr1.ipynb) to your working folder.
- Create a **src** folder [photo](srcfolder.png) in your working folder, and upload the [score.py](score.py) in the src folder.
- Modify the score.py, replace the **"deepseek-qwen-1o5b"**  with your model name filled for `target_azml_model_name` below.
- Modify the notebook parameters:

**Variable remarks:**
- `target_huggingface_model_id`: The Hugging Face model ID to download (e.g., `"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"`).
- `target_subscription_id`: Your Azure subscription ID.
- `target_resource_group_name`: The Azure resource group name where your ML workspace resides.
- `target_workspace_name`: The Azure ML workspace name.
- `target_azml_model_name`: The name to register your model under in Azure ML. (e.g.`deepseek-qwen-1o5b` )
- `target_azml_model_desp`: Description for your registered model. (e.g.`DeepSeek-R1-Distill-Qwen-1.5B for inference` )
- `target_managed_endpoint_name`: Name for the Azure ML managed online endpoint. (e.g.`myendpoint-deepseek-qwen-1o5b` )
- `target_deployment_prefix`: Prefix for deployment naming (often includes hardware type). (e.g.`nvidia-t4-4core` )
- `target_GPU_SKU`: The Azure VM SKU for GPU resources (e.g., `"Standard_NC4as_T4_v3"`). 
- `target_deepseek_env`: Name for the custom Azure ML environment. (e.g.`deepseek-env` )


### Step 3: Execute the Notebook

- Start your computer instance and make sure you are on this kernel **Python 3.10 - AzureML**
- Click **">>"** button for **Restart kernel and run all cells**

#### 3.1 Create and Register Azure ML Environment
#### 3.2 Authenticate and Initialize MLClient
#### 3.3 Download and Save Hugging Face Model
#### 3.4 Register Model in Azure ML
#### 3.5 Create and Deploy Managed Online Endpoint
#### 3.6 Set Default Deployment and Traffic
#### 3.7 Test the Deployed Endpoint

## score.py — Model Inference Script
This script is designed to load a pre-trained language model and generate text completions based on a user-provided prompt. It is typically used in a deployment setting (e.g., Azure ML) to serve a causal language model like deepseek-r1-distill-qwen-1.5b.

### Model Initialization (init)
Loads the tokenizer and model from a specified directory (defaulting to AZUREML_MODEL_DIR).
Moves the model to the appropriate device (cuda if available, otherwise cpu).
Sets the model to evaluation mode.

### Text Generation (run)
- Accepts a JSON input with the following optional fields:
    prompt: The input text to generate from.
    temperature: Controls randomness in generation (default: 0.7).
    top_p: Controls nucleus sampling (default: 0.9).
    do_sample: Enables sampling (default: True).
    max_new_tokens: Limits the number of tokens to generate (default: 100).
- Tokenizes the input prompt.
- Ensures the total token length does not exceed the model's maximum (16384 tokens).
- Generates text using the model.
- Returns only the generated portion (excluding the prompt) as a JSON response.

### Customization Areas
You can tailor the script to your specific use case in several ways:

#### Model & Tokenizer
- Change the model path in init() to load a different model.
- Use a different tokenizer if your model requires one.

#### Generation Parameters
- Modify default values for:
    temperature
    top_p
    do_sample
    max_new_tokens
- Add support for other generation parameters like repetition_penalty, top_k, or stop_sequences.

#### Token Limits
- Adjust max_length_constant if your model supports a different context window.

#### Output Handling
- Post-process generated_text to:
    Remove special tokens
    Clean up formatting
    Add metadata (e.g., generation time, token count)

## How much VRAM is needed for a 7B Model

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
- **Inference Overhead**: Memory usage increases with **the number of concurrent requests and batch sizes**.

### Rule of Thumb

- **8-bit model with 14 GB weights** → ~15 GB VRAM
- **16-bit model with 14 GB weights** → ~30 GB VRAM
- Add minimum **2–4 GB** extra for middleware and inference buffer

### Always plan for a buffer to ensure smooth operation under load

While deploying the DeepSeek R1 Distilled Qwen 7B model on a Standard_NC4as_T4_v3 instance with 16 GB of VRAM, the process initially proceeded without issues. However, it failed at the final loading stage due to a VRAM allocation error. The system reported only 15.75 GB of usable VRAM, slightly below the model’s requirement of just over 16 GB. This incident highlights the importance of planning for a memory buffer, as even minimal overhead or background processes can consume critical resources. Operating at or near hardware limits leaves no room for variability, leading to potential deployment failures.