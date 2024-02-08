from huggingface_hub import hf_hub_download
from config import MODEL_ID, MODEL_BASE_NAME, MODELS_PATH, CONTEXT_WINDOW_SIZE, MAX_NEW_TOKENS,N_BATCH,N_GPU_LAYERS
from langchain_community.llms import LlamaCpp

# Download the model from HuggingFace
def download_model(model_id=MODEL_ID, model_base=MODEL_BASE_NAME):
    print(f"Loading {model_id} from HuggingFace")
    # Download the models from Hugging face - mode documentation can be found here 
    # https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/file_download#huggingface_hub.hf_hub_download
    model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_base,
            resume_download=True,
            cache_dir=MODELS_PATH,
            local_files_only = False
            )
    return model_path

# Loading the model 
def mount_model(model_path):
    print(f"Using AutoModelForCausalLM Mounting {MODEL_ID} from HuggingFace")

    model = LlamaCpp(
        model_path=model_path,  # Download the model file first
        n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
    )

    return model
