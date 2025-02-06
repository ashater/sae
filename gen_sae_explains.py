# Import necessary libraries
import huggingface_hub
import os

# Login to Hugging Face Hub using the token from environment variables
huggingface_hub.login(os.environ['HF_TOKEN'])

# Import the LanguageModel class from nnsight and torch library
from nnsight import LanguageModel
import torch

# Initialize the language model with specific parameters
model = LanguageModel("google/gemma-2b-it", trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float16)
model.requires_grad_(False)

# Download the SAE weights file (commented out)
#!wget -c 'https://huggingface.co/jbloom/Gemma-2b-Residual-Stream-SAEs/resolve/main/gemma_2b_blocks.6.hook_resid_post_16384_anthropic_fast_lr/sae_weights.safetensors?download=true' -O 'sae.safetensors'

# Load the SAE weights using safetensors
from safetensors import safe_open
with safe_open("sae.safetensors", framework="pt") as st:
    w_dec = st.get_tensor("W_dec")

# Define parameters for self-explanation generation
feature = 1400  # Number of features
scale = 20.0  # Scaling factor for vectors
se_demo = True  # Whether to run a demonstration
max_new_tokens = 40  # Maximum number of new tokens to generate
n_generate = 10  # Number of sequences to generate

# Initialize an empty list to store explanations
explains = []

# Loop over each feature in the decoder weights
for ft in range(0, w_dec.shape[0]):
    # Define the prompt for the model
    prompt = '<start_of_turn>user\nWhat is the meaning of the word "X"?<end_of_turn>\n<start_of_turn>model\nThe meaning of the word "X" is "'
    
    # Find positions of the token "X" in the encoded prompt
    positions = [i for i, a in enumerate(model.tokenizer.encode(prompt)) if model.tokenizer.decode([a]) == "X"]
    
    # Generate new sequences based on the prompt
    with model.generate(prompt, max_new_tokens=max_new_tokens, num_return_sequences=n_generate, do_sample=True, scan=False, validate=False) as gen:
        # Normalize and scale the vector for the current feature
        vector = w_dec[[ft]]
        vector = vector / vector.norm()
        vector = vector * scale
        
        # Insert the vector into the model's output at the identified positions
        for position in positions:
            model.model.layers[2].output[0][:, position] = vector
        
        # Save the generated output
        out = model.generator.output.save()
    
    # Initialize a local list to store explanations for the current feature
    local = []
    
    # Decode the generated output and extract the relevant part of the response
    for i, l in enumerate(model.tokenizer.batch_decode(out)):
        local.append(repr(l.partition(prompt)[2].partition("<eos>")[0]))
    
    # Append the local explanations to the main list
    explains.append(local)