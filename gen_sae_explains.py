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
#from safetensors import safe_open
#with safe_open("sae.safetensors", framework="pt") as st:
#    w_dec = st.get_tensor("W_dec")

from sae_lens import SAE
layer = 12

# get the SAE for this layer
sae, cfg_dict, _ = SAE.from_pretrained(
    release = "gemma-2b-res-jb",
    sae_id = f"blocks.{layer}.hook_resid_post",
#    device = 'cuda:1'
)

# Define parameters for self-explanation generation
feature = 1400  # Number of features
scale = 56.0  # Scaling factor for vectors
se_demo = True  # Whether to run a demonstration
max_new_tokens = 40  # Maximum number of new tokens to generate
n_generate = 10  # Number of sequences to generate

# Initialize an empty list to store explanations
explains = []

test = [ 3629,   475,  6443,   787, 14794,  5972,  5503, 15488,  4383,  4722,
         8050,  4991,  7091, 12567,  1729,  8691, 12185, 16042, 13945, 13265,
        11450,  5611,  9251, 14709,  6738,  5659,  9041,  8431,  8211, 14557,
        11135,  6775,  1687,  8728, 10550,  9459,  6943, 14992,  9952, 15132,
         1004, 13303, 15616,  8370, 13180, 15194, 12101, 10275,   889, 14433,
        11472,  9166,  1254,  1276, 12955, 10202, 13693,  7128,  9954, 12760,
         1977,  4701, 10394, 10754,  7344,  8475, 15455,  1913,  2356,  5581,
         3519,   172, 12890, 13044,  9313, 11299, 10402, 10261,  5454,  6346,
         5624,  2969,  9894,  9198,  1568, 10725,  9632,  4256,    92, 15403,
         2624, 14645,  3363,  1053,  1419,  1258, 15607, 15804,   923, 15685]

# Loop over each feature in the decoder weights
#for ft in range(0, w_dec.shape[0]):
for ft in test:
    # Define the prompt for the model
    prompt = '<start_of_turn>user\nWhat is the meaning of the word "X"?<end_of_turn>\n<start_of_turn>model\nThe meaning of the word "X" is "'
    
    # Find positions of the token "X" in the encoded prompt
    positions = [i for i, a in enumerate(model.tokenizer.encode(prompt)) if model.tokenizer.decode([a]) == "X"]
    
    # Generate new sequences based on the prompt
    with model.generate(prompt, max_new_tokens=max_new_tokens, num_return_sequences=n_generate, do_sample=True, scan=False, validate=False) as gen:
        # Normalize and scale the vector for the current feature
        vector = sae.W_dec[[ft]]
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
   
    print(local)
    # Append the local explanations to the main list
    explains.append(local)


import pandas as pd
pd.Series(explains).to_csv('test.csv')
