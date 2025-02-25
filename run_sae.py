# Import necessary libraries
from nnsight import LanguageModel
import torch

# Initialize the language model with specific parameters
model = LanguageModel("google/gemma-2b-it", trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float16)
model.requires_grad_(False)

# Download the SAE weights file (commented out)
#!wget -c 'https://huggingface.co/jbloom/Gemma-2b-Residual-Stream-SAEs/resolve/main/gemma_2b_blocks.6.hook_resid_post_16384_anthropic_fast_lr/sae_weights.safetensors?download=true' -O 'sae.safetensors'

# Load the SAE weights using safetensors
from safetensors import safe_open
with safe_open("../sae.safetensors", framework="pt") as st:
    w_dec = st.get_tensor("W_dec")

# Define parameters for self-explanation generation
feature = 15881  # Feature index to use
scale = 20.0  # Scaling factor for vectors
se_demo = True  # Whether to run a demonstration
max_new_tokens = 40  # Maximum number of new tokens to generate
n_generate = 10  # Number of sequences to generate

# Define the prompt for the model
prompt = '<start_of_turn>user\nWhat is the meaning of the word "X"?<end_of_turn>\n<start_of_turn>model\nThe meaning of the word "X" is "'

# Find positions of the token "X" in the encoded prompt
positions = [i for i, a in enumerate(model.tokenizer.encode(prompt)) if model.tokenizer.decode([a]) == "X"]

# Generate new sequences based on the prompt
with model.generate(prompt, max_new_tokens=max_new_tokens, num_return_sequences=n_generate, do_sample=True, scan=False, validate=False) as gen:
    # Normalize and scale the vector for the current feature
    vector = w_dec[[feature]]
    vector = vector / vector.norm()
    vector = vector * scale
    
    # Insert the vector into the model's output at the identified positions
    for position in positions:
        model.model.layers[2].output[0][:, position] = vector
    
    # Save the generated output
    out = model.generator.output.save()

# Decode the generated output and print the relevant part of the response
for i, l in enumerate(model.tokenizer.batch_decode(out)):
    print(f"{i+1}.", repr(l.partition(prompt)[2].partition("<eos>")[0]))



import math
replacement_layer = 2  #@param {type: "integer"}
diagnostic_layer = 15 #@param {type: "integer"}
scale_guess = 17.0  #@param {type: "number"}
prompt = '<start_of_turn>user\nWhat is the meaning of the word "X"? Be verbose.<end_of_turn>\n<start_of_turn>model\nThe meaning of the word "X" is "'  #@param {type: "string"}
# prompt = '<start_of_turn>user\nWhat is the meaning of the word "X"?<end_of_turn>\n<start_of_turn>model\nThe meaning of the word "X" is "'  #@param {type: "string"}
positions = [i for i, a in enumerate(model.tokenizer.encode(prompt)) if model.tokenizer.decode([a]) == "X"]
def activation_replacement(vectors):
    for position in positions:
        model.model.layers[max(0, replacement_layer - 1)].output[0][:, position] = vectors# .half()

def compute_metric(vector):
    last = model.model.layers[max(0, diagnostic_layer - 1)].output[0][:, -1].to(torch.float64)
    last = last / last.norm(dim=-1)
    metric = (last @ vector).save()
    return metric


def get_metric(scale):
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        with model.trace(prompt):
            vector = w_dec.detach()[[feature]]#.half()
            vector = vector / vector.norm()
            activation_replacement(vector * scale)
            metric = compute_metric(vector[0]).save()
    return metric


def tune_scale(scale_guess):
    try:
        with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
            with model.trace(prompt):
                vector = w_dec.detach()[[feature]].to(torch.float64)
                vector = vector / vector.norm()
                vector = vector.detach()
                scale = torch.nn.Parameter(torch.tensor(float(scale_guess), dtype=torch.float64), requires_grad=True).save()
                activation_replacement(vector * scale)
                metric = compute_metric(vector[0])[0].save()
                metric.backward(create_graph=True, inputs=(scale,))
        derivative = scale.grad.item()
        scale.grad.backward(inputs=(scale,))
        second_derivative = scale.grad.item() - derivative
        results = metric.item(), derivative, second_derivative, 0.0
    except:
        # pythonic context management
        try:
            pass
        except NameError:
            pass
        raise


    return results