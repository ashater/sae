import huggingface_hub
import os
huggingface_hub.login(os.environ['HF_TOKEN'])
from nnsight import LanguageModel
import torch
model = LanguageModel("google/gemma-2b-it", trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float16)
model.requires_grad_(False);
#!wget -c 'https://huggingface.co/jbloom/Gemma-2b-Residual-Stream-SAEs/resolve/main/gemma_2b_blocks.6.hook_resid_post_16384_anthropic_fast_lr/sae_weights.safetensors?download=true' -O 'sae.safetensors'

from safetensors import safe_open
with safe_open("sae.safetensors", framework="pt") as st:
    w_dec = st.get_tensor("W_dec")


#@title Self-explanation in 18 lines
feature = 471  #@param {type: "integer"}
scale = 20.0  #@param {type: "number"}
se_demo = True  #@param {type: "boolean"}
max_new_tokens = 40  #@param {type: "integer"}
n_generate = 10  #@param {type: "integer"}

prompt = '<start_of_turn>user\nWhat is the meaning of the word "X"?<end_of_turn>\n<start_of_turn>model\nThe meaning of the word "X" is "'
positions = [i for i, a in enumerate(model.tokenizer.encode(prompt)) if model.tokenizer.decode([a]) == "X"]
with model.generate(prompt, max_new_tokens=max_new_tokens, num_return_sequences=n_generate, do_sample=True, scan=False, validate=False) as gen:
    vector = w_dec[[feature]]
    vector = vector / vector.norm()
    vector = vector * scale
    for position in positions:
        model.model.layers[2].output[0][:, position] = vector
    out = model.generator.output.save()
for i, l in enumerate(model.tokenizer.batch_decode(out)):
    print(f"{i+1}.", repr(l.partition(prompt)[2].partition("<eos>")[0]))
