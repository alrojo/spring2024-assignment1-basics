def memory(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    # memory:
    embedding_layer = vocab_size * d_model
    positional_layer = context_length * d_model
    qkv = 3*d_model**2
    ffn = 2*d_model*d_ff
    ln = d_model
    tblock = qkv+ffn+2*ln
    output_layer = d_model*vocab_size
    tmodel = embedding_layer + positional_layer + num_layers*tblock + ln + output_layer
    partial_memory = {
            "embeddings": (embedding_layer+positional_layer)/tmodel,
            "attn": qkv*num_layers/tmodel,
            "ffn": ffn*num_layers/tmodel,
            "output": output_layer/tmodel
    }
    return partial_memory, tmodel

def flops(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    # compute
    attn = 8*context_length*d_model**2 + 4*context_length**2*d_model
    ffn = 4*context_length*d_ff*d_model
    output_layer = 2*context_length*d_model*vocab_size
    total_cost = num_layers*(attn + ffn) + output_layer
    partial_costs = {
            "attn": attn*num_layers/total_cost,
            "ffn": ffn*num_layers/total_cost,
            "output": output_layer/total_cost
    }
    return partial_costs, total_cost

def pretty_print(dictionary):
    for key, value in dictionary.items():
        print("\t%s: %.3f" %(key, value))

def print_all_values(name, model):
    gpt_mem, gpt_tot_mem = memory(**model)
    gpt_flops, gpt_tot_flops= flops(**model)
    print("%s: memory, total memory: %.3f M" %(name, gpt_tot_mem/1e6))
    pretty_print(gpt_mem)
    print("%s: flops, total flops: %.3f Gflops" %(name, gpt_tot_flops/1e9))
    pretty_print(gpt_flops)

gpt2XL = {"vocab_size":50257, "context_length":1024, "num_layers":48, "d_model":1600, "num_heads":25, "d_ff":6400}
gpt2small = {"vocab_size":50257, "context_length":1024, "num_layers":12, "d_model":768, "num_heads":12, "d_ff":6400}
gpt2medium = {"vocab_size":50257, "context_length":1024, "num_layers":24, "d_model":1024, "num_heads":16, "d_ff":6400}
gpt2large = {"vocab_size":50257, "context_length":1024, "num_layers":36, "d_model":1280, "num_heads":20, "d_ff":6400}
gpt2XL_longcontext = {"vocab_size":50257, "context_length":16384, "num_layers":48, "d_model":1600, "num_heads":25, "d_ff":6400}
gpts = {
        "gpt2XL": gpt2XL,
        "gpt2small": gpt2small,
        "gpt2medium": gpt2medium,
        "gpt2large": gpt2large,
        "gpt2XL_longcontext": gpt2XL_longcontext,
}
print("MEMORY AND FLOPS")
for name, model in gpts.items():
    print_all_values(name, model)
    print()
