import argparse, os, json, torch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import Tokenizer
import cs336_basics.utils.io as io 
import cs336_basics.utils.nn as nn

parser = argparse.ArgumentParser(description="generator script for CS336")
parser.add_argument("--config_name", type=str, help="path to configurations",
                    default="base_config")
parser.add_argument("--checkpoint_name", type=str, help="path to configurations")
parser.add_argument("--temp", type=float, help="path to configurations",
                    default=1)
parser.add_argument("--nucleus", type=float, help="path to configurations",
                    default=0.9)
args = parser.parse_args()
def load_and_log_configs(config_name):
    config_path = "project_data/configs/%s.json" % config_name
    if not os.path.isfile(config_path):
        print("can't find configuration: %s, exiting ..." % config_path)
        assert False
    print("loading config from %s ..." % config_path)
    with open(config_path) as f:
        config = json.load(f)
    print("Configurations:")
    for key, value in config.items():
        print(f"\t{key}: {value}")
    return config, config_name
config, config_name = load_and_log_configs(args.config_name)
context_length = config["model"]["context_length"]
###
# 1 Create model
###
model = TransformerLM(**config['model'])
model.to(config["device"])
optimizer = AdamW(model.parameters(), **config['optimizer'])

###
# 2 Load checkpoint
###
if not os.path.isfile(config["checkpoint"]):
    print("specified checkpoint: (%s) does not exist, exiting ..."
          % config["checkpoint"])
    assert False
print("starting from previous checkpoint %s..." % config["checkpoint"])
io.load_checkpoint(config["checkpoint"], model, optimizer)

###
# 3 Load tokenizer
###
tokenizer = Tokenizer.from_files(vocab_filepath=config["vocab_path"],
                                 merges_filepath=config["merges_path"],
                                 special_tokens=config["special_tokens"])
###
# 4 Generate data
###
def text_to_gpu(x, _tokenizer, _device):
    x = _tokenizer.encode(x)
    x = torch.tensor(x).unsqueeze(0).to(_device)
    return x

def temp_sampling(x, _model, temp=1):
    model.eval()
    with torch.no_grad():
        logits = _model(x)
        logits = logits[:,-1,:] # last seq element
        probs = nn.softmax(logits/temp, dim=-1)
        sample = torch.multinomial(probs, num_samples=1)
        return sample.item()

def nucleus_sampling(x, _model, p=0.9, temperature=1, min_sampling=2):
    model.eval()
    with torch.no_grad():
        # get logits
        logits = _model(x)
        logits = logits[:,-1,:]/temperature # last seq element
        # get sorted, cumulative prob
        sorted_logits, indices = torch.sort(logits, descending=True)
        probs_sorted = nn.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
        # get cutoff
        cutoff_index = (cumulative_probs > p).nonzero(as_tuple=True)[1][0].item()
        cutoff_index = max(cutoff_index, min_sampling)
        probs = nn.softmax(logits, dim=-1)
        probs[indices[cutoff_index:]] = 0 # setting cutoff values to 0
        probs = probs/torch.sum(probs) # normalizing to 1
        sample = torch.multinomial(probs, num_samples=1)
        return sample.item()

while True:
    print()
    text = input("Prompt: ")
    max_len = int(input("max length: "))
    print(text, end="")
    for i in range(max_len):
        x = text_to_gpu(text, tokenizer, config["device"])
        x = x[:,-context_length:]
        next_token = nucleus_sampling(x, model, args.nucleus, args.temp)
        decoded_token = tokenizer.decode([next_token])
        print(decoded_token, end="")
        if decoded_token in config["special_tokens"]:
            break
        text += decoded_token
