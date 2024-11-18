import argparse, os, importlib, json, time, math, torch, logging
import numpy as np
from datetime import datetime, timedelta
import cs336_basics.utils.data as data
import cs336_basics.utils.io as io
import cs336_basics.utils.nn as nn
from cs336_basics.model import TransformerLM
from cs336_basics.ablation import TransformerLM_ablation
from cs336_basics.optimizer import AdamW, lr_schedule

###
# 1. Ability to configure and control the various model and optimizer hyperparameters
###
parser = argparse.ArgumentParser(description="training script for CS336")
parser.add_argument("--config_name", type=str, help="path to configurations",
                    required=True)
args = parser.parse_args()
def load_and_log_configs(config_name):
    config_path = "project_data/configs/%s.json" % config_name
    if not os.path.isfile(config_path):
        assert False
    with open(config_path) as f:
        config = json.load(f)
    return config, config_name
config, config_name = load_and_log_configs(args.config_name)

timestamp = timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (args.config_name, timestamp)
experiment_folder = "experiments/%s" % experiment_id
os.makedirs(experiment_folder)
# setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{experiment_folder}/experiment.log"),  # File in training log directory
        logging.StreamHandler()  # Console output
    ]
)
logger = logging.getLogger("ExperimentLogger")
logger.info("experiment started, saving to %s ..." % experiment_folder)
logger.info("Configurations: %s" % config_name)
for key, value in config.items():
    logger.info(f"\t{key}: {value}")

###
# 2. Memory-eﬀicient loading of training and validation large datasets with np.memmap
###
def load_and_log_dataloader(path, name, logger):
    logger.info("loading %s from %s with mmap ..." % (name, path))
    dataset = np.memmap(path, dtype=np.uint16, mode='r')
    million_tokens = len(dataset) / 1e6 
    logger.info("successfully loaded %s data with %.2f million tokens..." %
          (name, million_tokens))
    return dataset
train_data = load_and_log_dataloader(config["train_path"], "training data", logger)
valid_data = load_and_log_dataloader(config["valid_path"], "validation data", logger)
###
# 2.1 Create model
###
if config['ablation']:
    model = TransformerLM_ablation(**config['model'])
else:
    model = TransformerLM(**config['model'])
model.to(config["device"])
optimizer = AdamW(model.parameters(), **config['optimizer'])
###
# 3. Serializing checkpoints to a user-provided path.
###
if config["checkpoint"]:
    if not os.path.isfile(config["checkpoint"]):
        logger.info("specified checkpoint: (%s) does not exist, exiting ..."
              % config["checkpoint"])
        assert False
    logger.info("starting from previous checkpoint %s..." % config["checkpoint"])
    # load checkpoint
    io.load_checkpoint(config["checkpoint"], model, optimizer)

###
# 4.Periodically logging training and validation performance
#   (e.g., to console and/or an external service like Weights and Biases). 
###
if config["wandb"]:
    import wandb
    wandb.init(project=config["wandb"], name=config_name)

def run_validation(_model, _dataset, _config, t):
    _model.eval()
    cl = _config["model"]["context_length"]
    n = math.floor(len(_dataset) / cl)*cl
    losses = []
    with torch.no_grad():
        for x_batch, t_batch in data.get_batch_test(_dataset, _config):
            x_batch, t_batch = x_batch.long(), t_batch.long()
            y_batch = _model(x_batch)
            loss = nn.cross_entropy(y_batch, t_batch)
            losses.append(loss.item())
    avg_loss = np.mean(losses)
    avg_perplexity = math.exp(avg_loss)
    if config["wandb"]:
        wandb.log({'val_loss': avg_perplexity}, step=t)
    logger.info("Validation perplexity: %.4f" % avg_perplexity)

total_loss = 0
for i in range(config["total_iters"]):
    x_batch, t_batch = data.get_batch(
        train_data, config["batch_size"], config["model"]["context_length"], config["device"]
        )
    x_batch, t_batch = x_batch.long(), t_batch.long()
    y_batch = model(x_batch)
    loss = nn.cross_entropy(y_batch, t_batch)
    loss.backward()
    nn.gradient_clipping(model.parameters(), max_l2_norm=config["max_l2_norm"])
    lr_i = lr_schedule(t=i, lr_max=config["lr_max"], lr_min=config["lr_min"],
                                 T_w=config["T_w"], T_c=config["T_c"])
    optimizer.set_lr(lr_i)
    optimizer.step()
    total_loss = 0.9*total_loss + 0.1*math.exp(loss.item())
    if ((i % config["print_every"]) == 0):
        if config["wandb"]:
            wandb.log({'train_loss': total_loss, 'lr': lr_i}, step=i)
        logger.info("iter: %d, lr:%.6f, cost:%.4f" % (i, lr_i, total_loss))
        total_loss = 0
    if (((i+1) % config["valid_every"]) == 0):
        run_validation(model, valid_data, config, i)
    if (((i+1) % config["checkpoint_every"]) == 0):
        checkpoint_name = "checkpoint_%d" % i
        checkpoint_path = os.path.join(experiment_folder, checkpoint_name) 
        logger.info("saving checkpoint to %s ..." % checkpoint_path)
        io.save_checkpoint(model, optimizer, i, checkpoint_path)
