import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm
import evaluate

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 确保程序的每次运行都具有确定性，可以得到相同的结果，用于程序的复现
torch.backends.cudnn.deterministic = True

dataset = datasets.load_dataset("bentrevett/multi30k")
# print(dataset)
train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)

en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq=min_freq,
    special=special_tokens,
)