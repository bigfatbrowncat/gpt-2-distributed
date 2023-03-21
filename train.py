import argparse
import csv
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast as cuda_autocast

from model_wrapper_gpt import GPTModelWrapper

# Parsing the arguments
parser = argparse.ArgumentParser(
    prog='Distributed GPT-2 Inference',
    description='Infer the GPT-2 network on the CPUs and GPUs present in the system',
    epilog='Good luck!')

parser.add_argument('-c', '--config-filename')
parser.add_argument('-w', '--weights-filename')

args = parser.parse_args()

print("Loading...", end='')
model_wrapper = GPTModelWrapper(
    config_name=args.config_filename,
    weights_filename=args.weights_filename
)
print(" done")


class LogsDataset(Dataset):
    def __init__(self, dataset_path='data/dialogs'):
        super().__init__()
        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"

        for filename in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, filename), "r", encoding='utf8') as f:
                contents = f.read() + self.end_of_text_token
                self.joke_list.append(contents)

        #self.joke_list = self.joke_list * 1000
        #pass

    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]


def enc_gen_dec(text):
    return model_wrapper.decode(model_wrapper.generate(model_wrapper.encode(text), max_length=50))


def validate(model_wrapper):
    print(enc_gen_dec('User: What is your name?\nAI:'))


dataset = LogsDataset()
# data_item_loader = DataLoader(dataset, batch_size=1, shuffle=True)

with cuda_autocast(enabled=True, dtype=torch.float16):
    model_wrapper.train(dataset, "gpt2-xl-dialog", validator=validate, epochs=10, learning_rate=1e-4, iterations_in_bunch_count=1)
