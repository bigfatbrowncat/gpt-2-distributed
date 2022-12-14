import argparse
import csv
import os

from torch.utils.data import Dataset, DataLoader

from model_wrapper_gpt2 import GPT2ModelWrapper


# Parsing the arguments
parser = argparse.ArgumentParser(
                    prog = 'Distributed GPT-2 Inference',
                    description = 'Infer the GPT-2 network on the CPUs and GPUs present in the system',
                    epilog = 'Good luck!')

parser.add_argument('-m', '--model-filename')
parser.add_argument('-w', '--weights-filename')
parser.add_argument('-i', '--input')
parser.add_argument('-d', '--dev-map-filename')

args = parser.parse_args()

model_wrapper = GPT2ModelWrapper(
    config_name="config/device_map_cpu_and_gpu_24.json",
#    config_name="config/device_map_single_gpu_24.json",
    weights_filename=None
)


class ArithmeticsDataset(Dataset):
    def __init__(self, dataset_path='data/math'):
        super().__init__()

        short_jokes_path = os.path.join(dataset_path, 'math.csv')

        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"

        with open(short_jokes_path, encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            x = 0
            for row in csv_reader:
                joke_str = f"{row[0]}{self.end_of_text_token}"
                self.joke_list.append(joke_str)

    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]


dataset = ArithmeticsDataset()
# data_item_loader = DataLoader(dataset, batch_size=1, shuffle=True)

model_wrapper.train(dataset, "gpt2-medium-math-cpu", learning_bunch_size=4, epochs=20)
