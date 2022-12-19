import argparse
import csv
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast as cuda_autocast

from model_wrapper_gpt2 import GPT2ModelWrapper


# Parsing the arguments
parser = argparse.ArgumentParser(
                    prog = 'Distributed GPT-2 Inference',
                    description = 'Infer the GPT-2 network on the CPUs and GPUs present in the system',
                    epilog = 'Good luck!')

parser.add_argument('-c', '--config-filename')
parser.add_argument('-w', '--weights-filename')

args = parser.parse_args()

model_wrapper = GPT2ModelWrapper(
    config_name=args.config_filename,
    weights_filename=args.weights_filename
)

class WordsDataset(Dataset):
    def __init__(self, dataset_path='data/math'):
        super().__init__()

        short_jokes_path = os.path.join(dataset_path, 'math.csv')

        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"

        with open(short_jokes_path, encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')

            for row in csv_reader:
                line = row[0].replace('<br/>', '\n')
                joke_str = f"{line}{self.end_of_text_token}"
                self.joke_list.append(joke_str)

    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]


def enc_gen_dec(text):
    return model_wrapper.decode(model_wrapper.generate(model_wrapper.encode(text), max_length=50))

def validate(model_wrapper):
    BOT_NAME = "AI"
    USER_NAME = "User"

    print("=========== WORDS ==============")

    print(enc_gen_dec('word "come" starts with'))
    print(enc_gen_dec('word "come" ends with'))
    print(enc_gen_dec('The first letter in "with" is'))
    print(enc_gen_dec('The last letter in "with" is'))
    print(enc_gen_dec('The first letter in "welcome" is'))
    print(enc_gen_dec('The last letter in "welcome" is'))
    print(enc_gen_dec('The first letter in "crocodile" is'))
    print(enc_gen_dec('The last letter in "crocodile" is'))
    print(enc_gen_dec(f'{USER_NAME}: Write a word that ends with "a".'))
    print(enc_gen_dec(f'{USER_NAME}: Write a word with the letter "s" in its end.'))
    print(enc_gen_dec(f'{USER_NAME}: Write a word that starts with "e".'))
    print(enc_gen_dec(f'{USER_NAME}: Write a word that starts with the last letter of "crocodile".'))

    print("=========== MATH ==============")

    print(enc_gen_dec("Multiply two by two. The result is"))
    print(enc_gen_dec("Multiply three by two. The result is"))
    print(enc_gen_dec("Add five to six. The result is"))
    print(enc_gen_dec("Add two to 2. It is"))
    print(enc_gen_dec("Add 3 to three. The result is"))
    print(enc_gen_dec("3 multiplied by six equals"))
    print(enc_gen_dec("five multiplied by 2 equals"))
    print(enc_gen_dec("2 + 4 ="))
    print(enc_gen_dec("I am a very smart guy, I know that 3 * 7 ="))
    print(enc_gen_dec(f"{USER_NAME}: Is it true that 2 + 3 = 6?\n{BOT_NAME}:"))
    print(enc_gen_dec(f"{USER_NAME}: Is it true that seven multiplied by seven is 49?\n{BOT_NAME}:"))
    print(enc_gen_dec(f"{USER_NAME}: How much is seven multiplied by seven?\n{BOT_NAME}:"))
    print(enc_gen_dec("How much is 7 * 6?"))
    print(enc_gen_dec("How much is twenty multiplied by nineteen?"))
    print(enc_gen_dec("How much is one plus two plus three?"))


dataset = WordsDataset()
# data_item_loader = DataLoader(dataset, batch_size=1, shuffle=True)

with cuda_autocast(enabled=True, dtype=torch.float16):
    model_wrapper.train(dataset, "gpt2-large-auto", validator=validate, iterations_in_bunch_count=4, epochs=20)

