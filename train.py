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


def validate(model_wrapper):
    BOT_NAME = "AI"
    USER_NAME = "User"

    print("=========== WORDS ==============")

    print(model_wrapper.generate('word "come" starts with'))
    print(model_wrapper.generate('word "come" ends with'))
    print(model_wrapper.generate('The first letter in "with" is'))
    print(model_wrapper.generate('The last letter in "with" is'))
    print(model_wrapper.generate('The first letter in "welcome" is'))
    print(model_wrapper.generate('The last letter in "welcome" is'))
    print(model_wrapper.generate('The first letter in "crocodile" is'))
    print(model_wrapper.generate('The last letter in "crocodile" is'))
    print(model_wrapper.generate(f'{USER_NAME}: Write a word that ends with "a".'))
    print(model_wrapper.generate(f'{USER_NAME}: Write a word with the letter "s" in its end.'))
    print(model_wrapper.generate(f'{USER_NAME}: Write a word that starts with "e".'))
    print(model_wrapper.generate(f'{USER_NAME}: Write a word that starts with the last letter of "crocodile".'))

    print("=========== MATH ==============")

    print(model_wrapper.generate("Multiply two by two. The result is"))
    print(model_wrapper.generate("Multiply three by two. The result is"))
    print(model_wrapper.generate("Add five to six. The result is"))
    print(model_wrapper.generate("Add two to 2. It is"))
    print(model_wrapper.generate("Add 3 to three. The result is"))
    print(model_wrapper.generate("3 multiplied by six equals"))
    print(model_wrapper.generate("five multiplied by 2 equals"))
    print(model_wrapper.generate("2 + 4 ="))
    print(model_wrapper.generate("I am a very smart guy, I know that 3 * 7 ="))
    print(model_wrapper.generate(f"{USER_NAME}: Is it true that 2 + 3 = 6?\n{BOT_NAME}:"))
    print(model_wrapper.generate(f"{USER_NAME}: Is it true that seven multiplied by seven is 49?\n{BOT_NAME}:"))
    print(model_wrapper.generate(f"{USER_NAME}: How much is seven multiplied by seven?\n{BOT_NAME}:"))
    print(model_wrapper.generate("How much is 7 * 6?"))
    print(model_wrapper.generate("How much is twenty multiplied by nineteen?"))
    print(model_wrapper.generate("How much is one plus two plus three?"))


dataset = WordsDataset()
# data_item_loader = DataLoader(dataset, batch_size=1, shuffle=True)

model_wrapper.train(dataset, "gpt2-xl-auto", validator=validate, iterations_in_bunch_count=8, epochs=20)

