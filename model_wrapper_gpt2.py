import json, torch
import os
import time

from torch.utils.data import DataLoader

from distributed_modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW


class GPT2ModelWrapper:
    def __input_device(self):
        return self.model.transformer.device

    def __init__(self, config_name: str, weights_filename: str = None):
        with open(config_name) as json_file:
            config = json.load(json_file)
            device_map = config['device_map']
            topology = config['topology']

        self.tokenizer = GPT2Tokenizer.from_pretrained(topology, cache_dir="cache")
        self.model = GPT2LMHeadModel.from_pretrained(topology, pad_token_id=self.tokenizer.eos_token_id, cache_dir="cache")
        self.model.parallelize(device_map=device_map)

        if weights_filename is not None:
            self.model.load_state_dict(torch.load(weights_filename))

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors='pt').to(self.__input_device())

    def decode(self, data):
        return self.tokenizer.decode(data, skip_special_tokens=True)

    def generate(self, text, seed = None, max_length=100):
        self.model.eval()

        input = self.encode(text)

        if seed is not None:
            torch.manual_seed(seed)

        sample_output = self.model.generate(
            input,
            do_sample=True,
            max_length=max_length,
            top_p=0.78,
            top_k=0)

        return self.decode(sample_output[0])

    def train(self,
              dataset,
              output_prefix="snapshot",
              learning_rate=3e-5,
              warmup_steps=5000,
              training_steps=100,
              epochs=10,
              learning_bunch_size=16,
              max_sequence_length=400):

        self.model.train()

        data_item_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
        iteration_in_bunch = 0
        sum_loss = 0.0
        bunch_count = 0

        tmp_jokes_tens = None

        for epoch in range(epochs):

            print(f"EPOCH {epoch} started " + '=' * 30)

            start_time = time.time()

            lessons_count = len(data_item_loader)
            lesson = 0
            for idx, joke in enumerate(data_item_loader):
                #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
                joke_tens = torch.tensor(self.tokenizer.encode(joke[0])).unsqueeze(0).to(self.__input_device())
                lesson += 1
                # Skip sample from dataset if it is longer than MAX_SEQ_LEN
                if joke_tens.size()[1] > max_sequence_length:
                    continue

                # The first joke sequence in the sequence
                if not torch.is_tensor(tmp_jokes_tens):
                    tmp_jokes_tens = joke_tens
                    continue
                else:
                    # The next joke does not fit in so we process the sequence and leave the last joke
                    # as the start for next sequence
                    if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > max_sequence_length:
                        work_jokes_tens = tmp_jokes_tens
                        tmp_jokes_tens = joke_tens
                    else:
                        # Add the joke to sequence, continue and try to add more
                        tmp_jokes_tens = torch.cat([tmp_jokes_tens, joke_tens[:, 1:]], dim=1)
                        continue
                ################## Sequence ready, process it trough the model ##################

                outputs = self.model(work_jokes_tens, labels=work_jokes_tens)
                loss, logits = outputs[:2]
                loss.backward()
                sum_loss = sum_loss + loss.detach().data

                iteration_in_bunch += 1
                if iteration_in_bunch == learning_bunch_size:
                    time_per_lesson = (time.time() - start_time) / (lesson + 1)
                    print(f"Bunch {bunch_count} passed ({lesson} lessons of {lessons_count}, {time_per_lesson:.2} per lesson)")
                    iteration_in_bunch = 0
                    bunch_count += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.model.zero_grad()

                if bunch_count == 100:
                    print(f"100 bunches passed. Average loss per bunch is {sum_loss / bunch_count}")
                    bunch_count = 0
                    sum_loss = 0.0

            # Store the model after each epoch to compare the performance of them
            models_folder = "trained_models"
            if not os.path.exists(models_folder):
                os.mkdir(models_folder)
            torch.save(self.model.state_dict(), os.path.join(models_folder, f"{output_prefix}_{epoch}.pt"))
