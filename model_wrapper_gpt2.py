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
        self.model = GPT2LMHeadModel.from_pretrained(topology, pad_token_id=self.tokenizer.eos_token_id,
                                                     cache_dir="cache")

        self.model.parallelize(device_map=device_map)

        if weights_filename is not None:
            self.model.load_state_dict(torch.load(weights_filename))

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors='pt').to(self.__input_device())

    def decode(self, data, skip_special_tokens=False):
        return self.tokenizer.decode(data, skip_special_tokens=skip_special_tokens)

    def generate(self, input_tokens, seed = None, max_length=100):
        self.model.eval()

        if seed is not None:
            torch.manual_seed(seed)

        output_tokens = self.model.generate(
            input_tokens,
            do_sample=True,
            max_length=max_length,
            top_p=0.78,
            top_k=0,
            temperature=0.7
        )

        return output_tokens[0]

    def train(self,
              dataset,
              output_prefix="snapshot",
              validator=None,
              learning_rate=3e-5,
              warmup_steps=5000,
              training_steps=100,
              epochs=10,
              iterations_in_bunch_count=16,
              max_sequence_length=400):

        self.model.train()

        data_item_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        #self.model.base_model.disable_grads_for_layers(layers_to_disable=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
        for name, param in self.model.named_parameters():
            if param.is_cpu:
                print(f"Disabled training for {name}")
                param.requires_grad = False

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)
        iteration_in_bunch = 0
        sum_loss = 0.0
        bunch_count = 0

        tmp_jokes_tens = None

        # types_list = [torch.float16] * 36
        # types_list[0] = torch.float32
        # types_list[-1] = torch.float32

        #self.model.cast_types(types_list)

        for epoch in range(epochs):
            if validator is not None:
                #self.model.eval()  -- called in generate
                validator(self)
                self.model.train()


            #self.model.transformer.half()
            # self.model.transformer.wte.half()
            # self.model.transformer.wpe.half()
            # for index, layer in enumerate(self.model.transformer.h):
            #     # if index >= 15:     # TODO Get from the map
            #     layer.half()

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
                if iteration_in_bunch == iterations_in_bunch_count:
                    time_per_lesson = (time.time() - start_time) / (lesson + 1)
                    print(f"Bunch {bunch_count} passed ({lesson} lessons "
                          f"of {lessons_count}, {time_per_lesson * 1000:.1f}ms per lesson)")
                    iteration_in_bunch = 0
                    bunch_count += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.model.zero_grad()

                if bunch_count == 100:
                    print(f"{bunch_count} bunches passed. Average loss "
                          f"per iteration is {sum_loss / bunch_count / iterations_in_bunch_count}")
                    bunch_count = 0
                    sum_loss = 0.0

            # Store the model after each epoch to compare the performance of them
            models_folder = "trained_models"
            if not os.path.exists(models_folder):
                os.mkdir(models_folder)

            # types_list = [
            #                  torch.float32] * 36  # [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 ]
            # self.model.cast_types(types_list)

            torch.save(self.model.state_dict(), os.path.join(models_folder, f"{output_prefix}_{epoch}.pt"))


