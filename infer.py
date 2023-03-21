import argparse
import os.path
import sys

import torch
from torch.cuda.amp import autocast as cuda_autocast
from model_wrapper_gpt import GPTModelWrapper

# Parsing the arguments
parser = argparse.ArgumentParser(
                    prog = 'Distributed GPT-2 Inference',
                    description = 'Infer the GPT-2 network on the CPUs and GPUs present in the system',
                    epilog = 'Good luck!')

parser.add_argument('-c', '--config-filename')
parser.add_argument('-w', '--weights-filename')
parser.add_argument('-i', '--input')

args = parser.parse_args()

print("Loading...", end='')
model_wrapper = GPTModelWrapper(
    config_name=args.config_filename,
    weights_filename=args.weights_filename
)
print(" done")
print('')
USER_NAME = "User"
BOT_NAME = "AI"
SETTING = None #"Human Trainer talks to an AI called Lana. Lana represents a female robot."  #"User is talking to an AI. The AI trusts User."
SEED = 1985

previous_input = ""
if SETTING is not None:
    previous_input += f"{SETTING}"
    print(f"Setting: {SETTING}")
print(f"Seed: {SEED}")
print('')
print(f"{USER_NAME}: ", end=''); sys.stdout.flush()
for line in sys.stdin:
    if line.endswith('\n'): line = line[0:-1]
    if line != "":
        if line == "!exit" or line == "!quit":
            break
        elif line == "!save":
            n = 1
            while os.path.isfile(f"data/dialogs/log{n}.txt"):
                n += 1
            with open(f"data/dialogs/log{n}.txt", "w") as f:
                f.write(previous_input)
                break
        else:
            pred_index = -1
            previous_input += f"\n{USER_NAME}: {line}\n{BOT_NAME}:"
            pred_answer_index = len(previous_input)
            start_answering = True
            while pred_index == -1:
                #input = f"{inputQ}\n{inputA}"
                input = previous_input

                encoded_input = model_wrapper.encode(input)
                with cuda_autocast(enabled=True, dtype=torch.float16):
                    encoded_output = model_wrapper.generate(encoded_input, max_length=len(encoded_input[0]) + 50, seed=SEED)
                output = model_wrapper.decode(encoded_output)

                # Assuming that the bot answers only for itself, so cutting away any "predictions" about what the user will say next
                pred_index = output.find(f"\n{USER_NAME}:", pred_answer_index)
                if pred_index > -1:
                    output = output[0:pred_index]
                else:
                    # Looking for the end-of-text marker
                    eot_index = output.find(f"<|endoftext|>", pred_answer_index)
                    if eot_index > -1:
                        pred_index = eot_index
                        output = output[0:pred_index]
                        if output[-1] == '\n':
                            output = output[0:-1]
                    else:
                        pred_index = output.find(f"\n{BOT_NAME}:", pred_answer_index)
                        if pred_index > -1:
                            output = output[0:pred_index]

                # Printing only the appended text (not the whole history again)
                if start_answering:
                    start_answering = False
                    print(output[len(input) - 1 - len(BOT_NAME):], end=''); sys.stdout.flush()
                else:
                    print(output[len(input):], end=''); sys.stdout.flush()

                previous_input = output

            if len(input) > len(output):
                # That means we have already printed out a part of the User's name. We need to take it back. So printing the carriage return
                print('\r', end=''); sys.stdout.flush()
            else:
                print('')

            pred_answer_index = pred_index

    print(f"{USER_NAME}: ", end=''); sys.stdout.flush()
