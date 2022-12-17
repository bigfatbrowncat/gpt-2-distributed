import argparse
import sys

from model_wrapper_gpt2 import GPT2ModelWrapper

# Parsing the arguments
parser = argparse.ArgumentParser(
                    prog = 'Distributed GPT-2 Inference',
                    description = 'Infer the GPT-2 network on the CPUs and GPUs present in the system',
                    epilog = 'Good luck!')

parser.add_argument('-c', '--config-filename')
parser.add_argument('-w', '--weights-filename')
parser.add_argument('-i', '--input')

args = parser.parse_args()

model_wrapper = GPT2ModelWrapper(
    config_name=args.config_filename,
    weights_filename=args.weights_filename
)

BOT_NAME = "AI"
USER_NAME = "Trainer"

previous_input = ""
print(f"{USER_NAME}: ", end='')
for line in sys.stdin:
    if line.endswith('\n'): line = line[0:-1]
    if line != "":
        if line == "!exit" or line == "!quit":
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
                encoded_output = model_wrapper.generate(encoded_input, max_length=len(encoded_input[0]) + 50, seed=1985)
                output = model_wrapper.decode(encoded_output)

                # Assuming that the bot answers only for itself, so cutting away any "predictions" about what the user will say next
                pred_index = output.find(f"\n{USER_NAME}:", pred_answer_index)
                if pred_index > -1:
                    output = output[0:pred_index]
                else:
                    pred_index = output.find(f"\n{BOT_NAME}:", pred_answer_index)
                    if pred_index > -1:
                        output = output[0:pred_index]

                # Printing only the appended text (not the whole history again)
                if start_answering:
                    start_answering = False
                    print(output[len(input) - 1 - len(BOT_NAME):], end='')
                else:
                    print(output[len(input):], end='')

                previous_input = output

            if len(input) > len(output):
                # That means we have already printed out a part of the User's name. We need to take it back. So printing the carriage return
                print('\r', end='')
            else:
                print('')

            pred_answer_index = pred_index

    print(f"{USER_NAME}: ", end='')
