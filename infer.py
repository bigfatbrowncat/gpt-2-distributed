import argparse
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
    config_name="config/device_map_single_gpu_24.json",
#    config_name="config/device_map_cpu_and_gpu_24.json",
    weights_filename="trained_models/gpt2-medium-math_19.pt"
)

print(model_wrapper.generate("Multiply two by two. The result is"))
print(model_wrapper.generate("Multiply three by two. The result is"))
print(model_wrapper.generate("Add five to six. The result is"))
print(model_wrapper.generate("Add two to 2. It is"))
print(model_wrapper.generate("Add 3 to three. The result is"))
print(model_wrapper.generate("3 multiplied by six equals"))
print(model_wrapper.generate("five multiplied by 2 equals"))
print(model_wrapper.generate("2 + 4 ="))
print(model_wrapper.generate("I am a very smart guy, I know that 3 * 7 ="))
