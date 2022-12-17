import random
import sys

nums = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve',
    'thirteen',
    'fourteen',
    'fifteen',
    'sixteen',
    'seventeen',
    'eighteen',
    'nineteen',
    'twenty'
]

def gen_math(file):
    rnd = random.Random()

    BOT_NAME = "AI"
    USER_NAME = "User"

    with open(file, "a") as f:
        f.write(f"= means equals\n")
        f.write(f"* means multiply\n")
        f.write(f"+ means add\n")

        for i in range(20):
            # Digital to textual (and back) conversion
            f.write(f"{i+1} is {nums[i+1]}\n")
            f.write(f"{i+1} equals {nums[i+1]}\n")
            f.write(f"{nums[i+1]} is {i+1}\n")
            f.write(f"{nums[i+1]} equals {i+1}\n")
            f.write(f"{USER_NAME}: How much is {i + 1}?<br/>{BOT_NAME}: {(i + 1)}\n")
            f.write(f"{USER_NAME}: How much is {i + 1}?<br/>{BOT_NAME}: It is {(i + 1)}\n")
            f.write(f"{USER_NAME}: How much is {nums[i+1]}?<br/>{BOT_NAME}: {(i + 1)}\n")

        for i in range(20):
            for j in range(20):

                # Summing

                true_fact = [
                    "",
                    "",
                    "it is an example of a true mathematical fact that ",
                    "it is true that ",
                    "it is right that ",
                    "it is a right statement that ",
                    "undoubtedly, ",
                    "truely, ",
                ]


                f.write(f"{rnd.choice(true_fact)}{i+1} + {j+1} = {(i+1) + (j+1)}\n")
                f.write(f"{rnd.choice(true_fact)}{i+1} + {j+1} = {(i+1) + (j+1)}\n")
                f.write(f"{rnd.choice(true_fact)}{nums[i+1]} + {j+1} = {(i+1) + (j+1)}\n")
                f.write(f"{rnd.choice(true_fact)}{i+1} + {nums[j+1]} = {(i+1) + (j+1)}\n")
                f.write(f"{rnd.choice(true_fact)}{nums[i+1]} + {j+1} = {(i+1) + (j+1)}\n")
                f.write(f"{rnd.choice(true_fact)}{i+1} + {nums[j+1]} = {(i+1) + (j+1)}\n")

                f.write(f"{rnd.choice(true_fact)}{nums[i+1]} plus {nums[j+1]} equals {(i+1) + (j+1)}\n")
                f.write(f"{rnd.choice(true_fact)}{nums[i+1]} plus {nums[j+1]} is {(i+1) + (j+1)}\n")
                f.write(f"Add {nums[i+1]} to {nums[j+1]}. The result is {(i+1) + (j+1)}.\n")
                f.write(f"Add {nums[i+1]} to {nums[j+1]}. It is {(i+1) + (j+1)}.\n")
                f.write(f"Sum of {nums[i+1]} and {nums[j+1]} is {(i+1) + (j+1)}.\n")
                f.write(f"Sum of {nums[i+1]} and {nums[j+1]} equals {(i+1) + (j+1)}.\n")
                f.write(f"Sum of {nums[i+1]} with {nums[j+1]} equals {(i+1) + (j+1)}.\n")

                f.write(f"{USER_NAME}: How much is {i+1} + {j+1}?<br/>{BOT_NAME}: {(i+1) + (j+1)}\n")
                f.write(f"{USER_NAME}: How many is {i+1} + {j+1}?<br/>{BOT_NAME}: {(i+1) + (j+1)}\n")
                f.write(f"{USER_NAME}: How much is {i+1} plus {j+1}?<br/>{BOT_NAME}: {(i+1) + (j+1)}\n")
                f.write(f"{USER_NAME}: How many is {i+1} plus {j+1}?<br/>{BOT_NAME}: {(i+1) + (j+1)}\n")

                f.write(f"{USER_NAME}: Is it right that {i+1} + {j+1} = {(i+1) + (j+1)}?<br/>{BOT_NAME}: Yes, it is right.\n")
                f.write(f"{USER_NAME}: Is it true that {i+1} + {j+1} = {(i+1) + (j+1)}?<br/>{BOT_NAME}: Yes, precisely.\n")

                f.write(f"{USER_NAME}: Is it right that {i+1} + {j+1} = {(i+1) + (j+1) + 1}?<br/>{BOT_NAME}: No, it is wrong.\n")
                f.write(f"{USER_NAME}: Is it right that {i+1} + {j+1} = {(i+1) + (j+1) + 2}?<br/>{BOT_NAME}: No, not right.\n")
                f.write(f"{USER_NAME}: Is it true that {i+1} + {j+1} = {(i+1) + (j+1) - 2}?<br/>{BOT_NAME}: No, it is false.\n")
                f.write(f"{USER_NAME}: Is it right that {i+1} + {j+1} = {(i+1) + (j+1) + 10}?<br/>{BOT_NAME}: No, it is not.\n")

                # Multiplication

                f.write(f"{rnd.choice(true_fact)}{i+1} * {j+1} = {(i+1) * (j+1)}\n")
                f.write(f"{rnd.choice(true_fact)}{nums[i+1]} * {j+1} = {(i+1) * (j+1)}\n")
                f.write(f"{rnd.choice(true_fact)}{i+1} * {nums[j+1]} = {(i+1) * (j+1)}\n")
                f.write(f"{rnd.choice(true_fact)}{nums[i+1]} multiplied by {nums[j+1]} equals {(i+1) * (j+1)}\n")
                f.write(f"{nums[i+1]} multiplied by {nums[j+1]} is {(i+1) * (j+1)}\n")
                f.write(f"Multiply {nums[i+1]} by {nums[j+1]}. The result is {(i+1) * (j+1)}.\n")
                f.write(f"Multiply {nums[i+1]} by {nums[j+1]}. It is {(i+1) * (j+1)}.\n")
                f.write(f"Product of {nums[i+1]} and {nums[j+1]} is {(i+1) * (j+1)}.\n")
                f.write(f"Product of {nums[i+1]} and {nums[j+1]} equals {(i+1) * (j+1)}.\n")
                f.write(f"Product of {nums[i+1]} by {nums[j+1]} equals {(i+1) * (j+1)}.\n")

                f.write(f"{USER_NAME}: How much is {i+1} * {j+1}?<br/>{BOT_NAME}: {(i+1) * (j+1)}\n")
                f.write(f"{USER_NAME}: How many is {i+1} * {j+1}?<br/>{BOT_NAME}: {(i+1) * (j+1)}\n")
                f.write(f"{USER_NAME}: How much is {i+1} multiplied by {j+1}?<br/>{BOT_NAME}: {(i+1) * (j+1)}\n")
                f.write(f"{USER_NAME}: How many is {i+1} multiplied by {j+1}?<br/>{BOT_NAME}: {(i+1) * (j+1)}\n")

                f.write(f"{USER_NAME}: Is it right that {i+1} * {j+1} = {(i+1) * (j+1)}?<br/>{BOT_NAME}: Yes, it is right.\n")
                f.write(f"{USER_NAME}: Is it true that {i+1} * {j+1} = {(i+1) * (j+1)}?<br/>{BOT_NAME}: Yes, precisely.\n")
                f.write(f"{USER_NAME}: Is it right that {i+1} * {j+1} = {(i+1) * (j+1) + 1}?<br/>{BOT_NAME}: No, it is wrong.\n")
                f.write(f"{USER_NAME}: Is it right that {i+1} * {j+1} = {(i+1) * (j+1) * 2}?<br/>{BOT_NAME}: No, not at all.\n")
                f.write(f"{USER_NAME}: Is it true that {i+1} * {j+1} = {(i+1) * (j+1) - 2}?<br/>{BOT_NAME}: No, it is false.\n")
                f.write(f"{USER_NAME}: Is it right that {i+1} * {j+1} = {(i+1) * (j+1) + 10}?<br/>{BOT_NAME}: No, it is not.\n")

        # Double calculations explanation
        for i in range(20):
            for j in range(20):
                for k in range(20):
                    f.write(f"{i+1} + {j+1} + {k+1} is the same as sum {i+1} with {j+1} and then add {k+1} to the result.\n")
                    f.write(f"To calculate how much is {i+1} + {j+1} + {k+1}, add {i+1} to {j+1} and then add {k+1} to the sum.\n")


MATH_CSV = "math.csv"
# Clearing the file
with open(MATH_CSV, "w") as f:
    f.write('')

gen_math(MATH_CSV)
