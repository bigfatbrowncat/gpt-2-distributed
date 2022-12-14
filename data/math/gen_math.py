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
    'ten'
]

with open("math.csv", "w") as f:
    for k in range(40):
        f.write(f"= means equals\n")
        f.write(f"* means multiply\n")
        f.write(f"+ means add\n")

        for i in range(10):
            f.write(f"{i+1} is {nums[i+1]}\n")
            f.write(f"{i+1} equals {nums[i+1]}\n")
            f.write(f"{nums[i+1]} is {i+1}\n")
            f.write(f"{nums[i+1]} equals {i+1}\n")

        for i in range(10):
            for j in range(10):
                f.write(f"{i+1} + {j+1} = {(i+1) + (j+1)}\n")
                f.write(f"{nums[i+1]} + {j+1} = {(i+1) + (j+1)}\n")
                f.write(f"{i+1} + {nums[j+1]} = {(i+1) + (j+1)}\n")
                f.write(f"{nums[i+1]} plus {nums[j+1]} equals {(i+1) + (j+1)}\n")
                f.write(f"{nums[i+1]} plus {nums[j+1]} is {(i+1) + (j+1)}\n")
                f.write(f"Add {nums[i+1]} to {nums[j+1]}. The result is {(i+1) + (j+1)}.\n")
                f.write(f"Add {nums[i+1]} to {nums[j+1]}. It is {(i+1) + (j+1)}.\n")
                f.write(f"Sum of {nums[i+1]} and {nums[j+1]} is {(i+1) + (j+1)}.\n")
                f.write(f"Sum of {nums[i+1]} and {nums[j+1]} equals {(i+1) + (j+1)}.\n")
                f.write(f"Sum of {nums[i+1]} with {nums[j+1]} equals {(i+1) + (j+1)}.\n")

                f.write(f"{i+1} * {j+1} = {(i+1) * (j+1)}\n")
                f.write(f"{nums[i+1]} * {j+1} = {(i+1) * (j+1)}\n")
                f.write(f"{i+1} * {nums[j+1]} = {(i+1) * (j+1)}\n")
                f.write(f"{nums[i+1]} multiplied by {nums[j+1]} equals {(i+1) * (j+1)}\n")
                f.write(f"{nums[i+1]} multiplied by {nums[j+1]} is {(i+1) * (j+1)}\n")
                f.write(f"Multiply {nums[i+1]} by {nums[j+1]}. The result is {(i+1) * (j+1)}.\n")
                f.write(f"Multiply {nums[i+1]} by {nums[j+1]}. It is {(i+1) * (j+1)}.\n")
                f.write(f"Product of {nums[i+1]} and {nums[j+1]} is {(i+1) * (j+1)}.\n")
                f.write(f"Product of {nums[i+1]} and {nums[j+1]} equals {(i+1) * (j+1)}.\n")
                f.write(f"Product of {nums[i+1]} by {nums[j+1]} equals {(i+1) * (j+1)}.\n")
