from data.math.gen_math import gen_math
from data.words.gen_words import gen_words

filename = "auto_gen.csv"

# Clearing the file
with open(filename, "w") as f:
    f.write('')

gen_math(filename)
gen_words(filename)
