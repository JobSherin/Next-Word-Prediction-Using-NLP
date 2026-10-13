import pandas as pd
import re
from collections import defaultdict, Counter

df = pd.read_parquet('train-00000-of-00001.parquet')

texts = df['text'].dropna().astype(str)

model = defaultdict(Counter)

for text in texts:
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b',text)

    for i in range(len(words)-2):
        first_word = words[i]
        second_word = words[i+1]
        next_word = words[i+2]

        key = (first_word, second_word)

        model[key][next_word] += 1


sentence = input("Enter a sentence: ")

sentence = sentence.lower()

words = re.findall(r"\b[a-z]+\b", sentence)  #RE to extract words from the sentence

if len(words)<2:
    print('Please enter atleast 2 words')

else:
    key = (words[-2], words[-1])  #Take the last two words

    if key in model:  #Check whether the model knows these words , THIS ASKS Have I seen this pair of words in my training dataset?
        predictions = model[key].most_common(1)[0][0]
        print("Predicted word:",predictions)

    else:
            print("No predictions available for the given words.")
            