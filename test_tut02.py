from tensorflow import keras

# from tut02 import word_index

data = keras.datasets.imdb
word_index = data.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


model = keras.models.load_model("model.h5")

def review_encode(s):
    encoded = [1]
    for word in s:
        encoded.append(word_index.get(word, 2))
    return encoded

with open("data/text.txt", encoding="utf-8") as f:
    for line in f:
        nline = line.lower().replace(",", "")\
                .replace(".", "")\
                .replace("(", "")\
                .replace(")", "")\
                .replace(":", "")\
                .replace("\"", "")\
                .strip()\
                .split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences(
                [encode], value=word_index["<PAD>"],
                padding="post", maxlen=250)
        print(type(encode))
        print(encode)
        predict = model.predict(encode)
        print(line)
        print(predict[0])



