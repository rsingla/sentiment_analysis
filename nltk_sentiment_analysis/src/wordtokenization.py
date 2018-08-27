import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

base_path = "../../nltk_sentiment_analysis/resources/"

list_play_files = ['julius_caesar',"macbeth", "merchantofvenice","romeoandjuliet", "tempest"]

word_dict = {}

for playname in list_play_files:
    file = open(base_path+playname,"r")
    name = file.name
    completeLine = []
    for line in file:
        if len(line) > 0:
            completeLine = completeLine + word_tokenize(line)
    file.close()
    word_dict[name] = completeLine

for key,value in  word_dict.items():
    print(key)
    print(value)
    print(len(value))
