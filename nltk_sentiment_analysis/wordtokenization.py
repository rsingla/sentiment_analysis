import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

list_play_files = ["../nltk_sentiment_analysis/julius_caesar",
                    "../nltk_sentiment_analysis/macbeth", "../nltk_sentiment_analysis/merchantofvenice",
                   "../nltk_sentiment_analysis/romeoandjuliet", "../nltk_sentiment_analysis/tempest"]

word_dict = {}

for filepath in list_play_files:
    file = open(filepath,"r")
    count = 0
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