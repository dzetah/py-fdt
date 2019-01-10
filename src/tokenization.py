import sys
import spacy

from datatools import load_dataset

nlp = spacy.load('fr')

def main():
    print(spacy.lang.fr.stop_words.STOP_WORDS)
    sys.exit(1)
    devfile = load_dataset('../data/frdataset1_dev.csv')
    for text in devfile['text']:
        doc = nlp(text)
        tokens = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ not in ["PUNCT", "NUM"] and token.is_stop is not True:
                    tokens.append(token.lemma_.strip().lower())
        print(tokens)

if __name__ == "__main__":
    main()
