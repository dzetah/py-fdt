import sys
import spacy

from datatools import load_dataset

nlp = spacy.load('fr')

def main():
    stop_words = ' '.join(spacy.lang.fr.stop_words.STOP_WORDS)
    doc = nlp(stop_words)
    tokens = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_:
                tokens.append(token.text)
    print(tokens)

if __name__ == "__main__":
    main()
