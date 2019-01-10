from gensim.models import KeyedVectors as kv

def main():
    # load Word2Vec model
    embeddingsFilePath = "../../data/frWac_non_lem_no_postag_no_phrase_500_skip_cut200.bin"
    model = kv.load_word2vec_format(embeddingsFilePath, binary=True, encoding='UTF-8', unicode_errors='ignore')
    syn = model.most_similar(positive=["maison"], topn=10)
    print([word[0] for word in syn])

if __name__ == "__main__":
    main()
