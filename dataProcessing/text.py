def train_word2vec_model(iter_sent, path_save_model):
    import gensim

    model = gensim.models.Word2Vec(iter_sent, size=100, window=5, min_count=5, workers=4, iter=10)
    model.save('/tmp/mymodel')

    return model
