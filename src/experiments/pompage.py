
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

VOCAB_SIZE = 50
MAX_SEQ_LENGTH = 4

def encode_doc(docs):
    encoded_docs = [one_hot(d, VOCAB_SIZE) for d in docs]
    return pad_sequences(encoded_docs, maxlen=MAX_SEQ_LENGTH, padding='post')

def create_model():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 8, input_length=MAX_SEQ_LENGTH))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

# train data
train_docs = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better.'
]
train_labels = array([1,1,1,1,1,0,0,0,0,0])

# dev data
dev_docs = [
    'Excellent work!',
    'Nice effort',
    'Incredible',
    'Bad performance',
    'Not good',
    'Could have performed better'
]
dev_labels = array([1,1,1,0,0,0])

# prepare data
X_train = encode_doc(train_docs)
X_dev = encode_doc(dev_docs)
Y_train = train_labels
Y_dev = dev_labels

# create the model
model = create_model()

# summarize the model
print(model.summary())

# fit the model
model.fit(X_train, Y_train, epochs=100, verbose=1)

# evaluate the model
loss, accuracy = model.evaluate(X_dev, Y_dev, verbose=0)
print('Accuracy: %.2f' % (accuracy * 100))
