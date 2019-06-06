from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

class ModelPredict:

    MAX_LENGTH = 18

    def loadModel(self):
        self.loadWord2IndexAndTag2Index();
        return

    word2IndexPath = "D:\\hlchen\\CodeHub\Learning2Rank\\train_result\\word2Index.txt"
    tag2IndexPath = "D:\\hlchen\\CodeHub\\Learning2Rank\\train_result\\tag2Index.txt"
    modelPath = "D:\\hlchen\\CodeHub\\Learning2Rank\\train_result\\rnn_onebox_tag.model"

    def loadWord2IndexAndTag2Index(self):



        with open('word2index.pkl', 'rb') as f:
            self.word2index =  pickle.load(f)

        with open('tag2index.pkl', 'rb') as f:
            self.tag2index = pickle.load(f)


        # model = Sequential()
        # model.add(InputLayer(input_shape=(18,)))
        # model.add(Embedding(len(self.word2index), 128))
        # model.add(Bidirectional(LSTM(256, return_sequences=True)))
        # model.add(TimeDistributed(Dense(len(self.tag2index))))
        # model.add(Activation('softmax'))

        self.model = load_model("onebox_tagging.rnn")

        return

    def predict(self, input):
        tokens = input.split(" ")
        test_samples_X =[]
        one_hots = []
        for token in tokens:
            try:
                one_hots.append(self.word2index[token.lower()])
            except KeyError:
                one_hots.append(1)
        test_samples_X.append(one_hots)
        test_samples_X = pad_sequences(test_samples_X, maxlen=18, padding='post')
        predictions = self.model.predict(test_samples_X)

        print(self.logits_to_tokens(predictions, {i: t for t, i in self.tag2index.items()}))
        return

    def logits_to_tokens(self, sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            token_sequence = []
            for categorical in categorical_sequence:
                token_sequence.append(index[np.argmax(categorical)])

            token_sequences.append(token_sequence)

        return token_sequences

predictor = ModelPredict()
predictor.loadModel()
predictor.predict("wayfare restauran main st 311")

print("  ")