import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
import keras
import datetime

import pickle


class RNN_Tag_Trainer:
    filePath = "can.txt"
    train_rate = 0.08
    test_rate = 0.02

    def __init__(self):
        self.train();

    def train(self):
        sequences, sequences_tags = self.load_raw_data(self.filePath)
        train_sequences, test_sequences, train_tags, test_tags = self.generateTrainAndTestCase(sequences,
                                                                                               sequences_tags)
        word2index, tag2index = self.buildWord2IndexAndTag2Index(train_sequences, train_tags)
        train_sequences_X, test_sequences_X, train_tags_y, test_tags_y = self.generateTrainAndTestRecords(
            train_sequences, test_sequences, train_tags, test_tags, word2index, tag2index)
        self.model = self.kerasTrain(train_sequences_X, train_tags_y, word2index, tag2index)
        self.evaluateModel(self.model, test_sequences_X, test_tags_y, tag2index)
        self.word2index = word2index
        self.tag2index = tag2index
        self.persistence(self.word2index,self.tag2index,self.model)
        return

    def parseLine(self, line):
        words = line.split()
        tokens, tags = [], []
        for word in words:
            try:
                params = word.split("_")
                tokens.append(params[0])
                tags.append(params[1])
            except IndexError as e:
                e
                # dirtydata
                # print("index error, skip data")
        return (tokens, tags)

    def load_raw_data(self, filePath):
        sequences, sequences_tags = [], []
        fh = open(filePath)
        for line in fh:
            tokens, tags = self.parseLine(line)
            if (len(tokens) != 0):
                sequences.append(np.array(tokens))
                sequences_tags.append(np.array(tags))
        fh.close()
        return sequences, sequences_tags

    def generateTrainAndTestCase(self, sequences, sequences_tags):
        return train_test_split(sequences, sequences_tags, train_size=self.train_rate, test_size=self.test_rate, random_state=42,
                                shuffle=True)

    def buildWord2IndexAndTag2Index(self, train_sequences, train_tags):
        words, tags = set([]), set([])
        for s in train_sequences:
            for w in s:
                words.add(w.lower())

        for ts in train_tags:
            for t in ts:
                tags.add(t)

        word2index = {w: i + 2 for i, w in enumerate(list(words))}
        word2index['-PAD-'] = 0  # The special value used for padding
        word2index['-OOV-'] = 1  # The special value used for OOVs

        tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
        tag2index['-PAD-'] = 0  # The special value used to padding

        return (word2index, tag2index)

    def generateTrainAndTestRecords(self, train_sequences, test_sequences, train_tags, test_tags, word2index,
                                    tag2index):
        train_sequences_X, test_sequences_X, train_tags_y, test_tags_y = [], [], [], []

        for s in train_sequences:
            s_int = []
            for w in s:
                try:
                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])
            train_sequences_X.append(s_int)

        for s in test_sequences:
            s_int = []
            for w in s:
                try:
                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])
            test_sequences_X.append(s_int)

        for s in train_tags:
            try:
                train_tags_y.append([tag2index[t] for t in s])
            except:
                train_tags_y.append([0])

        for s in test_tags:
            try:
                test_tags_y.append([tag2index[t] for t in s])
            except:
                test_tags_y.append([0])

        MAX_LENGTH = len(max(train_sequences_X, key=len))
        train_sequences_X = pad_sequences(train_sequences_X, maxlen=MAX_LENGTH, padding='post')
        test_sequences_X = pad_sequences(test_sequences_X, maxlen=MAX_LENGTH, padding='post')
        train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
        test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

        return (train_sequences_X, test_sequences_X, train_tags_y, test_tags_y)

    def kerasTrain(self, train_sequences_X, train_tags_y, word2index, tag2index):
        MAX_LENGTH = len(max(train_sequences_X, key=len))
        train_sequences_X = pad_sequences(train_sequences_X, maxlen=MAX_LENGTH, padding='post')
        train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')

        model = Sequential()
        model.add(InputLayer(input_shape=(MAX_LENGTH,)))
        model.add(Embedding(len(word2index), 128))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(TimeDistributed(Dense(len(tag2index))))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(0.001),
                      metrics=['accuracy'])

        model.summary()

        logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)



        model.fit(train_sequences_X, self.to_categorical(train_tags_y, len(tag2index)), batch_size=1024, epochs=3,
                  validation_split=0.2,callbacks=[tensorboard_callback])
        return model

    def evaluateModel(self, model, test_sequences_X, test_tags_y, tag2index):
        scores = model.evaluate(np.array(test_sequences_X), self.to_categorical(test_tags_y, len(tag2index)))
        print(f"{model.metrics_names[1]}: {scores[1] * 100}")  # acc: 99.09751977804825
        return

    def to_categorical(self, sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)

    def demo(self):
        test_samples = ["king street".split(), "wayfare restauran main st 311".split(),
                        "falls cadillac pres de la thorold stone rd".split()]
        test_samples_X = []
        for s in test_samples:
            s_int = []
            for w in s:
                try:
                    s_int.append(self.word2index[w.lower()])
                except KeyError:
                    s_int.append(self.word2index['-OOV-'])
                test_samples_X.append(s_int)
        test_samples_X = pad_sequences(test_samples_X, maxlen=self.MAX_LENGTH, padding='post')
        predictions = self.model.predict(test_samples_X)
        print(self.logits_to_tokens(predictions, {i: t for t, i in self.tag2index.items()}))

    def logits_to_tokens(sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            token_sequence = []
            for categorical in categorical_sequence:
                token_sequence.append(index[np.argmax(categorical)])

            token_sequences.append(token_sequence)

        return token_sequences


    def persistence(self, word2index,tag2index,model):
        model.save("onebox_tagging.rnn")

        with open('word2index.pkl', 'wb') as f:
            pickle.dump(word2index, f, pickle.HIGHEST_PROTOCOL)

        with open('tag2index.pkl', 'wb') as f:
            pickle.dump(tag2index, f, pickle.HIGHEST_PROTOCOL)
        print("finish to persist model and index")

        return


    def logits_to_tokens(self,sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            token_sequence = []
            for categorical in categorical_sequence:
                token_sequence.append(index[np.argmax(categorical)])

            token_sequences.append(token_sequence)

        return token_sequences

    def predict(self,input):
        tokens = input.split(" ")
        test_samples_X = []
        one_hots = []
        for t in tokens:
            try:
                one_hots.append(self.word2index[t.lower()])
            except KeyError:
                one_hots.append(1)
            test_samples_X.append(one_hots)
        test_samples_X = pad_sequences(test_samples_X,maxlen=18,padding='post')
        predictions = self.model.predict(test_samples_X)

        print(self.logits_to_tokens(predictions, {i: t for t, i in self.tag2index.items()}))
        return

trainer = RNN_Tag_Trainer()
trainer.predict("king street")
print("finish predict")