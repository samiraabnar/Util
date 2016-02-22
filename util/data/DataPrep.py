import nltk
import itertools
import csv
import numpy as np

import operator


SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

class DataPrep(object):

    def train_for_reddit_comments(filepath, vocab_size):
        unknown_token = "UNKNOWN_TOKEN"
        sentence_start_token = "SENTENCE_START"
        sentence_end_token = "SENTENCE_END"

        print("Reading CSV file at %s" %(filepath))
        sentences = []
        with open(filepath) as file:
            reader = csv.reader(file, skipinitialspace = True)
            header = next(reader)

            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

        print("Parsed %d sentences!" %(len(sentences)))

        tokenized_sents = [nltk.word_tokenize(sent) for sent in sentences]

        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sents))
        print("Found %d unique words tokens." % len(word_freq.items()))


        vocab = word_freq.most_common(vocab_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        print("Using vocabulary size %d." % vocab_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sents):
            tokenized_sents[i] = [w if w in word_to_index else unknown_token for w in sent]

        print("\nExample sentence: '%s'" % sentences[0])
        print("\nExample sentence after Pre-processing: '%s'" % tokenized_sents[0])

        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sents])
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sents])

        return X_train,y_train


    def load_data_reditcomment(filename="data/reddit-comments-2015-08.csv", vocabulary_size=2000, min_sent_characters=0):

        word_to_index = []
        index_to_word = []

        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        print("Reading CSV file...")
        with open(filename, 'rt') as f:
            reader = csv.reader(f, skipinitialspace=True)
            header = next(reader)
            # Split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
            # Filter sentences
            sentences = [s for s in sentences if len(s) >= min_sent_characters]
            sentences = [s for s in sentences if "http" not in s]
            # Append SENTENCE_START and SENTENCE_END
            sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
        print("Parsed %d sentences." % (len(sentences)))

        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size-2]
        print("Using vocabulary size %d." % vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
        index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

        onehot_x = [ np.eye(vocabulary_size)[X_train[i]] for i in np.arange(X_train.shape[0])]
        onehot_y = [ np.eye(vocabulary_size)[y_train[i]] for i in np.arange(y_train.shape[0])]


        return onehot_x, onehot_y, word_to_index, index_to_word

    def load_one_hot_sentiment_data(filename,vocabulary_count=0, start=0, end=-1):

        with open(filename,'r') as filedata:
            data = filedata.readlines()[start:end]

        labelSet = set()
        tokenized_sentences_with_labels = []
        tokenized_sentences = []
        for sent in data:
            tokenized = nltk.word_tokenize(sent.lower())
            tokenized_sentences_with_labels.append((int(tokenized[0]),tokenized[1:]))
            tokenized_sentences.append(tokenized[1:])
            labelSet.add(int(tokenized[0]))

        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))

        if vocabulary_count == 0:
            vocabulary_count = len(word_freq.items())

        vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_count]
        print("Using vocabulary size %d." % vocabulary_count)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
        index_to_word = [UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        one_hot_sentences = []
        one_hot_sentiments = []
        label_count = max(labelSet)+1
        one_hot_labels = np.eye(label_count,dtype=np.float32 )
        one_hot_vocab = np.eye(vocabulary_count+1,dtype=np.float32)
        for sent in tokenized_sentences_with_labels:
            label = one_hot_labels[sent[0]]
            sentence = [one_hot_vocab[word_to_index[term]] for term in sent[1]]
            one_hot_sentences.append(sentence)
            one_hot_sentiments.append(label)




        return one_hot_sentences, one_hot_sentiments, word_to_index, index_to_word, label_count





