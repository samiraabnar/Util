import nltk
import itertools
import csv
import numpy as np

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





