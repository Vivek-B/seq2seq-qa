import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np

from numpy import array, argmax

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed, Input
from keras.layers import dot, Concatenate, Activation, Bidirectional
from keras.callbacks import ModelCheckpoint

from sklearn.cross_validation import train_test_split


class squad(object):
	def __init__(self, max_seqlen_context = 2000, max_seqlen_question = 125, max_seqlen_answer = 100):

		with open('data/dev.context', 'r') as f:
			self.context = f.read().split('\n')
			self.context = ['sss '+i+' eee' for i in self.context]

		with open('data/dev.question', 'r') as f:
			self.questions = f.read().split('\n')
			self.questions = ['sss '+i+' eee' for i in self.questions]

		with open('data/dev.answer', 'r') as f:
			self.answers = f.read().split('\n')
			self.answers = ['sss '+i+' eee' for i in self.answers]

		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(self.context+self.questions+self.answers)
		self.word_index = tokenizer.word_index
		self.index_word = {}
		for i in self.word_index.keys():
			self.index_word[self.word_index[i]]=i
		self.vocab_size = len(self.word_index) + 1

		self.encoded_context = tokenizer.texts_to_sequences(self.context)
		self.encoded_questions = tokenizer.texts_to_sequences(self.questions)
		self.encoded_answers = tokenizer.texts_to_sequences(self.answers)

		self.padded_context = pad_sequences(self.encoded_context, maxlen=max_seqlen_context, padding='post')
		self.padded_questions = pad_sequences(self.encoded_questions, maxlen=max_seqlen_question, padding='post')
		self.padded_answers = pad_sequences(self.encoded_answers, maxlen=max_seqlen_answer, padding='post')

	def load_data(self):
		print('Hi')
		return(self.context, self.questions, self.answers)



