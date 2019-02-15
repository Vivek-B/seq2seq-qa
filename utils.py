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



# tokenizer = Tokenizer(num_words=10000)
# tokenizer.fit_on_texts(questions+answers)

# vocab_size = len(tokenizer.word_index)
# print (vocab_size)

# max_seq_length = max(len(line.split()) for line in questions+answers)
# max_seq_length = 50
# print(max_seq_length)
# embedding_size = 100

# encoded_questions = tokenizer.texts_to_sequences(questions)
# encoded_answers = tokenizer.texts_to_sequences(answers)

# padded_questions = pad_sequences(encoded_questions, maxlen=max_seq_length, padding='post')
# padded_answers = pad_sequences(encoded_answers, maxlen=max_seq_length, padding='post')

# def attention():
# 	# Embedding
# 	embedding_encoder = variable_scope.get_variable("embedding_encoder", [src_vocab_size, embedding_size], ...)
# 	# Look up embedding:
# 	#   encoder_inputs: [max_time, batch_size]
# 	#   encoder_emb_inp: [max_time, batch_size, embedding_size]
# 	encoder_emb_inp = embedding_ops.embedding_lookup(embedding_encoder, encoder_inputs)

# 	# Build RNN cell
# 	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# 	# Run Dynamic RNN
# 	#   encoder_outputs: [max_time, batch_size, num_units]
# 	#   encoder_state: [batch_size, num_units]
# 	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, 
# 													   encoder_emb_inp,
# 													   sequence_length=source_sequence_length, 
# 													   time_major=True)

# 	# # Build RNN cell
# 	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# 	# attention_states: [batch_size, max_time, num_units]
# 	attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

# 	# Create an attention mechanism
# 	attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, 
# 															attention_states,
# 															memory_sequence_length=source_sequence_length)

# 	decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
# 													   attention_mechanism,
# 													   attention_layer_size=num_units)

# 	# Helper
# 	helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True)

# 	# Decoder
# 	decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,output_layer=projection_layer)

# 	# Dynamic decoding
# 	outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
# 	logits = outputs.rnn_output

# 	projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)

# 	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
# 	train_loss = (tf.reduce_sum(crossent * target_weights)/batch_size)

# 	# Calculate and clip gradients
# 	params = tf.trainable_variables()
# 	gradients = tf.gradients(train_loss, params)
# 	clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

# 	# Optimization
# 	optimizer = tf.train.AdamOptimizer(learning_rate)
# 	update_step = optimizer.apply_gradients(zip(clipped_gradients, params))


# 	sess = tf.Session()
# 	sess.run(tf.global_variables_initializer())

# 	ntr_order = np.arange(n_tr)
# 	for epoch in epochs:
# 		np.random.shuffle(ntr_order)
# 		for j in range(steps_epoch):                                           # Iterate over steps per epoch
# 			start = j*batch_size                                               # Start index for the data
# 			end = min((j+1)*batch_size, n_tr)                                  # End index for the data
# 			X_mlp = x_context_train[ntr_order[start:end]]                      # Input batch for MLP
# 			xcs = np.zeros((X_mlp.shape[0], lstm_dim))                         # Input batch for cell state
# 			X_lstm = x_features_train[ntr_order[start:end]]                    # Input batch for LSTM
# 			Y = y_train[ntr_order[start:end]]                                  # Target batch

# 			sess.run(train, feed_dict={x_mlp:X_mlp, current_state:xcs, x_lstm:X_lstm, target:Y, lr_rate:lr})  








# limit = tf.sqrt(6 / (200. + s.vocab_size))
# w = tf.Variable(tf.random_uniform(maxval=limit, minval=-limit, shape=[lstm_size, s.vocab_size]),name='W',dtype=tf.float32)
# b = tf.Variable(tf.zeros(shape=[s.vocab_size]),name='B',dtype=tf.float32)

# def fc_layer(inp):
# 	act = tf.matmul(inp,w)+b
# 	return(act)