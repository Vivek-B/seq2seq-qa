import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils import squad
s = squad()

BEAM_WIDTH = 1
BATCH_SIZE = 128


g = tf.Graph()

with g.as_default():
	# INPUTS
	X = tf.placeholder(tf.int32, [BATCH_SIZE, None], name='X')
	X_context = tf.placeholder(tf.int32, [BATCH_SIZE, None], name='X_context')
	Y = tf.placeholder(tf.int32, [BATCH_SIZE, None], name='Y')
	X_seq_len = tf.placeholder(tf.int32, [None], name='X_seq_len')
	X_seq_len_context = tf.placeholder(tf.int32, [None], name='X_seq_len_context')
	Y_seq_len = tf.placeholder(tf.int32, [None], name='Y_seq_len')


	# ENCODER         
	encoder_out, encoder_state = tf.nn.dynamic_rnn(cell = tf.nn.rnn_cell.BasicLSTMCell(128), 
													inputs = tf.contrib.layers.embed_sequence(X, 10000, 128),
													sequence_length = X_seq_len,
													dtype = tf.float32)

	context_out, context_state = tf.nn.dynamic_rnn(cell = tf.nn.rnn_cell.BasicLSTMCell(128, reuse=True), 
													inputs = tf.contrib.layers.embed_sequence(X_context, 10000, 128),
													sequence_length = X_seq_len_context,
													dtype = tf.float32)


	# DECODER COMPONENTS
	s.vocab_size = 10000
	decoder_embedding = tf.Variable(tf.random_uniform([s.vocab_size, 128], -1.0, 1.0))
	projection_layer = Dense(s.vocab_size)


	# ATTENTION (TRAINING)
	attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = 128, 
															memory = encoder_out,
															memory_sequence_length = X_seq_len)

	decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell = tf.nn.rnn_cell.BasicLSTMCell(128),
														attention_mechanism = attention_mechanism,
														attention_layer_size = 128)


	# DECODER (TRAINING)
	training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = tf.nn.embedding_lookup(decoder_embedding, Y),
														sequence_length = Y_seq_len,
														time_major = False)

	state = tf.contrib.rnn.LSTMStateTuple(context_state[1], encoder_state[1])
	training_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,
														helper = training_helper,
														initial_state = decoder_cell.zero_state(BATCH_SIZE,tf.float32).clone(cell_state=state),
														output_layer = projection_layer)

	training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = training_decoder,
																	  impute_finished = True,
																	  maximum_iterations = tf.reduce_max(Y_seq_len))
	training_logits = training_decoder_output.rnn_output


	# BEAM SEARCH TILE
	encoder_out = tf.contrib.seq2seq.tile_batch(encoder_out, multiplier=BEAM_WIDTH)
	X_seq_len = tf.contrib.seq2seq.tile_batch(X_seq_len, multiplier=BEAM_WIDTH)
	encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=BEAM_WIDTH)


	# ATTENTION (PREDICTING)
	attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = 128, 
															memory = encoder_out,
															memory_sequence_length = X_seq_len)

	decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell = tf.nn.rnn_cell.BasicLSTMCell(128),
														attention_mechanism = attention_mechanism,
														attention_layer_size = 128)


	# DECODER (PREDICTING)
	predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = decoder_cell,
															  embedding = decoder_embedding,
															  start_tokens = tf.tile(tf.constant([5], dtype=tf.int32), [BATCH_SIZE]),
															  end_token = 6,
															  initial_state = decoder_cell.zero_state(BATCH_SIZE * BEAM_WIDTH,tf.float32).clone(cell_state=encoder_state),
															  beam_width = BEAM_WIDTH,
															  output_layer = projection_layer,
															  length_penalty_weight = 0.0)

	predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = predicting_decoder,
																		impute_finished = False,
																		maximum_iterations = 2 * tf.reduce_max(Y_seq_len))
	predicting_logits = predicting_decoder_output.predicted_ids[:, :, 0]

	print('successful')


context_inp = np.array(s.padded_context[:128], dtype=np.int32)
question_inp = np.array(s.padded_questions[:128], dtype=np.int32)
decoder_inp = np.array(s.padded_answers[:128], dtype=np.int32)

with g.as_default():
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sl = np.array(np.repeat(np.array([200]), repeats=128), dtype=np.int32)
	o = sess.run(predicting_logits, feed_dict={X:question_inp, X_context:context_inp, Y:decoder_inp, X_seq_len:sl, X_seq_len_context:sl, Y_seq_len:sl})
	o.shape
