import sys
import tensorflow as tf


class Model: 
	"minimalistic TF model for HTR"

	# model constants
	batchSize = 50
	imgSize = (128, 32)
	maxTextLen = 32

	def __init__(self, charList, useBeamSearch=False, mustRestore=False):
		"init model: add CNN, RNN and CTC and initialize TF"
		self.charList = charList
		self.useBeamSearch = useBeamSearch
		self.mustRestore = mustRestore
		self.snapID = 0

		# CNN
		self.inputImgs = tf.placeholder(tf.float32, shape=(Model.batchSize, Model.imgSize[0], Model.imgSize[1]))
		cnnOut4d = self.setupCNN(self.inputImgs)

		# RNN
		rnnOut3d = self.setupRNN(cnnOut4d)

		# CTC
		(self.loss, self.decoder) = self.setupCTC(rnnOut3d)

		# optimizer for NN parameters
		self.optimizer = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

		# initialize TF
		(self.sess, self.saver) = self.setupTF()

			
	def setupCNN(self, cnnIn3d):
		"create CNN layers and return output of these layers"
		cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

		# list of parameters for the layers
		kernelVals = [5, 5, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)

		# create layers
		pool = cnnIn4d # input to first CNN layer
		for i in range(numLayers):
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			relu = tf.nn.relu(conv)
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

		return pool


	def setupRNN(self, rnnIn4d):
		"create RNN layers and return output of these layers"
		rnnIn3d = tf.squeeze(rnnIn4d, axis=[2])

		# basic cells which is used to build RNN
		numHidden = 256
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		# stack basic cells
		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		# bidirectional RNN
		# BxTxF -> BxTx2H
		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
									
		# BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
									
		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		return tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def setupCTC(self, ctcIn3d):
		"create CTC loss and decoder and return them"
		# BxTxC -> TxBxC
		ctcIn3dTBC = tf.transpose(ctcIn3d, [1, 0, 2])
		# ground truth text as sparse tensor
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
		# calc loss for batch
		self.seqLen = tf.placeholder(tf.int32, [None])
		loss = tf.nn.ctc_loss(labels=self.gtTexts, inputs=ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True)
		# decoder: either best path decoding or beam search decoding
		if self.useBeamSearch:
			decoder = tf.nn.ctc_beam_search_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
		else:
			decoder = tf.nn.ctc_greedy_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen)
		return (tf.reduce_mean(loss), decoder)


	def setupTF(self):
		"initialize TF"
		print('Python: '+sys.version)
		print('Tensorflow: '+tf.__version__)

		sess=tf.Session() # TF session

		saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
		modelDir = '../model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		# load saved model if available
		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess,saver)


	def toSparse(self, texts):
		"put ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def fromSparse(self, ctcOutput):
		"extract texts from sparse tensor"
		# ctc returns tuple, first element is SparseTensor 
		decoded=ctcOutput[0][0] 

		# go over all indices and save mapping: batch -> values
		idxDict = { b : [] for b in range(Model.batchSize) }
		encodedLabelStrs = [[] for i in range(Model.batchSize)]
		for (idx, idx2d) in enumerate(decoded.indices):
			label = decoded.values[idx]
			batchElement = idx2d[0] # index according to [b,t]
			encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		"feed a batch into the NN to train it"
		sparse = self.toSparse(batch.gtTexts)
		(_, lossVal) = self.sess.run([self.optimizer, self.loss], { self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * Model.batchSize } )
		return lossVal


	def inferBatch(self, batch):
		"feed a batch into the NN to recngnize the texts"
		decoded = self.sess.run(self.decoder, { self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * Model.batchSize } )
		return self.fromSparse(decoded)
	

	def save(self):
		"save model to file"
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
 
