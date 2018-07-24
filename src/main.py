import sys
import argparse
import cv2
from DataLoader import DataLoader, Batch
from Model import Model
from SamplePreprocessor import preprocess


# filenames and paths to data
fnCharList = '../model/charList.txt'
fnAccuracy = '../model/accuracy.txt'
fnTrain = '../data/'
fnInfer = '../data/test.png'
useBeamSearch = False


def train(filePath):
	"train NN"
	# load training data
	loader = DataLoader(filePath, Model.batchSize, Model.imgSize, Model.maxTextLen)

	# create TF model
	model = Model(loader.charList, useBeamSearch)

	# save characters of model for inference mode
	open(fnCharList, 'w').write(str().join(loader.charList))

	# train forever
	epoch = 0 # number of training epochs since start
	bestAccuracy = 0.0 # best valdiation accuracy
	noImprovementSince = 0 # number of epochs no improvement of accuracy occured
	earlyStopping = 3 # stop training after this number of epochs without improvement
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		print('Validate NN')
		loader.validationSet()
		numOK = 0
		numTotal = 0
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			print('Batch:', iterInfo[0],'/', iterInfo[1])
			batch = loader.getNext()
			recognized = model.inferBatch(batch)
			
			print('Ground truth -> Recognized')	
			for i in range(len(recognized)):
				isOK = batch.gtTexts[i] == recognized[i]
				print('[OK]' if isOK else '[ERR]','"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
				numOK += 1 if isOK else 0
				numTotal +=1
		
		# print validation result
		accuracy = numOK / numTotal
		print('Correctly recognized words:', accuracy * 100.0, '%')
		
		# if best validation accuracy so far, save model parameters
		if accuracy > bestAccuracy:
			print('Accuracy improved, save model')
			bestAccuracy = accuracy
			noImprovementSince = 0
			model.save()
			open(fnAccuracy, 'w').write('Validation accuracy of saved model: '+str(accuracy))
		else:
			print('Accuracy not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(filePath):
	"validate NN"
	# load training data
	loader = DataLoader(filePath, Model.batchSize, Model.imgSize, Model.maxTextLen)

	# create TF model
	model = Model(loader.charList, useBeamSearch)

	# save characters of model for inference mode
	open(fnCharList, 'w').write(str().join(loader.charList))

	print('Validate NN')
	loader.validationSet()
	numOK = 0
	numTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		recognized = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			isOK = batch.gtTexts[i] == recognized[i]
			print('[OK]' if isOK else '[ERR]','"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
			numOK += 1 if isOK else 0
			numTotal +=1
	
	# print validation result
	accuracy = numOK / numTotal
	print('Correctly recognized words:', accuracy * 100.0, '%')


def infer(filePath):
	"recognize text in image provided by file path"
	model = Model(open(fnCharList).read(), useBeamSearch, mustRestore=True)
	img = preprocess(cv2.imread(fnInfer, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img] * Model.batchSize)
	recognized = model.inferBatch(batch)
	print('Recognized:', '"' + recognized[0] + '"')


if __name__ == '__main__':
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--validate", help="validate the NN", action="store_true")
	parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
	args = parser.parse_args()

	# use beam search (better accuracy, but slower) instead of best path decoding
	if args.beamsearch:
		useBeamSearch = True
	
	# train or validate NN, or infer text on the text image
	if args.train:
		train(fnTrain)
	elif args.validate:
		validate(fnTrain)
	else:
		infer(fnInfer)

