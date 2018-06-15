import sys
import cv2
from DataLoader import DataLoader, Batch
from Model import Model
from SamplePreprocessor import preprocess


# filenames and paths to data
fnCharList = '../model/charList.txt'
fnTrain = '../data/'
fnInfer = '../data/test.png'


def train(filePath):
	"train NN"
	# load training data
	loader = DataLoader(filePath, Model.batchSize, Model.imgSize, Model.maxTextLen)

	# create TF model
	model = Model(loader.charList)

	# save characters of model for inference mode
	open(fnCharList, 'w').write(str().join(loader.charList))

	# train forever
	epoch = 0 # number of training epochs since start
	bestAccuracy = 0.0 # best valdiation accuracy
	noImprovementSince = 0 # number of epochs no improvement of accuracy occured
	earlyStopping = 3 # stop training after this number of epochs without improvement
	while True:
		print('Epoch:', epoch)
		epoch += 1

		# train
		print('Train NN')
		loader.trainSet()
		loader.shuffle()
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
			loss = model.trainBatch(batch)
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
		
		# if best validation accuracy so far, save model parameters (>= instead of > to avoid stopping while accuracy is at 0% for the first epochs)
		if accuracy >= bestAccuracy:
			print('Accuracy improved, save model')
			bestAccuracy = accuracy
			noImprovementSince = 0
			model.save()
		else:
			print('Accuracy not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def infer(filePath):
	"recognize text in image provided by file path"
	model = Model(open(fnCharList).read(), mustRestore=True)
	img = preprocess(cv2.imread(fnInfer, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img] * Model.batchSize)
	recognized = model.inferBatch(batch)
	print('Recognized:', '"' + recognized[0] + '"')


if __name__ == '__main__':
	if len(sys.argv) == 2 and sys.argv[1] == 'train':
		train(fnTrain)
	else:
		infer(fnInfer)
