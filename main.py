import argparse
import json
from typing import Tuple, List
import os

import cv2 
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
from autocorrect import Speller


#print("Current Directory:", current_directory)
class FilePaths:
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    """Filenames and paths to data."""
    fn_char_list = os.path.join(current_directory, 'model', 'charList.txt')
    fn_summary = os.path.join(current_directory, 'model', 'summary.json')
    fn_corpus = os.path.join(current_directory, 'data', 'corpus.txt')


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def infer(model: Model, folder_path: Path) -> None:
    """Recognizes text in images provided by a folder path."""
    # Retrieve the list of filenames in the folder
    filenames = os.listdir(folder_path)
    recognized_texts = []
    #autocorrect
    spell = Speller()
    for filename in filenames:
        # Check if the file has a valid image extension
        if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            continue

        # Create the full file path
        file_path = folder_path / filename

        # Load the image using OpenCV
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        assert img is not None

        # Preprocess the image
        preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
        img = preprocessor.process_img(img)

        # Create a batch containing the image
        batch = Batch([img], None, 1)

        # Perform inference on the batch
        recognized, probability = model.infer_batch(batch, True)
        if probability[0] < 0.75:
            recognized_texts.append(spell(recognized[0]))
        else:
            recognized_texts.append(recognized[0])

        # Print the recognized text and probability
        print(f'Image: {file_path}')
        print(f'Recognized: "{recognized[0]}"')
        print(f'Probability: {probability[0]}')
        
        transcribed_sentence = ' '.join(recognized_texts)

    #saves the sentence to a file
    with open('./transcribed.txt', 'w') as f:
        f.write(transcribed_sentence)
    print(f'Transcribed Sentence: {transcribed_sentence}')
    #removes all images     
    for filename in os.listdir(folder_path):
        file_path = folder_path / filename
        if os.path.isfile(file_path):
            os.remove(file_path)
    


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()


def main():
    """Main function."""

    # parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # infer text on test image
    if args.mode == 'infer':
        word_images_folder = Path('./output')  
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        infer(model, word_images_folder)
    

if __name__ == '__main__':
    main()