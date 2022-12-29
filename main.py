import argparse
import numpy as np
import skimage.transform

from PIL import Image
from model import FaceRecognitionPipeline
from model import IMAGE_SHAPE


DEFAULT_EPOCHS_NO = 10


def test_model(pipeline, path, person_name):
    image = np.array(Image.open(path)).astype(np.float64)
    image = skimage.transform.resize(image, IMAGE_SHAPE)

    # Match the name using the model
    predicted_name, probability = pipeline.match_name(image)
    print('Prediction should be: {}\nResult is: {}, with probability = {}\n'.format(person_name, predicted_name, probability))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Webcam images collector script')
    parser.add_argument("--epochs", help="The number of epochs used in training", type=int)
    parser.add_argument("--train_path", help="The training data directory path")
    parser.add_argument("--validate_path", help="The validation data directory path")
    parser.add_argument("--test_path", help="The test data directory path")

    args = parser.parse_args()

    epochs = args.epochs if args.epochs else DEFAULT_EPOCHS_NO
    train_data_dir = args.train_path if args.train_path else 'data/train'
    validation_data_dir = args.validate_path if args.validate_path else 'data/validate'
    test_data_dir = args.test_path if args.test_path else 'data/test'

    pipeline = FaceRecognitionPipeline(train_data_dir, validation_data_dir)
    pipeline.train_model(epochs)
    test_loss, test_acc = pipeline.evaluate_model(test_data_dir)

    test_model(pipeline, 'data/extra_tests/emanuel/1.jpg', 'Emanuel')
    test_model(pipeline, 'data/extra_tests/simona/1.jpg', 'Simona')
    test_model(pipeline, 'data/extra_tests/simona/2.jpg', 'Simona')
