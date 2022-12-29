import cv2
import os
import argparse
import random


DATASET_PATH = 'data'
DATASET_DIRS = ['train', 'validate', 'test']
DATASET_DIRS_DISTRIBUTION = [0.7, 0.15, 0.15]


warnings_number = 0


def print_warning(message):
    global warnings_number
    warnings_number += 1

    print('<<warning [{}]>> : {}\n'.format(warnings_number, message))


def create_model_dirs(person_name, path=DATASET_PATH):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print_warning('Directory {} already exists.'.format(path))

    person_dirs = [path + '/' + DATASET_DIRS[i] + '/' + person_name for i in range(3)]
    for dir in person_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print_warning('Directory {} already exists.'.format(dir))

    return person_dirs


parser = argparse.ArgumentParser(description='Webcam images collector script')
parser.add_argument("--name", help="The name of the person currently collecting images for", required=True)
parser.add_argument("--images", help="The number of images to collect for the current person", required=True, type=int)
args = parser.parse_args()


images_number = args.images
person_name = args.name
person_dirs = create_model_dirs(person_name)

video_capture = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

step = 0

while True:
    ret, frame = video_capture.read()
    faces = face_detect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        step = step + 1

        current_dir_path = random.choices(population=person_dirs, weights=DATASET_DIRS_DISTRIBUTION)[0]
        current_path = current_dir_path + '/' + str(step) + '.jpg'
        print("Creating Images........." + current_path)

        cv2.imwrite(current_path, frame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("WindowFrame", frame)
    cv2.waitKey(1)

    if step > images_number:
        break

video_capture.release()
cv2.destroyAllWindows()
