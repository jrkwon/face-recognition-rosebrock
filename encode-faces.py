# %% [markdown]
# # Encode the faces

# %%
#import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import time
import argparse

# %% [markdown]
# # Create facial embeddings

# %%
def create_facial_embeddings(args):
    # grab the paths to the input images in our dataset
    print('[INFO] quantifying faces...')
    image_paths = list(paths.list_images(args['dataset']))

    # initialize the list of known encodings and known names
    known_encodings = []
    known_names = []
    for (i, image_path) in enumerate(image_paths):
        print(i, image_path)

    # OpenCV orders color channels in BGR, but the dlib actually expects RGB. The face_recognition module uses dlib, so we need to swap color spaces and name the new image rgb
    ti = time.time()
    print('[INFO] processing image...')
    
    # loop over the image paths
    for (i, image_path) in enumerate(image_paths):
        # extract the person name from the image path
        print('{}/{}'.format(i+1, len(image_paths)), end=', ')
        name = image_path.split(os.path.sep)[-2]
        # load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x,y)-coordinates of the bounding boxes corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,  model=args['detection_method'])
        # compute the facial embedding for the face, ie, to turn the bounding boxes of the face into a list of 128 numbers
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and encodings
            known_encodings.append(encoding)
            known_names.append(name)
    print('Done!')
    print('Time taken: {:.1f} minutes'.format((time.time() - ti)/60))

    # dump the names and encodings to disk for future recall
    # encodings.pickle contains the 128-d face embeddings for each face in our dataset
    print('[INFO] serializing encodings...')
    data = {'encodings': known_encodings, 'names': known_names}
    f = open(args['encodings'], 'wb')
    f.write(pickle.dumps(data))
    f.close()
    print('Done!')

# %%

def construct_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--dataset', required=True, 
        help='path to input directory of faces + images')
    ap.add_argument('-e', '--encodings', required=True, 
        help='path to serialized db of facial encodings')
    ap.add_argument('-d', '--detection-method', type=str, default='cnn',
        help='face detection model to use: either `hog` or `cnn`')
    args = vars(ap.parse_args())
    return args
    
# %%

def main():
    args = construct_args()
    create_facial_embeddings(args)

# %%
if __name__ == '__main__':
    main()

