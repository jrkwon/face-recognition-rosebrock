# %%
from imutils import paths
from imutils.video import VideoStream
import imutils, face_recognition, cv2, os, pickle, time
import argparse
from collections import Counter
import numpy as np

# %%
def load_facial_encodings(args):
    print('[INFO] loading facial embeddings...')
    try:
        encodings = pickle.loads(open(args['encodings'], 'rb').read())    #encodings here
    except FileNotFoundError:
        print(f"File {args['encodings']} not found.")
        SystemExit

    return encodings

def recognize_faces_image(input_image_path, output_image_path, 
                          facial_encodings, detection_method='cnn'):
    print('[INFO] recognizing faces in image...')

    ti = time.time()
    # load the input image and convert it from BGR to RGB
    image = cv2.imread(input_image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect the (x,y)-coordinates of the bounding boxes corresponding to each face in the input image, then compute the facial embeddings for each face
    print('[INFO] recognizing faces...')
    boxes = face_recognition.face_locations(rgb, model=detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    # initialize the list of names for each face detected
    names = []

    for encoding in encodings:
        votes = face_recognition.compare_faces(facial_encodings['encodings'], encoding)
        if True in votes:
            names.append(Counter([name for name, vote in list(zip(facial_encodings['names'], votes)) if vote == True]).most_common()[0][0])
        else:
            names.append('Unknown')

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(names), 3), dtype="uint8")

    for ((top, right, bottom, left), name) in zip(boxes, names):
        color = [int(c) for c in COLORS[names.index(name)]]
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.imwrite(output_image_path, image)

    print('Done! \nTime taken: {:.1f} secs'.format((time.time() - ti)))    


# %%
def recognize_faces_video(input_video_path, output_video_path, 
                          facial_encodings, detection_method='cnn'):
    print('[INFO] recognizing faces in video...')
    stream = cv2.VideoCapture(input_video_path)    #test video here
    writer = None

    ti = time.time()
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        boxes = face_recognition.face_locations(rgb, model=detection_method) 
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        
        for encoding in encodings:
            votes = face_recognition.compare_faces(facial_encodings['encodings'], encoding)
            if True in votes:
                names.append(Counter([name for name, vote in list(zip(facial_encodings['names'], votes)) if vote == True]).most_common()[0][0])
            else:
                names.append('Unknown')

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(names), 3), dtype="uint8")

        for ((top, right, bottom, left), name) in zip(boxes, names):
            color = [int(c) for c in COLORS[names.index(name)]]
            top, right, bottom, left = int(top * r), int(right * r), int(bottom * r), int(left * r)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        
        if writer is None:
            writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame.shape[1], frame.shape[0]), True)
        
        writer.write(frame)
        cv2.imshow('Video file', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    stream.release()
    writer.release()
    print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti)/60))


# %%
def recognize_faces_webcam(input_webcam_num, output_video_path, 
                           facial_encodings, detection_method='cnn'):
    print('[INFO] recognizing faces in webcam...')
    vs = VideoStream(src=input_webcam_num).start()    #access webcam
    time.sleep(2.0)    #warm up webcam
    writer = None

    ti = time.time()
    while True:
        frame = vs.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        boxes = face_recognition.face_locations(rgb, model=detection_method)    #detection_method here
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        for encoding in encodings:
            votes = face_recognition.compare_faces(facial_encodings['encodings'], encoding)
            if True in votes:
                names.append(Counter([name for name, vote in list(zip(facial_encodings['names'], votes)) if vote == True]).most_common()[0][0])
            else:
                names.append('Unknown')

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(names), 3), dtype="uint8")

        for ((top, right, bottom, left), name) in zip(boxes, names):
            color = [int(c) for c in COLORS[names.index(name)]]
            top, right, bottom, left = int(top * r), int(right * r), int(bottom * r), int(left * r)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if writer is None:
            writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    vs.stop()
    writer.release()
    print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti)/60))


# %%
def construct_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--mode', required=True,
        help='image|video|webcam')
    ap.add_argument('-i', '--input', default=0,
        help='path to input image|video or webcam_num')
    ap.add_argument('-o', '--output', required=True, 
        help='path to output image filename')
    ap.add_argument('-e', '--encodings', required=True, 
        help='path to serialized db of facial encodings')
    ap.add_argument('-d', '--detection-method', type=str, default='cnn',
        help='face detection model to use: either `hog` or `cnn`')
    args = vars(ap.parse_args())
    return args
    
# %%
def main():
    args = construct_args()

    mode = args['mode']
    input = args['input']
    output = args['output']
    detection_method = args['detection_method']

    facial_encodings = load_facial_encodings(args)
    if mode == 'image':
        recognize_faces_image(input, output, facial_encodings, detection_method)
    elif mode == 'video':
        recognize_faces_video(input, output, facial_encodings, detection_method)
    elif mode == 'webcam':
        recognize_faces_webcam(int(input), output, facial_encodings, detection_method)
    else:
        print(f'{mode} is not proper mode. Use image|video|webcam.')

# %%
if __name__ == '__main__':
    main()

