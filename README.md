# Face Recognition

This repo was forked from https://github.com/JNYH/Face_Recognition.  
The tutorial is originally from - https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/

![image](output_images_jurassic_park/test-1.jpg)

## Environment

```bash
$ conda env create -f environment.yaml
```

Make sure that this conda env is activated. 
```bash
$ conda activate face-recognition-rosebrock
```


## Install Packages

### face_recogntion
https://anaconda.org/conda-forge/face_recognition
```bash
$ conda install -c conda-forge face_recognition
```

### Replace `dlib` with `dlib-cuda (GPU support)`
The `face_recognition` package inclues `dlib`. Yet, this `dlib` does not support GPU. You may compile and build `dlib` from source code to make it support GPU. I found a `conda` package from `zeroae` conda channel. We can replace the `dlib` (CPU only) with `dlib-cuda` (GPU support).

```bash
$ conda install -c zeroae dlib-cuda 
```

### Install `imutils`
```bash
$ pip install imutils
```

## Folder Structure

### Data Files for Facial Features Encodings
```
- dataset_jurassic_park/ # jurassic park main characters
- dataset_webacm/ # photos for webcam live face recognition
```

### Encodings
```
- concodings/encodings_jurassic_park.pickle
- concodings/encodings_webcam.pickle
```

### Test Input Images and Videos
```
- input_images_jurassic_park
- input_videos_jurassic_park
```

### Output Folders
```
- output_images_jurassic_park
- output_videos_jurassic_park
- output_webcam
```

## Script

Activate the conda env first. 
```bash
$ conda activate face-recognition-rosebrock
```

### Package Installation Jupyter Notebook
You may not need to use this if you create a conda environment using `$ conda env create -f environment.yaml`

```
- install-packages.ipynb
```
### Encode Faces for Dataset
```bash
(face-recognition-rosebrock) $ python encode-faces.py -i dataset_jurassic_park -e encodings/jurassic_park.pickle
```

### Encode Faces for Webcam
```bash
(face-recognition-rosebrock) $ python encode-faces.py -i dataset_webcam -e encodings/webcam.pickle
```

### Face Recognition (Image)
```bash
(face-recognition-rosebrock) $ python face-recognition.py -m image -i input_images_jurassic_park/test-1.jpg -o output_images_jurassic_park/test-1.jpg -e encodings/jurassic_park.pickle
```

### Face Recognition (Video)
```bash
(face-recognition-rosebrock) $ python face-recognition.py -m video -i input_videos_jurassic_park/test-1.mp4 -o output_videos_jurassic_park/test-1.avi -e encodings/jurassic_park.pickle
```

### Face Recognition (Webcam)
```bash
(face-recognition-rosebrock) $ python face-recognition.py -m webcam -i 0 -o output_webcam/output.avi -e encodings/webcam.pickle
```

