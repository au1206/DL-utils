import cv2 as cv
import numpy as np
import os


class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []
        for (i, imagePath) in enumerate(imagePaths):
            image = cv.imread(imagePath)
            # assuming the paths are organised as /dataset_name/class/image.jpg
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0:
                print("[INFO]: Processed {}/{}".format(i+1, len(imagePaths)))

        return (np.array(data), np.array(labels))
