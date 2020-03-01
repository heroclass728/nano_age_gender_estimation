import scipy.io
import pandas as pd
import numpy as np
import cv2
import time

from keras.preprocessing import image
from settings import DATA_MAT_PATH, PKL_PATH


class DataFrameTool:

    def __init__(self):

        self.model = None

    @staticmethod
    def extract_names(name):

        return name[0]

    @staticmethod
    def get_image_pixels(image_path):

        return cv2.imread("imdb_data_set/%s" % image_path[0])  # pixel values in scale of 0-255

    def prepare_data_frame(self, model):

        self.model = model

        # download imdb data set here: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ . Faces only version (7 GB)
        mat = scipy.io.loadmat(DATA_MAT_PATH)
        print("imdb.mat meta data file loaded")

        columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score",
                   "second_face_score", "celeb_names", "celeb_id"]

        instances = mat['imdb'][0][0][0].shape[1]

        df = pd.DataFrame(index=range(0, instances), columns=columns)

        for i in mat:
            if i == "imdb":
                current_array = mat[i][0][0]
                for j in range(len(current_array)):
                    # print(j,". ",columns[j],": ",current_array[j][0])
                    df[columns[j]] = pd.DataFrame(current_array[j][0])

        print("data frame loaded (", df.shape, ")")

        # -------------------------------

        # remove pictures does not include any face
        df = df[df['face_score'] != -np.inf]

        # some pictures include more than one face, remove them
        df = df[df['second_face_score'].isna()]

        # discard inclear ones
        df = df[df['face_score'] >= 5]

        # -------------------------------
        # some speed up tricks. this is not a must.

        # discard old photos
        df = df[df['photo_taken'] >= 2000]

        print("some instances ignored (", df.shape, ")")

        df['celebrity_name'] = df['name'].apply(self.extract_names)

        tic = time.time()
        df['pixels'] = df['full_path'].apply(self.get_image_pixels)
        toc = time.time()

        print("reading pixels completed in ", toc - tic, " seconds...")  # 3.4 seconds

        tic = time.time()
        df['face_vector_raw'] = df['pixels'].apply(self.find_face_representation)  # vector for raw image
        toc = time.time()
        print("extracting face vectors completed in ", toc - tic, " seconds...")

        tic = time.time()
        df.to_pickle(PKL_PATH)
        toc = time.time()
        print("storing representations completed in ", toc - tic, " seconds...")

        return

    def find_face_representation(self, img):

        detected_face = img

        try:
            detected_face = cv2.resize(detected_face, (224, 224))
            # plt.imshow(cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB))

            # normalize detected face in scale of -1, +1

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 127.5
            img_pixels -= 1

            representation = self.model.predict(img_pixels)[0, :]
        except Exception as e:

            representation = None
            print(e)

        return representation


if __name__ == '__main__':

    data_tool = DataFrameTool()
    data_tool.prepare_data_frame(model=None)
