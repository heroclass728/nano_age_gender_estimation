import os

from utils.folder_file_manager import make_directory_if_not_exists


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils'))
AGE_MODEL_PATH = os.path.join(MODEL_DIR, 'age_model_weights.h5')
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, 'gender_model_weights.h5')
DATA_MAT_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model', 'data_frame', 'imdb_crop'))
DATA_MAT_PATH = os.path.join(DATA_MAT_DIR, 'imdb.mat')
PKL_PATH = os.path.join(DATA_MAT_DIR, 'representations.pkl')

ICON_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'icons'))
MALE_ICON_PATH = os.path.join(ICON_DIR, 'male.jpg')
FEMALE_ICON_PATH = os.path.join(ICON_DIR, 'female.jpg')
FRONT_FACE_PATH = os.path.join(CUR_DIR, 'utils', "haarcascade_frontalface_default.xml")

SAVING_TIME = 1
HOST = ""
PORT = ""
# LOCAL = True
LOCAL = False
