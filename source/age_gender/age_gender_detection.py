import datetime
import numpy as np
import cv2

from keras.preprocessing import image
from utils.age_gender_model import age_model, gender_model
from settings import MALE_ICON_PATH, FEMALE_ICON_PATH, FRONT_FACE_PATH, SAVING_TIME, LOCAL
from source.face.face_match import FaceMatching
from source.rest_api.request_post import send_data
from utils.folder_file_manager import log_print


class AgeGenderDetector:

    def __init__(self):

        self.face_manager = FaceMatching()
        self.face_info = {
            "id": [],
            "encoding": [],
            "age": [],
            "gender": [],
            "t_stamp": [],
            "type": [],
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "csi_port": []
        }
        self.face_cascade = cv2.CascadeClassifier(FRONT_FACE_PATH)
        self.enableGenderIcons = True
        # you can find male and female icons here: https://github.com/serengil/tensorflow-101/tree/master/dataset

        male_icon = cv2.imread(MALE_ICON_PATH)
        self.male_icon = cv2.resize(male_icon, (40, 40))

        female_icon = cv2.imread(FEMALE_ICON_PATH)
        self.female_icon = cv2.resize(female_icon, (40, 40))
        # -----------------------

        self.age_mdl = age_model()
        self.gender_mdl = gender_model()

        # age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
        self.output_indexes = np.array([i for i in range(0, 101)])

    @staticmethod
    def gstreamer_pipeline(
            sensor_id,
            capture_width=1280,
            capture_height=720,
            display_width=1280,
            display_height=720,
            framerate=60,
            flip_method=0,
    ):
        return (
                "nvarguscamerasrc sensor-id=%d! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    sensor_id,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
        )

    def detect_age_gender(self):

        if LOCAL:
            cap = cv2.VideoCapture(0)
        else:
            cap_1 = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0, sensor_id=0), cv2.CAP_GSTREAMER)
            cap_2 = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0, sensor_id=1), cv2.CAP_GSTREAMER)

        while True:

            if LOCAL:
                ret, img = cap.read()
                self.detect_one_frame(img=img, csi_port=None)
            else:
                ret, img_1 = cap_1.read()
                ret, img_2 = cap_2.read()
            # img = cv2.resize(img, (640, 360))

                self.detect_one_frame(img=img_1, csi_port=0)
                self.detect_one_frame(img=img_2, csi_port=1)
            self.face_info = self.customize_on_time()
            cv2.imshow("image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
        # kill open cv things
        if LOCAL:
            cap.release()
            # cv2.destroyAllWindows()
        else:
            cap_1.release()
            cap_2.release()

    def detect_one_frame(self, img, csi_port):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)

        if len(faces) == 0:

            self.face_manager.update_face_info(info=self.face_info)

        else:
            face_boxes = []
            ages = []
            genders = []
            for (x, y, w, h) in faces:
                if w > 130:  # ignore small faces

                    # mention detected face
                    """overlay = img.copy(); output = img.copy(); opacity = 0.6
                    cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),cv2.FILLED) #draw rectangle to main image
                    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)"""
                    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128), 1)  # draw rectangle to main image

                    # extract detected face
                    detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                    face_box = [int(y), int(x + w), int(y + h), int(x)]

                    try:
                        # age gender data set has 40% margin around the face. expand detected face.
                        margin = 30
                        margin_x = int((w * margin) / 100)
                        margin_y = int((h * margin) / 100)
                        detected_face = img[int(y - margin_y):int(y + h + margin_y),
                                        int(x - margin_x):int(x + w + margin_x)]
                        face_box = [int(y - margin_y), int(x + w + margin_x), int(y + h + margin_y),
                                    int(x - margin_x)]

                    except Exception as e:
                        log_print(e)
                        # print("detected face has no margin")
                        # print(e)

                    try:
                        # vgg-face expects inputs (224, 224, 3)
                        detected_face = cv2.resize(detected_face, (224, 224))

                        img_pixels = image.img_to_array(detected_face)
                        img_pixels = np.expand_dims(img_pixels, axis=0)
                        img_pixels /= 255

                        # find out age and gender
                        age_distributions = self.age_mdl.predict(img_pixels)
                        apparent_age = int(np.floor(np.sum(age_distributions * self.output_indexes, axis=1))[0])

                        gender_distribution = self.gender_mdl.predict(img_pixels)[0]
                        gender_index = np.argmax(gender_distribution)
                        face_boxes.append(face_box)
                        ages.append(apparent_age)
                        if LOCAL:

                            enable_gender_icons = True
                            genders.append(gender_index)
                            if gender_index == 0:
                                gender = "F"
                            else:
                                gender = "M"

                            # background for age gender declaration
                            info_box_color = (46, 200, 255)
                            # triangle_cnt = np.array([(x+int(w/2), y+10), (x+int(w/2)-25, y-20),
                            # (x+int(w/2)+25, y-20)])
                            triangle_cnt = np.array(
                                [(x + int(w / 2), y), (x + int(w / 2) - 20, y - 20), (x + int(w / 2) + 20, y - 20)])
                            cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
                            cv2.rectangle(img, (x + int(w / 2) - 50, y - 20), (x + int(w / 2) + 50, y - 90),
                                          info_box_color, cv2.FILLED)

                            # labels for age and gender
                            cv2.putText(img, apparent_age, (x + int(w / 2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 111, 255), 2)
                            male_icon = cv2.imread(MALE_ICON_PATH)
                            female_icon = cv2.imread(FEMALE_ICON_PATH)

                            if enable_gender_icons:
                                if gender == 'M':
                                    gender_icon = male_icon
                                else:
                                    gender_icon = female_icon

                                img[y - 75:y - 75 + male_icon.shape[0],
                                x + int(w / 2) - 45:x + int(w / 2) - 45 + male_icon.shape[1]] = gender_icon
                            else:
                                cv2.putText(img, gender, (x + int(w / 2) - 42, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 111, 255), 2)

                    except Exception as e:
                        log_print(e)
                        # print("exception", str(e))

            self.face_info, move_in_idx, move_out_idx = self.face_manager.recognize_face(face_info=self.face_info,
                                                                                         face_box=face_boxes,
                                                                                         rgb_frame=img_rgb,
                                                                                         age=ages,
                                                                                         gender=genders,
                                                                                         csi_port=csi_port)

            if move_in_idx:
                if LOCAL:
                    print("id:", self.face_info["id"])
                    print("age:", self.face_info["age"])
                    print("gender:", self.face_info["gender"])
                    print("tstamp:", self.face_info["t_stamp"])
                    print("status:", self.face_info["type"])
                else:
                    send_data(data=self.face_info, send_idx=move_in_idx)

            if move_out_idx:
                if LOCAL:
                    print("id:", self.face_info["id"])
                    print("age:", self.face_info["age"])
                    print("gender:", self.face_info["gender"])
                    print("tstamp:", self.face_info["t_stamp"])
                    print("status:", self.face_info["type"])
                else:
                    send_data(data=self.face_info, send_idx=move_out_idx)

            return

    def customize_on_time(self):

        current_time = datetime.datetime.now()

        for i, dt_time in enumerate(self.face_info["t_stamp"][:]):

            diff_current_data_time = current_time - dt_time
            diff_min = int(diff_current_data_time.total_seconds() / 60)

            if diff_min > SAVING_TIME:
                self.face_info["id"].pop(i)
                self.face_info["encoding"].pop(i)
                self.face_info["age"].pop(i)
                self.face_info["gender"].pop(i)
                self.face_info["type"].pop(i)
                self.face_info["t_stamp"].pop(i)

        return self.face_info


if __name__ == '__main__':
    AgeGenderDetector().detect_age_gender()
