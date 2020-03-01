import platform

from source.age_gender.age_gender_detection import AgeGenderDetector


def running_on_jetson_nano():

    return platform.machine() == "aarch64"


if __name__ == '__main__':

    AgeGenderDetector().detect_age_gender()
