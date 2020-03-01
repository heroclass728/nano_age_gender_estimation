import requests

from settings import HOST, PORT


def send_data(data, send_idx):

    for idx in send_idx:

        if data["gender"][idx - 1] == 0:
            gender = "Female"
        else:
            gender = "Male"

        send_url = "http://" + HOST + ":" + PORT + "/camera" + "/event/" + data["csi_port"][idx - 1]
        p_data = {
            "timestamp": data["t_stamp"][idx - 1],
            "type": data["type"][idx - 1],
            "data": {
                "id": data["id"][idx - 1],
                "face": {
                    "x": data["x"][idx - 1],
                    "y": data["y"][idx - 1],
                    "w": data["w"][idx - 1],
                    "h": data["h"][idx - 1]
                },
                "age": data["age"][idx - 1],
                "gender": gender
            }
        }

        send = requests.post(send_url, data=p_data)
        print(send.text)

    return


if __name__ == '__main__':

    send_data(data="", send_idx="")
