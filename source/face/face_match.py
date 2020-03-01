import face_recognition
import datetime


class FaceMatching:

    def __init__(self):

        self.info = None
        self.age = None
        self.gender = None

    def recognize_face(self, face_info, face_box, rgb_frame, age, gender, csi_port):

        self.info = face_info
        send_move_in_idx = []
        send_move_out_idx = []

        encodings = face_recognition.face_encodings(rgb_frame, face_box)

        if not face_info["encoding"]:

            for i, encoding in enumerate(encodings):

                self.insert_new_info(idx=i + 1, status="move_in", encoding=encoding, age=age[i], gender=gender[i],
                                     x=face_box[i][3], y=face_box[i][0], w=face_box[i][1], h=face_box[i][2],
                                     csi_port=csi_port)
                send_move_in_idx.append(i + 1)

        else:

            in_ids = []
            for i, encoding in enumerate(encodings):

                matches = face_recognition.compare_faces(self.info["encoding"], encoding)
                if True in matches:

                    # find the indexes of all matched faces then initialize a dictionary to count the total
                    # number of times each face was matched
                    matched_ides = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for j in matched_ides:
                        name = self.info["id"][j]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number of votes (note: in the event
                    # of an unlikely tie Python will select first entry in the dictionary)
                    idx = max(counts, key=counts.get)
                    in_ids.append(idx)

                    face_index = self.info["id"].index(idx)
                    self.info["type"][face_index] = "move_in"
                    self.info["age"][face_index].append(age[i])
                    self.info["gender"][face_index].append(gender[i])
                    self.info["t_stamp"][face_index] = datetime.datetime.now()
                    self.info["x"][face_index] = face_box[i][3]
                    self.info["y"][face_index] = face_box[i][0]
                    self.info["w"][face_index] = face_box[i][1]
                    self.info["x"][face_index] = face_box[i][2]
                    self.info["csi_port"][face_index] = csi_port

                    # update the list of names
                else:

                    max_id = max(self.info["id"])
                    self.insert_new_info(idx=max_id + 1, encoding=encoding, status="move_in", age=age[i],
                                         gender=gender[i], x=face_box[i][3], y=face_box[i][0], w=face_box[i][1],
                                         h=face_box[i][2], csi_port=csi_port)
                    in_ids.append(max_id)
                    send_move_in_idx.append(max_id + 1)

            for i, status in enumerate(self.info["type"]):

                if status == "move_in":

                    if self.info["id"][i] not in in_ids:

                        send_move_out_idx.append(self.info["id"])

                        self.info["type"][i] = "move_out"
                        self.info["age"][i] = [self.customize_age_gender(age_gender_list=self.info["age"][i])]
                        self.info["gender"][i] = [self.customize_age_gender(age_gender_list=self.info["gender"][i])]

        return face_info, send_move_in_idx, send_move_out_idx

    def insert_new_info(self, encoding, idx, status, age, gender, x, y, w, h, csi_port):

        self.info["id"].append(idx)
        self.info["encoding"].append(encoding)
        self.info["age"].append([age])
        self.info["gender"].append([gender])
        self.info["t_stamp"].append(datetime.datetime.now())
        self.info["type"].append(status)
        self.info["x"].append(x)
        self.info["y"].append(y)
        self.info["w"].append(w)
        self.info["h"].append(h)
        self.info["csi_port"].append(csi_port)

        return

    def update_face_info(self, info):

        for i, status in enumerate(info["type"]):

            if status == "move_in":

                info["type"][i] = "move_out"
                self.info["age"][i] = [self.customize_age_gender(age_gender_list=self.info["age"][i])]
                self.info["gender"][i] = [self.customize_age_gender(age_gender_list=self.info["gender"][i])]

        return

    @staticmethod
    def customize_age_gender(age_gender_list):

        count = len(age_gender_list)
        avg_value = int(sum(age_gender_list) / count)

        return avg_value
