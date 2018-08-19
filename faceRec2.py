import face_recognition
import cv2
import os
import mysql
import mysql.connector
import dlib
from imutils import face_utils
import imutils

config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'port': '8889',
    'database': 'faceRec',
    'raise_on_warnings': True,
}

try:
    con = mysql.connector.connect(**config)

except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)


# for (id, name) in cursor:
#     print("{},{}".format(id, name))


def write_file(data, filename):
    folderName = "/Users/Edon/PycharmProjects/FaceRecognition/facedata"
    #plt.savefig(os.path.join(folderName, filename))

    with open(os.path.join(folderName, filename), 'wb') as f:
        f.write(data)


def read_blob():
    names = []
    images = []
    try:
        query = "SELECT name,image FROM pictures"

        cursor = con.cursor()

        cursor.execute(query)

        result = cursor.fetchall()

        for (name, image) in result:
            write_file(image, name+".jpg")
            names.append(name)
            images.append(image)

    finally:
        cursor.close()
        con.close()


read_blob()


###################################################


video_capture = cv2.VideoCapture(0)

path = "/Users/Edon/PycharmProjects/FaceRecognition/facedata"

known_face_encodings = []
known_face_names = []

def getImages(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    for imagePath in imagePaths:
        if imagePath == '/Users/Edon/PycharmProjects/FaceRecognition/facedata/.DS_Store':
            continue  # ignores the .DS_Store file (hidden by default)
        name = os.path.split(imagePath)[-1].split(".")[0]
        known_face_names.append(name)
        known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(imagePath))[0])



# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

getImages(path)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    faces = detector(frame, 0)
    faceshapes = []

    for face in faces:
        faceshape = predictor(frame, face)
        faceshape = face_utils.shape_to_np(faceshape)  # convert to numpy array to be iterable
        for (x, y) in faceshape:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), 2)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
                cv2.putText(frame, "Faces Found:" + str(len(face_names)), (0, 700), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()