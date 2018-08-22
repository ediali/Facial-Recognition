import face_recognition
import cv2
import os
import numpy as np
import mysql
import mysql.connector
from mysql import connector
from mysql.connector import errorcode
import PIL.Image
import base64
from io import BytesIO
from datetime import datetime
from numpy import ndarray

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


images = []
ids = []
def read_blob():

    try:
        query = "SELECT name,image, id FROM pictures"

        cursor = con.cursor()

        cursor.execute(query)

        result = cursor.fetchall()

        for (name, image, id) in result:
            dt = PIL.Image.open(BytesIO(image)).convert('RGB')
            if len(face_recognition.face_encodings(np.array(dt, 'uint8'))) > 0:
                query2 = "UPDATE pictures SET Encodings = %s WHERE id = %s"
                cursor.execute(query2, ((face_recognition.face_encodings(np.array(dt, 'uint8'))[0]).tostring(),id))
                print((face_recognition.face_encodings(np.array(dt, 'uint8'))[0]).tostring())
                print(str(id) + ": " + name)
                con.commit()

    finally:
        cursor.close()
        con.close()


read_blob()
