# Facial-Recognition
A facial recognition program written in python which connects to a given database and uses that information (pictures and names) to recognize faces. 

## Setup

You only need to install the imported modules at the start of the file. These are opencv, face_recognition, numpy, mysql.connector, PIL. 
Then you need to input the username, password and host of your database in the `config` variable.

## Features

This program will connect to your MySQL database and use the data found within it for facial recognition. It correctly displays rectangles around subject's faces. 

The file `faceRec2` displays facial landmarks around the subject's faces, however this is a slower version of the program. Furthermore, this version requires a `facaedata` folder to be created since it saves the pictures from the database before use.

The final version is `faceRec` which does not save the images from the database onto the computer. 

# toFaceEcondings

This python file converts the stored images in your database and adds them to a "face encodings" column in the table that is being used. This allows so that the process only needs to be done once for the whole database, and therefore speed up the process of scanning for the right person in a database of 3+ million people.
