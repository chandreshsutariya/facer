# main.py

import os
import face_recognition
import sqlite3
import numpy as np

# Load a sample picture and learn how to recognize it
# Assume the directory 'known_people' exists with persons' images
KNOWN_PEOPLE_DIR = '/Users/apple/cjs/known_people'
DB_FILE = 'face_recognition.db'
# Create a threshold for face distance to determine matches
MATCH_THRESHOLD = 0.53


def create_database(db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_name TEXT NOT NULL,
                        photo_file_path TEXT NOT NULL,
                        encoding BLOB NOT NULL
                      )''')
    connection.commit()
    connection.close()


def load_known_faces(known_people_dir=KNOWN_PEOPLE_DIR, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    for person_name in os.listdir(known_people_dir):
        person_dir = os.path.join(known_people_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                encoding = encodings[0]
                cursor.execute('INSERT INTO faces (person_name, photo_file_path, encoding) VALUES (?, ?, ?)',
                               (person_name, filepath, sqlite3.Binary(encoding.tobytes())))

    connection.commit()
    connection.close()


def recognize_face(image_path, db_file=DB_FILE):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    # Load the image from which we want to recognize the face
    image_to_recognize = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image_to_recognize)

    if not encodings:
        print("No face detected in the image.")
        return

    encoding_to_compare = encodings[0]

    # Fetch all known face encodings from database
    cursor.execute('SELECT person_name, encoding FROM faces')
    known_faces = cursor.fetchall()

    for known_person, known_encoding_blob in known_faces:
        # Convert the BLOB back into a numpy array
        known_encoding = np.frombuffer(known_encoding_blob, dtype=np.float64)

        # Calculate the face distance and check it against the threshold
        face_distance = face_recognition.face_distance([known_encoding], encoding_to_compare)[0]
        if face_distance < MATCH_THRESHOLD:
            print(f"Match found: {known_person}, Distance: {face_distance}")
            connection.close()
            return

    print("No match found.")
    connection.close()


if __name__ == "__main__":
    # create_database()
    # load_known_faces()
    print("Database has been populated with known faces.")
    # Sample usage (path to the new image to recognize needs to be specified):
    recognize_face('/Users/apple/cjs/unknown_people/UN_chief_Antonio.png')
