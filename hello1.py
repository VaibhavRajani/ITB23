# python -m http.server
# from the output folder to open http on 8000 port

from flask import Flask, render_template, request, Response, redirect
from werkzeug.utils import secure_filename           # Used to store filename
import os
import re
from werkzeug.utils import secure_filename
import recog
from client4 import Vidcamera
from own_pc import Vidcamera1
import recog
import cv2
import face_recognition
import pickle
from imutils import paths

app = Flask(__name__)


# front page

@app.route('/')
def front_page():
    return render_template('index1.html')

# about us


@app.route('/main1')
def main_page():
    return render_template('aboutus.html')

# Extra


@app.route('/uploads')
def uploader():
    path = 'static/uploads/'
    uploads = sorted(os.listdir(path), key=lambda x: os.path.getctime(
        path+x))        # Sorting as per image upload date and time
    print(uploads)
    #uploads = os.listdir('static/uploads')
    uploads = ['uploads/' + file for file in uploads]
    uploads.reverse()
    # Pass filenames to front end for display in 'uploads' variable
    return render_template("index.html", uploads=uploads)


app.config['UPLOAD_PATH'] = 'static/uploads'             # Storage path


@app.route("/upload", methods=['GET', 'POST'])
def upload_file():                                       # This method is used to upload files
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        # f.save(secure_filename(f.filename))
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        # Redirect to route '/' for displaying images on fromt end
        return redirect("/uploads")

# video capture


@app.route('/video_1')
def index_1():
    return render_template('video.html')


@app.route('/video_feed_1')
def video_feed_1():
    webcam_video_stream = cv2.VideoCapture(0)

    # load the sample images and get the 128 face embeddings from them
    modi_image = face_recognition.load_image_file(
        'images/samples/Narendra_Modi.jpg')
    modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

    batul_image = face_recognition.load_image_file(
        'images/samples/Batul_Khambata.jpg')
    batul_face_encodings = face_recognition.face_encodings(batul_image)[0]

    vaibhav_image = face_recognition.load_image_file(
        'images/samples/Vaibhav_Rajani.jpg')
    vaibhav_face_encodings = face_recognition.face_encodings(vaibhav_image)[0]

    harsh_image = face_recognition.load_image_file(
        'images/samples/Harsh_Sangani.jpg')
    harsh_face_encodings = face_recognition.face_encodings(harsh_image)[0]

    donald_image = face_recognition.load_image_file(
        'images/samples/Donald_Trump.jpg')
    donald_face_encodings = face_recognition.face_encodings(donald_image)[0]

    elon_image = face_recognition.load_image_file(
        'images/samples/Elon_Musk.jpeg')
    elon_face_encodings = face_recognition.face_encodings(elon_image)[0]

    messi_image = face_recognition.load_image_file(
        'images/samples/Lionel_Messi.jpg')
    messi_face_encodings = face_recognition.face_encodings(messi_image)[0]

    # save the encodings and the corresponding labels in seperate arrays in the same order
    known_face_encodings = [modi_face_encodings,
                            batul_face_encodings,
                            vaibhav_face_encodings,
                            harsh_face_encodings,
                            donald_face_encodings,
                            elon_face_encodings,
                            messi_face_encodings]
    known_face_names = ["Narendra_Modi",
                        "Batul_Khambata",
                        "Vaibhav_Rajani",
                        "Harsh_Sangani",
                        "Donald_Trump",
                        "Elon_Musk",
                        "Lionel_Messi"]

    # initialize the array variable to hold all face locations, encodings and names
    all_face_locations = []
    all_face_encodings = []
    all_face_names = []

    # loop through every frame in the video
    while True:
        # get the current frame from the video stream as an image
        ret, current_frame = webcam_video_stream.read()
        # resize the current frame to 1/4 size to proces faster
        current_frame_small = cv2.resize(
            current_frame, (0, 0), fx=0.25, fy=0.25)
        # detect all faces in the image
        # arguments are image,no_of_times_to_upsample, model
        all_face_locations = face_recognition.face_locations(
            current_frame_small, number_of_times_to_upsample=1, model='hog')

        # detect face encodings for all the faces detected
        all_face_encodings = face_recognition.face_encodings(
            current_frame_small, all_face_locations)

        # looping through the face locations and the face embeddings
        for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
            # splitting the tuple to get the four position values of current face
            top_pos, right_pos, bottom_pos, left_pos = current_face_location

            # change the position maginitude to fit the actual size video frame
            top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4

            # find all the matches and get the list of matches
            all_matches = face_recognition.compare_faces(
                known_face_encodings, current_face_encoding)

            # string to hold the label
            name_of_person = 'Unknown face'

            # check if the all_matches have at least one item
            # if yes, get the index number of face that is located in the first index of all_matches
            # get the name corresponding to the index number and save it in name_of_person
            if True in all_matches:
                first_match_index = all_matches.index(True)
                name_of_person = known_face_names[first_match_index]

            # draw rectangle around the face
            cv2.rectangle(current_frame, (left_pos, top_pos),
                          (right_pos, bottom_pos), (255, 0, 0), 2)

            # display the name as text in the image
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame, name_of_person, (left_pos+10,
                        bottom_pos+20), font, 0.5, (255, 255, 255), 1)

        # display the video
        cv2.imshow("Webcam Video", current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the stream and cam
    # close all opencv windows open
    webcam_video_stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
