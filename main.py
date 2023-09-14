import PySimpleGUI as sg
import cv2

# Create a GUI layout
layout = [
    [sg.Image(key='-IMAGE-')],
    [sg.Text('People Count: 0', key='-TEXT-', expand_x=True, justification='c')]
]

# Create the GUI window
window = sg.Window('Face Detector', layout)

# Initialize the video capture and face detection
video_capture = cv2.VideoCapture(0)
face_cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    event, values = window.read(timeout=0)

    if event == sg.WINDOW_CLOSED:
        break

    # Capture a frame from the video
    _, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    detected_faces = face_cascade_classifier.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=7,
        minSize=(50, 50)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Update the displayed image in the GUI
    img_bytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=img_bytes)

    # Update the text with the number of people detected
    window['-TEXT-'].update(f'People Count: {len(detected_faces)}')

# Close the GUI window and release the video capture
window.close()
video_capture.release()
