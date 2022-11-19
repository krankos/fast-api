import numpy as np
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

from fastapiPrediction import *

origins = ['*']


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = cv2.VideoCapture(0)


def gen_frames():
    # generate frame by frame from camera
    # check if camera is opened and open it if not

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.get("/")
def root():
    # check if camera is opened and close it if it is
    if camera.isOpened():
        camera.release()
    return {"message": "Hello World"}


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...)):

    image = read_imagefile(file)
    result_image = prediction_image(image, ['Khalil', 'Others'])
    if result_image == False:
        return {"message": "No faces found"}
    else:
        # return {
        #     "predictions": predictions,
        #     "result_image": FileResponse(path=result_image, media_type="image/jpg")
        # }
        return FileResponse(path=result_image, media_type="image/jpg")


@app.post("/api/predict/list")
async def predict_list(file: UploadFile = File(...)):
    image = read_imagefile(file)
    predictions = prediction_list(image, ['Khalil', 'Others'])
    return {
        "predictions": predictions
    }

    # def read_imagefile(file):
    #     print(file.read())
    #     image = cv2.imread(str(file.read()))
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     return image

    # def preprocess_image(image):
    #     faces = detect_faces(image)
    #     rois = []
    #     for (x, y, w, h) in faces:
    #         roi = image[y:y+h, x:x+w]
    #         roi = cv2.resize(roi, (256, 256))
    #         roi = np.expand_dims(roi, axis=0)
    #         rois.append(roi)
    #     return rois

    # def detect_faces(image):
    #     haar_file = 'haarcascade_frontalface_default.xml'
    #     face_cascade = cv2.CascadeClassifier(haar_file)
    #     faces = face_cascade.detectMultiScale(image, 1.1, 4)
    #     return faces

    # def get_roi(face):
    #     x, y, w, h = face
    #     roi = image[y:y+h, x:x+w]
    #     roi = cv2.resize(roi, (256, 256))
    #     roi = np.expand_dims(roi, axis=0)
    #     return roi

    # def get_rois(faces):
    #     rois = []
    #     for (x, y, w, h) in faces:
    #         roi = image[y:y+h, x:x+w]
    #         roi = cv2.resize(roi, (256, 256))
    #         roi = np.expand_dims(roi, axis=0)
    #         rois.append(roi)
    #     return rois

    # def load_model():
    #     model = tf.keras.models.load_model('model9.h5')
    #     return model

    # def predict_face(model, roi, class_names):
    #     prediction = model.predict(roi)
    #     score = tf.nn.softmax(prediction[0])
    #     return class_names[np.argmax(prediction)], round(np.max(score)*100)

    # def prediction(class_names):
    #     model = load_model()
    #     rois = preprocess_image(image)
    #     predictions = []
    #     for roi in rois:
    #         predictions.append(predict_face(model, roi, class_names))
    #     return predictions

    # def draw_faces(image):
    #     faces = detect_faces(image)
    #     for (x, y, w, h) in faces:
    #         # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
    #         # use plt to draw the rectangle
    #         plt.gca().add_patch(plt.Rectangle((x, y), w, h,
    #                                           fill=False, edgecolor='red', linewidth=2))
    #         roi = get_roi((x, y, w, h))
    #         name, accuracy = predict_face(model, roi, class_names)
    #         plt.text(x, y, name,
    #                  color='white', fontsize=20, backgroundcolor='red')
    #         # show accuracy
    #         plt.text(x, y+h, str(accuracy)+'%',
    #                  color='white', fontsize=8, backgroundcolor='green')

    #     plt.axis('off')
    #     plt.savefig('predicteds.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    #     # read the image and return it

    #     return "predicteds.jpg"

    # def prediction_result(image, class_names):
    #     return draw_faces(image), prediction(class_names)
