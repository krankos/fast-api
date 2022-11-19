import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
# haar_file = 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_file)

# image = cv2.imread('many.jpeg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# faces = face_cascade.detectMultiScale(image, 1.1, 4)
# if faces is ():
#     print('No faces found')
#     plt.text(0, 0, 'No faces found', fontsize=20)
# for (x, y, w, h) in faces:
#     # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
#     # use plt to draw the rectangle
#     plt.gca().add_patch(plt.Rectangle((x, y), w, h,
#                                       fill=False, edgecolor='red', linewidth=2))
#     # predict and show the name
#     roi = image[y:y+h, x:x+w]
#     roi = cv2.resize(roi, (256, 256))
#     roi = np.expand_dims(roi, axis=0)
#     prediction = model.predict(roi)
#     score = tf.nn.softmax(prediction[0])
#     print(prediction)
#     print(class_names[np.argmax(prediction)])
#     plt.text(x, y, class_names[np.argmax(prediction)],
#              color='white', fontsize=20, backgroundcolor='red')
#     # show accuracy
#     plt.text(x, y+h, str(round(np.max(score)*100))+'%',
#              color='white', fontsize=8, backgroundcolor='green')

# plt.axis('off')
# plt.savefig('predicteds.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


def read_imagefile(file):
    image = cv2.imdecode(np.fromstring(
        file.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def preprocess_image(image):
    faces = detect_faces(image)
    rois = []
    for (x, y, w, h) in faces:
        roi = image[y:y+h, x:x+w]
        roi = cv2.resize(roi, (256, 256))
        roi = tf.keras.utils.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        rois.append(roi)
    return rois


def detect_faces(image):
    haar_file = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    return faces


def get_roi(face, image):
    x, y, w, h = face
    roi = image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (256, 256))
    roi = tf.keras.utils.img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi


def get_rois(faces, image):
    rois = []
    for (x, y, w, h) in faces:
        roi = image[y:y+h, x:x+w]
        roi = cv2.resize(roi, (256, 256))
        roi = tf.keras.utils.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        rois.append(roi)
    return rois


def load_model():
    model = tf.keras.models.load_model('model9.h5')
    return model


def predict_face(model, roi, class_names):

    prediction = model.predict(roi)
    score = tf.nn.softmax(prediction[0])
    return class_names[np.argmax(prediction)], round(np.max(score)*100)


def prediction(class_names, model, image):
    # model = load_model()
    rois = preprocess_image(image)
    predictions = []
    for roi in rois:
        predictions.append(predict_face(model, roi, class_names))
    return predictions


def draw_faces(image, model, class_names):
    faces = detect_faces(image)
    print(faces)
    if faces == ():
        print('No faces found')
        return False
    else:
        cv2_im = image.copy()
        for (x, y, w, h) in faces:

            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
            # use plt to draw the rectangle
            # plt.gca().add_patch(plt.Rectangle((x, y), w, h,
            #                                   fill=False, edgecolor='red', linewidth=2))

            cv2.rectangle(cv2_im, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi = get_roi((x, y, w, h), image)
            name, accuracy = predict_face(model, roi, class_names)
            # plt.text(x, y, name,
            #          color='white', fontsize=20, backgroundcolor='red')
            cv2.putText(cv2_im, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        6, (255, 255, 255), 2, cv2.LINE_AA)
            # show accuracy
            # plt.text(x, y+h, str(accuracy)+'%',
            #          color='white', fontsize=8, backgroundcolor='green')
            cv2.putText(cv2_im, str(accuracy)+'%', (x, y+h), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (255, 255, 255), 2, cv2.LINE_AA)

        # plt.axis('off')
        # plt.savefig('predicteds.jpg', bbox_inches='tight', pad_inches=0, dpi=10)

        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2.imwrite('predicteds.jpg', cv2_im)

        # read the image and return it

        return "predicteds.jpg"


def prediction_image(image, class_names):
    model = load_model()
    return draw_faces(image, model, class_names)


def prediction_list(image, class_names):
    model = load_model()
    return prediction(class_names, model, image)
