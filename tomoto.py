import cv2
import tensorflow
import numpy as np
from tensorflow.keras.models import load_model


# model=load_model("/home/harikrishnan/VSCode/OpenCV/ProjectX/tomato_leaf_classifier_X3.h5")
model=load_model("/home/harikrishnan/VSCode/OpenCV/ProjectX/tomato_leaf_classifier_X_new1.h5")
from tensorflow.keras.preprocessing import image

def predict_image(frame): 
    input_shape=(256,256)
    img = cv2.resize(frame,input_shape) #
    img = image.img_to_array(img) #converting to array
    img = np.expand_dims(img,axis=0)  #expanding dims to match for model
    img /=255.0 #normalize
    prediction = model.predict(img)
    predicted_leaf_index = np.argmax(prediction)
    return predicted_leaf_index

# Dictionary mapping species indices to species names
species_mapping = {
      0: 'Bacterial_spot',

      1: 'Early blight',

      2: 'Late blight',

      3: 'Leaf Mold',

      4: 'Septoria leaf spot',

      5: 'Spider_nites Two-spotted spider mite',

      6: 'Target Spot',

      7: 'Tomato Yellow Leaf Curl_Virus',

      8: 'Tomato_mosaic_virus',

      9: 'healthy',

     10: 'powdery mildew'

}

video_capture=cv2.VideoCapture('/home/harikrishnan/VSCode/OpenCV/ProjectX/Tomato_DIsease_Videos/Mosaic Virus.mp4')
while True:
    success,frame=video_capture.read()
    predicted_index=predict_image(frame)
    predicted_name = species_mapping.get(predicted_index, 'Unknown Species')
    final_frame=cv2.putText(frame,text=predicted_name,org=(100,200),fontFace=2,color=(255,0,0),fontScale=2)
    cv2.imshow("Tomoto Leaf",final_frame)
    if cv2.waitKey(1) & 0xFF==27:
        break
video_capture.release()
cv2.destroyAllWindows()