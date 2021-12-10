import haar_cascades
import hog_svm
import DNN
import faceDet

if __name__ == "__main__":

    # Haar Cascades using OpenCV
    print("Haar Cascades model")
    hc = haar_cascades.HaarCascades()
    hc.read_preprocess()
    hc.train()
    hc.predict()
    hc.evaluate()

    # HOG SVM using facial_recognition Dlib package
    print("HOG SVM model")
    hog = hog_svm.HOG_SVM()
    hog.read_preprocess()
    hog.train()
    hog.predict()
    hog.evaluate()

    # DNN (Same models)
    # DNN using opencv with res10 caffe model
    print("DNN model with OpenCV res10")
    myDNN = DNN.DNN()
    myDNN.read_preprocess()
    myDNN.train()
    myDNN.predict()
    myDNN.draw_faces()
    myDNN.evaluate()
    # DNN using opencv with face_detection caffe model
    print("DNN with openCV face detection model")
    my_face_Det = faceDet.FaceDetDNN()
    my_face_Det.read_preprocess()
    my_face_Det.train()
    my_face_Det.predict()
    my_face_Det.draw_faces()
    my_face_Det.evaluate()
