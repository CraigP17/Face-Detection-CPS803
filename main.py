import haar_cascades
import hog_svm
import DNN
import faceDet

if __name__ == "__main__":

    # Haar Cascades using OpenCV
    hc = haar_cascades.HaarCascades()
    hc.read_preprocess()
    hc.train()
    hc.predict()
    hc.evaluate()

    # HOG SVM using facial_recognition Dlib package
    hog = hog_svm.HOG_SVM()
    hog.read_preprocess()
    hog.train()
    hog.predict()
    hog.evaluate()

    # DNN (Same models)
    # DNN using opencv with res10 caffe model
    myDNN = DNN.DNN()
    myDNN.read_preprocess()
    myDNN.train()
    myDNN.predict()
    myDNN.draw_faces()
    myDNN.evaluate()
    # DNN using opencv with face_detection caffe model
    my_face_Det = faceDet.FaceDetDNN()
    my_face_Det.read_preprocess()
    my_face_Det.train()
    my_face_Det.predict()
    my_face_Det.draw_faces()
    my_face_Det.evaluate()
