import haar_cascades
import hog_svm

if __name__ == "__main__":

    # Haar Cascades using OpenCV
    hc = haar_cascades.HaarCascades()
    hc.read_preprocess()
    hc.train()
    hc.predict()

    # HOG SVM using facial_recognition Dlib package
    hog = hog_svm.HOG_SVM()
    hc.read_preprocess()
    hc.train()
    hc.predict()
