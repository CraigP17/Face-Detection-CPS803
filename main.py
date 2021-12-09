import haar_cascades

if __name__ == "__main__":

    # Haar Cascades
    hc = haar_cascades.HaarCascades()
    hc.read_preprocess()
    hc.train()
    hc.predict()
