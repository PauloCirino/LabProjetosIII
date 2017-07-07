import numpy as np
import cv2
import sys
import time
import pickle
import openface


# bgrImg = img
def getRep(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	bbs = align.getLargestFaceBoundingBox(img)
    reps = []
    for bb in bbs:
        alignedFace = align.align( args.imgDim, img, bb,
        	  					   landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE )
        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

def main(img, le, classifier):




if __name__ == "__main__":
  	pic_file_name = sys.argv[1]
  	classifier_file_name = sys.argv[2]

  	img = cv2.imread(pic_file_name, 1)


  	with open(classifier_file_name, 'rb') as classifier_file:
  		le, classifier = pickle.load(classifier_file, encoding='latin1')

  	main(img = img, le = le, classifier = classifier)
