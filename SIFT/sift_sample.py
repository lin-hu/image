#!/usr/bin/env python3

import cv2 as cv
import matplotlib.pyplot as plt

class sift():
    def init():
        self.

    def mark_keypoint(filename):
        """
        mark SIFT keypoint on image
        input: image file
        return: gray plt image obj with keypoint marked
        """
        #show keypoint
        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #keypoints
        sift = cv.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        img = cv.drawKeypoints(gray, keypoints, img)
        return img

    def show_img_sbs(filename1, filename2):
        """
        show two images sidebyside
        """
        #read image
        img1 = cv.imread(filename1)
        img2 = cv.imread(filename2)
        #show two images side-by-side
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(img1, cmap='gray')
        plt.subplot(122)
        plt.imshow(img2, cmap='gray')
        plt.show()
        
    def detect_mark_kp(filename1, filename2):
        """
        find keypoints and connect them
        return: number of matched KP,
                image obj of images with matched KP connected
        """
        #read image
        img1 = cv.imread(filename1)
        img2 = cv.imread(filename2)
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
        sift = cv.SIFT_create()
        kp1, descriptors_1 = sift.detectAndCompute(img1,None)
        kp2, descriptors_2 = sift.detectAndCompute(img2,None)
        print("Number of keypoint {} {}".format(len(kp1), len(kp2)))
        
        #feature matching
        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        print("*** good matches: {} ***" .format(len(good)))
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return len(good), img3

if __name__ == '__main__':
    """
    image files:
    eiffel_orig - original img
    eiffel1 - same as orig
    eiffel2 - diff size
    eiffel_rotate - rotated
    eiffel_flip - flipped
    """
    plt.imshow(img3)
    plt.show()
    print("Done!")
