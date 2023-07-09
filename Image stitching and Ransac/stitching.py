import cv2

import numpy as np
class Stitcher :
    def stitch(self, images, ratio=0.75, reprojThresh=5.0,showMatches=False):

         (imageB, imageA) = images
         (kpsA, featuresA) = self.detectAndDescribe(imageA)
         (kpsB, featuresB) = self.detectAndDescribe(imageB)

         M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
         (matches, H, status) = M

         if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

         if M is None:
           return None





         result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1]+1, imageA.shape[0]+imageB.shape[0]+1))



         result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB




         if showMatches:
            return (result, vis)

         return result


    def detectAndDescribe(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)


        kps = np.float32([kp.pt for kp in kps])


        return (kps, features)


    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []

        for m in rawMatches:

            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        print(matches)
        if len(matches) > 4:

            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])


            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)

            return (matches, H, status)

        return None


    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):

        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB


        for ((trainIdx, queryIdx), s) in zip(matches, status):

           if s == 1:

             ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
             ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))


             cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

imageA = cv2.imread('foto5A.jpg') #imageA should be the image that is going to be on the top or on the left in the stitched image
imageB = cv2.imread('foto5B.jpg') #imageB should be the image that is going to be on the bottom or on the right in the stitched image

stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
# cv2.imwrite('5.jpg', result)
cv2.waitKey(0)