import argparse
import cv2
import numpy as np


def computeFineGrainedSaliency(image):
    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)

    if not success:
        print("Failed to compute saliency!")
        exit(-1)

    return saliencyMap


def thresholdImage(image, thresh_type=cv2.THRESH_OTSU, thresh_value=0):
    valid_types = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU,
                   cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV, cv2.THRESH_TRIANGLE,
                   cv2.THRESH_TRUNC]

    if thresh_type not in valid_types:
        print("Invalid threshold type! Valid types: {}".format(valid_types))
        return None

    if thresh_value < 0 or thresh_value > 255:
        print("Invalid threshold value! Should be in range [0,255]")
        return None

    return cv2.threshold(image, thresh_value, 255, thresh_type)[1]


def findKeypointsORB(gray_image):
    # Initiate ORB object
    orb = cv2.ORB_create(nfeatures=20)
    # find the keypoints with ORB
    keypoints = orb.detect(gray_image, None)
    # compute the descriptors with ORB
    return orb.compute(gray_image, keypoints)
    # keypoints, descriptors = orb.compute(gray, keypoints)
    # draw only the location of the keypoints without size or orientation
    # final_keypoints = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=image)
    # cv2.imshow('ORB keypoints', final_keypoints)


def randomAffineTransformation(img):
    # Not actually true random transformation, but some parameters are randomized
    rows, cols = img.shape[:2]
    a = np.random.random()
    horizontal_mirror = True if np.random.random() > 0.5 else False
    print('horizontal mirror: {}'.format(horizontal_mirror))
    # vertical_mirror = False
    vertical_mirror = True if np.random.random() > 0.5 else False
    print('vertical mirror: {}'.format(vertical_mirror))

    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    if horizontal_mirror:
        a = max(0.8, a*2)
        # print("a: {}".format(a))
        dst_points = np.float32([[int(a*(cols - 1)), 0], [0, 0], [cols-1, rows - 1]])
    else:
        a = max(0.8, a*2)
        # print("a: {}".format(a))
        dst_points = np.float32([[0, 0], [int(a*(cols - 1)), 0], [int((1-a)*(cols-1)), rows - 1]])

    if vertical_mirror:
        dst_points[0][1] = rows - 1
        dst_points[1][1] = rows - 1
        dst_points[2][1] = 0

    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))
    # cv2.imshow('Input', img)
    # cv2.imshow('Output', img_output)
    rows, cols = img_output.shape[:2]
    return img_output, affine_matrix


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Path to the input image')

    args = vars(ap.parse_args())

    image = cv2.imread(args['image'])
    img = cv2.imread(args['image'])

    saliencyMap = computeFineGrainedSaliency(image)
    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    threshMap = thresholdImage((saliencyMap*255).astype("uint8"), cv2.THRESH_OTSU, 0)


    # show the images
    cv2.imshow("Image", image)
    # cv2.imshow("Output", saliencyMap)
    # cv2.imshow("Thresh", threshMap)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)

    #---------------------------------------------------------------------------
    # Harris corner extraction
    # dst = cv2.cornerHarris(gray, 4, 5, 0.04)  # to detect only sharp corners
    # dst = cv2.cornerHarris(gray, 14, 5, 0.04) # to detect soft corners
    # Result is dilated for marking the corners
    # dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    # image[dst > 0.01 * dst.max()] = [0, 0, 0]
    # cv2.imshow('Harris Corners', image)

    #---------------------------------------------------------------------------
    # Shi-Tomasi 'N' strongest corners extraction
    orig_corners = cv2.goodFeaturesToTrack(gray, 7, 0.05, 25)
    orig_corners = np.float32(orig_corners)
    for item in orig_corners:
        x, y = item[0]
        cv2.circle(image, (x, y), 5, 255, -1)
    cv2.imshow("Top 'k' features", image)

    #---------------------------------------------------------------------------
    # sift = cv2.xfeatures2d.SIFT_create()
    # keypoints = sift.detect(gray, None)
    # keypoints, descriptors = sift.detectAndCompute(gray, None)
    # image = cv2.drawKeypoints(image, keypoints,
    #                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    #                                 outImage=image)
    # cv2.imshow('SIFT features', image)

    #---------------------------------------------------------------------------
    # surf = cv2.xfeatures2d.SURF_create()
    # # This threshold controls the number of keypoints
    # surf.setHessianThreshold(8000)
    # kp, des = surf.detectAndCompute(gray, None)
    # image = cv2.drawKeypoints(image, kp, None, (0, 255, 0), 4)
    # cv2.imshow('SURF features', image)

    #---------------------------------------------------------------------------
    # Initiate ORB object
    # orb = cv2.ORB_create()
    # find the keypoints with ORB
    # keypoints = orb.detect(gray, None)
    # compute the descriptors with ORB
    # keypoints, descriptors = orb.compute(gray, keypoints)
    keypoints, descriptors = findKeypointsORB(gray)
    # draw only the location of the keypoints without size or orientation
    final_keypoints = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=image)
    cv2.imshow('ORB keypoints', final_keypoints)


    #---------------------------------------------------------------------------
    # Transformations
    rows, cols = img.shape[:2]

    #---------------------------------------------------------------------------
    #Affine transformation
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    dst_points = np.float32([[int(0.8*(cols - 1)), 0], [0, 0], [cols-1, rows - 1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))
    # cv2.imshow('Input', img)
    # cv2.imshow('Output', img_output)
    rows, cols = img_output.shape[:2]

    #---------------------------------------------------------------------------
    #Projective transformation
    # src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols-1, rows - 1]])
    # dst_points = np.float32([[0, 0], [cols - 1, 0], [int(0.2 * cols), rows - 1],
    #                          [int(0.8 * cols), rows - 1]])
    # projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # img_output = cv2.warpPerspective(img_output, projective_matrix, (cols, rows))
    # cv2.imshow('Input', img)

    gray = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
    new_corners = cv2.goodFeaturesToTrack(gray, 7, 0.05, 25)
    new_corners = np.float32(new_corners)
    for item in new_corners:
        x, y = item[0]
        cv2.circle(img_output, (x, y), 5, 255, -1)
    cv2.imshow("Top 'k' features transformed", img_output)
    # keypoints = orb.detect(gray, None)
    # compute the descriptors with ORB
    # keypoints, descriptors = orb.compute(gray, keypoints)
    keypoints, descriptors = findKeypointsORB(gray)
    # draw only the location of the keypoints without size or orientation
    img_output = cv2.drawKeypoints(img_output, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=img_output)
    cv2.imshow('ORB keypoints on perspective', img_output)
    # cv2.imshow('Output', img_output)

    # Need to find the transformation matrix
    for i in range(len(orig_corners)-3):
        supposed_affine_matrix = cv2.getAffineTransform(orig_corners[i:(i+3)], new_corners[i:(i+3)])
        new_img = cv2.warpAffine(img, supposed_affine_matrix, image.shape[:2])
        cv2.imshow("My image {}".format(i), new_img)

    cv2.waitKey()
