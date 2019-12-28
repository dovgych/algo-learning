import argparse
import cv2


def computeFineGrainedSaliency(image):
    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)

    if not success:
        print("Failed to compute saliency!")
        return None

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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Path to the input image')

    args = vars(ap.parse_args())

    image = cv2.imread(args['image'])

    saliencyMap = computeFineGrainedSaliency(image)
    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    threshMap = thresholdImage((saliencyMap*255).astype("uint8"), cv2.THRESH_OTSU, 0)


    # show the images
    cv2.imshow("Image", image)
    cv2.imshow("Output", saliencyMap)
    cv2.imshow("Thresh", threshMap)
    cv2.waitKey(0)