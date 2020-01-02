from skimage.measure import _structural_similarity as ssim
import static_saliency as saliency
import argparse
import cv2
import numpy as np
import random


def tryToFindTransformation(orig_keypoints, trans_keypoints):
    indexes_orig = random.sample(range(len(orig_keypoints)), 3)
    indexes_trans = random.sample(range(len(trans_keypoints)), 3)
    orig_chosen_points = [orig_keypoints[i].pt for i in indexes_orig]
    trans_chosen_points = [trans_keypoints[i].pt for i in indexes_trans]
    return cv2.getAffineTransform(np.float32(orig_chosen_points), np.float32(trans_chosen_points))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Path to the input image')
    args = vars(ap.parse_args())

    img = cv2.imread(args['image'])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transformed_img, affine_matrix = saliency.randomAffineTransformation(img)
    gray_trans = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Original", img)
    cv2.imshow("Transformed", transformed_img)
    print("transformation matrix:\n{}".format(affine_matrix))

    rows, cols = img.shape[:2]
    img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))

    kp, des = saliency.findKeypointsORB(gray)
    kp_trans, des_trans = saliency.findKeypointsORB(gray_trans)

    max_iterations = len(kp)**3
    print("Max number of iterations = {}". format(max_iterations))
    best_ssim = -1
    best_transformation_matrix = None
    sufficient_ssim = 0.6
    iter = 0

    while best_ssim < sufficient_ssim and iter < max_iterations:
        supposed_affine_matrix = tryToFindTransformation(kp, kp_trans)
        new_img = cv2.warpAffine(gray, supposed_affine_matrix, (cols, rows))
        # cv2.imshow("new image", new_img)
        # cv2.waitKey()
        new_ssim = ssim.compare_ssim(gray_trans, new_img, full=True)[0]
        if new_ssim > best_ssim:
            best_ssim = new_ssim
            best_transformation_matrix = supposed_affine_matrix
        iter += 1
        if iter % 100 == 1:
            print('*', end='')

    print('Completed {} iterations'.format(iter))
    print('Best transformation matrix found:\n{}'.format(best_transformation_matrix))

    final_img = cv2.warpAffine(img, best_transformation_matrix, (cols, rows))
    cv2.imshow('final image', final_img)
    final_img_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('final image', final_img)

    (test_ssim, diff) = ssim.compare_ssim(final_img_gray, gray_trans, full=True)
    print("Similarity between the actual transformed image and the computed transformation: {}".format(test_ssim))
    diff = (diff * 255).astype("uint8")
    cv2.imshow("Final difference", diff)


    # print("numb of descriptors: {}   transformed: {}".format(len(des), len(des_trans)))
    # print("descriptor: {}".format(des[0]))
    # final_keypoints = cv2.drawKeypoints(img, kp, flags=0, outImage=img)
    # cv2.imshow('ORB keypoints', final_keypoints)


    cv2.waitKey()
