from skimage.measure import _structural_similarity as ssim
import static_saliency as saliency
import argparse
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import cProfile
import time


def try_to_find_transformation(orig_keypoints, trans_keypoints):
    indexes_orig = random.sample(range(len(orig_keypoints)), 3)
    indexes_trans = random.sample(range(len(trans_keypoints)), 3)
    orig_chosen_points = [orig_keypoints[i].pt for i in indexes_orig]
    trans_chosen_points = [trans_keypoints[i].pt for i in indexes_trans]
    return cv2.getAffineTransform(np.float32(orig_chosen_points), np.float32(trans_chosen_points))


def mse(origImage, transformedImage):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((origImage.astype("float") - transformedImage.astype("float")) ** 2)
    err /= float(origImage.shape[0] * origImage.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def mean_distance(pts1, pts2):
    dist = 0
    for i in range(len(pts1)):
        dist += np.linalg.norm(pts2[i]-pts1[i])
    return dist / len(pts1)


def get_all_combinations_of_three(points):
    all_indexes = []
    for i in range(len(points)):
        for j in range(len(points)):
            for k in range(len(points)):
                all_indexes.append([i, j, k])
    return all_indexes


def transformPointsAffine(points, matrix):
    B = [[matrix[0][0], matrix[0][1]],
         [matrix[1][0], matrix[1][1]]]
    C = [matrix[0][2],
         matrix[1][2]]
    transformed = np.zeros((points.shape))

    for i, row in enumerate(points):
        transformed[i, :] = (np.dot(B, np.transpose(row)) + C)

    return transformed


def mainFunction(args):
    img = cv2.imread(args['image'])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transformed_img, affine_matrix = saliency.random_affine_transformation(img)
    gray_trans = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
    N_MATCHES = 30

    sift = cv2.xfeatures2d.SIFT_create()
    img1, kp_query, des_query = saliency.image_detect_and_compute(sift, img)
    img2, kp_train, des_train = saliency.image_detect_and_compute(sift, transformed_img)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_query, des_train, k=1)
    matches = [match[0] for match in matches]

    # orb = cv2.ORB_create()
    # matches, kp_query, des_query, kp_train, des_train = saliency.get_image_matches(orb, img, transformed_img)
    matches = matches[:N_MATCHES]
    points_query = [kp_query[match.queryIdx].pt for match in matches]
    points_train = [kp_train[match.trainIdx].pt for match in matches]

    # img_matches = cv2.drawMatches(img, kp_query, transformed_img, kp_train, matches[:][0], transformed_img, flags=2)  # Show top 10 matches
    # cv2.imshow('Top {} SIFT matches'.format(N_MATCHES), img_matches)
    # plt.figure(figsize=(16, 16))
    # plt.title('Top {} ORB matches'.format(N_MATCHES))
    # plt.imshow(img_matches);
    # plt.show()
    # cv2.imshow('Top {} ORB matches'.format(N_MATCHES), img_matches)

    rows, cols = img.shape[:2]
    # img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))

    all_indexes = get_all_combinations_of_three(matches)
    max_iterations = len(all_indexes)
    print("Max number of iterations = {}". format(max_iterations))
    # best_ssim = -1
    HUGE_ERROR = 100000
    best_error = HUGE_ERROR
    best_transformation_matrix = None
    # sufficient_ssim = 0.6
    sufficient_error = 15
    iter = 0
    # getMatrixTimes = []
    # warpTimes = []
    # compareTimes = []

    while best_error > sufficient_error and iter < max_iterations:
        # indexes = random.sample(range(len(matches)), 3)
        indexes = all_indexes[iter]
        orig_chosen_points = [kp_query[matches[idx].queryIdx].pt for idx in indexes]
        trans_chosen_points = [kp_train[matches[idx].trainIdx].pt for idx in indexes]
        # tp1 = time.process_time()
        matrix = cv2.getAffineTransform(np.float32(orig_chosen_points), np.float32(trans_chosen_points))
        # tp2 = time.process_time()
        points = np.array(points_query)
        p_rows, p_cols = points.shape
        # new_points = cv2.warpAffine(points, matrix, points.shape)
        # new_points = np.dot(points, matrix)
        new_points = transformPointsAffine(points, matrix)
        error = mean_distance(points_train, new_points)
        # new_img = cv2.warpAffine(gray, matrix, (cols, rows))
        # tp3 = time.process_time()
        # error = mse(gray_trans, new_img)
        # new_ssim = ssim.compare_ssim(gray_trans, new_img, full=True)[0]
        # tp4 = time.process_time()
        # if new_ssim > best_ssim:
        #     best_ssim = new_ssim
        if error < best_error:
            best_error = error
            best_transformation_matrix = matrix
        iter += 1
        # getMatrixTimes.append(tp2 - tp1)
        # warpTimes.append(tp3 - tp2)
        # compareTimes.append(tp4 - tp3)
        if iter % 500 == 1:
            print('*', end='')

    print('\n')
    # print('Completed {} iterations. Best SSIM = '.format(iter, best_ssim))
    if best_error == HUGE_ERROR:
        print('Failed to find transformation matrix')
    else:
        print('Completed {} iterations. Best error = {}'.format(iter, best_error))
        print('Best transformation matrix found:\n{}'.format(best_transformation_matrix))
        print('Original transformation matrix:\n{}'.format(affine_matrix))

    # print("Average time for getAffineTransform(): {}".format(np.mean(getMatrixTimes)))
    # print("Average time for warpAffine(): {}".format(np.mean(warpTimes)))
    # print("Average time for compare_ssim(): {}".format(np.mean(compareTimes)))

        final_img = cv2.warpAffine(img, best_transformation_matrix, (cols, rows))
        final_img_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('final image', final_img_gray)
        cv2.imshow('transformed image', transformed_img)
        (test_ssim, diff) = ssim.compare_ssim(final_img_gray, gray_trans, full=True)
        print("Similarity between the actual transformed image and the computed transformation: {}".format(test_ssim))
        diff = (diff * 255).astype("uint8")
        cv2.imshow("Final difference", diff)


    # sift = cv2.xfeatures2d.SIFT_create()
    # img1, kp1, des1 = saliency.image_detect_and_compute(sift, img)
    # img2, kp2, des2 = saliency.image_detect_and_compute(sift, transformed_img)
    #
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    #
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    #
    # matchesMask = [[0, 0] for i in range(len(matches))]
    # # ratio test as per Lowe's paper
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.55 * n.distance:
    #         matchesMask[i] = [1, 0]
    #
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask,
    #                    flags=0)
    #
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    # plt.figure(figsize=(18, 18))
    # plt.imshow(img3);
    # plt.show()

    # cv2.imshow("Original", img)
    # cv2.imshow("Transformed", transformed_img)
    # print("transformation matrix:\n{}".format(affine_matrix))
    #
    # rows, cols = img.shape[:2]
    # img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))
    #
    # kp, des = saliency.find_keypoints_orb(gray)
    # kp_trans, des_trans = saliency.find_keypoints_orb(gray_trans)
    #
    # max_iterations = len(kp)**3
    # print("Max number of iterations = {}". format(max_iterations))
    # best_ssim = -1
    # best_transformation_matrix = None
    # sufficient_ssim = 0.6
    # iter = 0
    #
    # while best_ssim < sufficient_ssim and iter < max_iterations:
    #     supposed_affine_matrix = tryToFindTransformation(kp, kp_trans)
    #     new_img = cv2.warpAffine(gray, supposed_affine_matrix, (cols, rows))
    #     # cv2.imshow("new image", new_img)
    #     # cv2.waitKey()
    #     new_ssim = ssim.compare_ssim(gray_trans, new_img, full=True)[0]
    #     if new_ssim > best_ssim:
    #         best_ssim = new_ssim
    #         best_transformation_matrix = supposed_affine_matrix
    #     iter += 1
    #     if iter % 100 == 1:
    #         print('*', end='')
    #
    # print('Completed {} iterations'.format(iter))
    # print('Best transformation matrix found:\n{}'.format(best_transformation_matrix))
    #
    # final_img = cv2.warpAffine(img, best_transformation_matrix, (cols, rows))
    # cv2.imshow('final image', final_img)
    # final_img_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow('final image', final_img)
    #
    # (test_ssim, diff) = ssim.compare_ssim(final_img_gray, gray_trans, full=True)
    # print("Similarity between the actual transformed image and the computed transformation: {}".format(test_ssim))
    # diff = (diff * 255).astype("uint8")
    # cv2.imshow("Final difference", diff)


    # print("numb of descriptors: {}   transformed: {}".format(len(des), len(des_trans)))
    # print("descriptor: {}".format(des[0]))
    # final_keypoints = cv2.drawKeypoints(img, kp, flags=0, outImage=img)
    # cv2.imshow('ORB keypoints', final_keypoints)


    cv2.waitKey()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Path to the input image')
    args = vars(ap.parse_args())

    cProfile.run('mainFunction(args)')
