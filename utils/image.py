# https://github.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/blob/master/four_point_object_extractor.py

import cv2
import numpy as np


CONTOUR_MIN_THRESHOLD = 100


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def findLargestCountours(cntList, cntWidths):
    first_largest_cnt_pos = cntWidths.index(max(cntWidths))

    return cntList[first_largest_cnt_pos], cntWidths[first_largest_cnt_pos]


def convert_object(mask, image):
    gray = mask
    mask_shape = (mask.shape[0], mask.shape[1], 1)
    # mask_shape = mask.shape
    mask_ = np.zeros(mask_shape, dtype=np.uint8)

    #gray = onehot_to_grayscale(gray)
    #gray = gray.astype(np.uint8)

    cv2.imwrite('./model/bw_gray1.png', gray)
    #gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.medianBlur(gray, 5)

    gray = remove_noise(gray)

    # TODO THIS IS "FIXING" using CV_32S or CV_32F
    gray = cv2.convertScaleAbs(gray, alpha=255 / gray.max())
    gray = gray.astype(np.uint8)
    cv2.imwrite('./model/bw_gray2.png', gray)

    countours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for cnt in countours:
        peri = cv2.arcLength(cnt, True)
        convex_hull = cv2.convexHull(cnt)
        cv2.drawContours(mask_, [convex_hull], 0, (255), -1)

    countours, _ = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(countours, key=cv2.contourArea, reverse=True)
    screenCntList = []
    scrWidths = []

    for cnt in cnts:
        convex_hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(convex_hull, True)
        approx = cv2.approxPolyDP(convex_hull, 0.02 * peri, True)
        screenCnt = approx
        if (len(screenCnt) == 4) & (cv2.contourArea(screenCnt) > CONTOUR_MIN_THRESHOLD):
            (X, Y, W, H) = cv2.boundingRect(screenCnt)
            screenCntList.append(screenCnt)
            scrWidths.append(W)

    if len(scrWidths) == 0:
        print("ID Card not found.")
        pass
    else:
        screenCnt, scrWidth = findLargestCountours(screenCntList, scrWidths)

        pts = screenCnt.reshape(4, 2)
        warped = four_point_transform(image, pts)
        # -1 signifies drawing all contours
        cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
        cv2.imwrite('./model/bw_contours.png', image)

        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

# https://stackoverflow.com/questions/43884463/how-to-convert-rgb-image-to-one-hot-encoded-3d-array-based-on-color-using-numpy
# color_dict = {0: (0,   255, 255),
#               1: (255, 255,   0),
#               ....}


# def rgb_to_onehot(rgb_arr, color_dict):
#     num_classes = len(color_dict)
#     shape = rgb_arr.shape[:2]+(num_classes,)
#     arr = np.zeros( shape, dtype=np.int8 )
#     for i, cls in enumerate(color_dict):
#         arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
#     return arr


# def onehot_to_rgb(onehot, color_dict):
#     single_layer = np.argmax(onehot, axis=-1)
#     output = np.zeros( onehot.shape[:2]+(3,) )
#     for k in color_dict.keys():
#         output[single_layer==k] = color_dict[k]
#     return np.uint8(output)


def onehot_to_grayscale(onehot):
    # print(onehot.shape)
    # print(onehot[0][0][0])
    # print(onehot[0][0][1])
    # print(np.argmax(onehot[0][0]))
    output = np.zeros(onehot.shape[:2]+(1,))
    # print(output.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] = 255 * onehot[i][j][0] if np.argmax(onehot[i][j]) == 0 else 0

    return output


def remove_noise(grayimage):
    shape = grayimage.shape
    print("SHAPE SHAPE =>", shape)
    output = np.zeros(grayimage.shape[:2]+(1,))
    for i in range(grayimage.shape[0]):
        for j in range(grayimage.shape[1]):
            output[i][j] = grayimage[i][j] if grayimage[i][j] > 0.5 else 0
    return output
