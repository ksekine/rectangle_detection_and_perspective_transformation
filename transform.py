import collections
from curses.textpad import rectangle
import cv2
import glob
import os
import argparse
import numpy as np


def find_contours(image, area_threshold=50000):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # Approximate contours to line and get only rectangle
    rectangle_contours = []
    for con in contours:
        area = cv2.contourArea(con, False)
        if area > area_threshold:
            arcLength = cv2.arcLength(con, True)
            approx = cv2.approxPolyDP(con, 0.01*arcLength, True)
            if len(approx) == 4:
                rectangle_contours.append(approx)

    return rectangle_contours


def match_points(src_points, dst_points):
    assert len(src_points) == 4 and len(dst_points) == 4

    # Calc distances for all point pairs
    distances = []
    for src in src_points:
        distances_per_points = []
        for dst in dst_points:
            x_diff = src[0] - dst[0]
            y_diff = src[1] - dst[1]
            distance = x_diff * x_diff + y_diff * y_diff
            distances_per_points.append(distance)
        distances.append(distances_per_points)

    # Get nearest point index
    dst_points_order = []
    for dist in distances:
        min_dist = min(dist)
        min_index = dist.index(min_dist)
        dst_points_order.append(min_index)
    
    # If there are duplicates, avoid them by operation
    # TODO: Support this case
    if len(dst_points_order) != len(set(dst_points_order)):
        return src_points, dst_points

    reordered_dst_points = np.array([dst_points[i] for i in dst_points_order], dtype=np.float32)

    return src_points, reordered_dst_points


def warpPerspective(image, rectangle_contours):
    height = image.shape[0]
    width = image.shape[1]
    
    out_images = []
    for con in rectangle_contours:
        # Get the min rectangle that contains the trapezoid
        x_min = width
        x_max = 0
        y_min = height
        y_max = 0
        for point in con:
            if point[0][0] < x_min:
                x_min = point[0][0]
            if point[0][0] > x_max:
                x_max = point[0][0]
            if point[0][1] < y_min:
                y_min = point[0][1]
            if point[0][1] > y_max:
                y_max = point[0][1]
        crop_image = image[y_min:y_max, x_min:x_max]
        
        # Get the src and dst points for perspective transform 
        # and reorder the dst points in order to match the src points
        src_points = np.array(
            [
                [con[0][0][0]-x_min, con[0][0][1]-y_min],
                [con[1][0][0]-x_min, con[1][0][1]-y_min],
                [con[2][0][0]-x_min, con[2][0][1]-y_min],
                [con[3][0][0]-x_min, con[3][0][1]-y_min]
            ],
            dtype=np.float32
        )
        dst_points = np.array(
            [
                [0, 0], 
                [0, y_max-y_min], 
                [x_max-x_min, y_max-y_min], 
                [x_max-x_min, 0]
            ], 
            dtype=np.float32
        )
        src_points, dst_points = match_points(src_points, dst_points)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        out_images.append(cv2.warpPerspective(crop_image, matrix, (x_max-x_min, y_max-y_min)))

    return out_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, required=True, help='input dir that contains rgb images')
    parser.add_argument('--save_dir', type=str, required=True, help='output dir that saves y, u and v images')
    parser.add_argument('--ext', type=str, default='jpg', help='search extension')
    parser.add_argument('--save_contour', action='store_true', help='specify if you want to save the image drawn the contour')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.load_dir, '*.' + args.ext)))
    for i, file in enumerate(files):
        file_name = os.path.splitext(os.path.basename(file))[0]

        image = cv2.imread(file)
        contours = find_contours(image)

        if args.save_contour:
            contour_image = image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 5)

            contour_dir = os.path.join(args.save_dir, file_name, 'contour')
            if not os.path.exists(contour_dir):
                os.makedirs(contour_dir)
            cv2.imwrite(os.path.join(contour_dir, 'contour.png'), contour_image)

        # TODO: Fix aspect ratio
        perspective_images = warpPerspective(image, contours)
        for j, perspective_image in enumerate(perspective_images):
            perspective_dir = os.path.join(args.save_dir, file_name, 'perspective')
            if not os.path.exists(perspective_dir):
                os.makedirs(perspective_dir)
            cv2.imwrite(os.path.join(perspective_dir, '{:0=3}.png'.format(j)), perspective_image)

        print('{} / {} finished! Detected {} rectangles.'.format(i+1, len(files), len(contours)))
