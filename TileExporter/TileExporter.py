import math
import cv2
import numpy as np
import cv2 as cv
import os
import glob

png_dir = 'C:\\BNRMisc\\AssetCollections\\MacAssets48Git\\ConvertedAssets\\pvr2png\\tilemaps'
base_out_dir = 'C:\\BNRMisc\\AssetCollections\\MacAssets48Git\\ConvertedAssets\\pvr2png\\split_tiles'

INITIAL_Y_COORD = 59
INITIAL_X_COORD = 59
HALF_DELTA_X = 60
HALF_DELTA_Y = 30
DELTA_X = 2 * HALF_DELTA_X
DELTA_Y = 2 * HALF_DELTA_Y

count = 0

# Need a black tile to act as the 0 tile for tilemaps
black_tile = np.zeros((DELTA_Y, DELTA_X), np.uint8)


def imshow(img, name="window", delay=0):
    cv.imshow(name, img)
    cv.waitKey(delay)


def imwrite(img, masked_alpha, out_dir, base_name):
    """
    Writes a tile texture to disk
    :param img: the texture to write to disk
    :param masked_alpha: the texture's alpha data
    :param out_dir: where the texture should be saved
    :param base_name: the name of the tileset
    :return:
    """
    global count

    # If the tile contains no meaningful data, make it a black tile, otherwise keep original tile data
    if masked_alpha.mean() > 120:
        rgba = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    else:
        rgba = cv.cvtColor(black_tile, cv.COLOR_BGR2BGRA)

    # Readd the mask to keep proper transparency
    rgba[:, :, 3] = mask
    cv.imwrite(os.path.join(out_dir, str(count) + '-' + base_name), rgba)
    count += 1


def write_initial_tile(out_dir, base_name):
    """
    Writes the initial black tile to the texture folder
    :param out_dir: where the texture should be saved
    :param base_name: the name of the tileset
    """
    global count

    rgba = cv.cvtColor(black_tile, cv.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    cv.imwrite(os.path.join(out_dir, str(count) + '-' + base_name), rgba)
    count += 1


if __name__ == '__main__':
    # Generating mask to crop the exact tile shape
    mask = np.zeros((DELTA_Y, DELTA_X), np.uint8)
    for y in range(0, DELTA_Y):
        for x in range(0, DELTA_X):
            if x == 0:
                if y == 29:
                    mask[y, x] = 255
            elif 1 <= x < HALF_DELTA_X:
                if -(math.floor(0.5 * x) - 29) <= y <= -(math.floor(-0.5 * x + 1) - 30):
                    mask[y, x] = 255
            elif HALF_DELTA_X <= x < DELTA_X:
                if -(math.floor(-0.5 * x + 1) + 29) <= y <= -(math.floor(0.5 * x + 1) - 89):
                    mask[y, x] = 255
            elif x == DELTA_X:
                if y == 30:
                    mask[y, x] = 255

    # Glob for all tilemap pngs extracted from original pvr textures
    for file_name in glob.glob(png_dir + '\\*.png'):
        base_name = os.path.basename(file_name)
        out_dir = os.path.join(base_out_dir, base_name.split('.')[0])
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        count = 0

        # Create blank filler tile
        write_initial_tile(out_dir, base_name)

        # Read in alpha channel
        RGBA = cv.imread(file_name, cv.IMREAD_UNCHANGED)

        # Split alpha from color channels
        alpha = RGBA[:, :, 3]
        img = RGBA[:, :, :3]

        # Read 1st column offset
        for y in range(INITIAL_Y_COORD, img.shape[0], DELTA_Y):
            for x in range(INITIAL_X_COORD, img.shape[1], DELTA_X):
                if y - HALF_DELTA_Y < 0 or x + HALF_DELTA_X > img.shape[1]:
                    break
                pts = np.array([[x, y],
                                [x + HALF_DELTA_X, y - HALF_DELTA_Y + 1],
                                [x, y - DELTA_Y + 1],
                                [x - HALF_DELTA_X + 1, y - HALF_DELTA_Y + 1]])
                rect_x, rect_y, w, h = cv2.boundingRect(pts)
                cropped = img[rect_y:rect_y + h, rect_x:rect_x + w].copy()
                cropped_alpha = alpha[rect_y:rect_y + h, rect_x:rect_x + w].copy()

                masked = cv.bitwise_and(cropped, cropped, mask=mask)
                masked_alpha = cv.bitwise_and(cropped_alpha, cropped_alpha, mask=mask)

                imwrite(masked, masked_alpha, out_dir, base_name)
            # Skip the offset row indexes due to how isometric grid is laid out
            count += 8

        # Reset count to 9 to do the offset rows
        count = 9
        # Read 2nd column offset (Due to isometric grid layout)
        for y in range(INITIAL_Y_COORD + HALF_DELTA_Y, img.shape[0], DELTA_Y):
            for x in range(DELTA_X - 1, img.shape[1], DELTA_X):
                if y - HALF_DELTA_Y < 0 or x + HALF_DELTA_X > img.shape[1]:
                    break
                pts = np.array([[x, y],
                                [x + HALF_DELTA_X, y - HALF_DELTA_Y + 1],
                                [x, y - DELTA_Y + 1],
                                [x - HALF_DELTA_X + 1, y - HALF_DELTA_Y + 1]])
                rect_x, rect_y, w, h = cv2.boundingRect(pts)
                cropped = img[rect_y:rect_y + h, rect_x:rect_x + w].copy()
                cropped_alpha = alpha[rect_y:rect_y + h, rect_x:rect_x + w].copy()

                masked = cv.bitwise_and(cropped, cropped, mask=mask)
                masked_alpha = cv.bitwise_and(cropped_alpha, cropped_alpha, mask=mask)
                imwrite(masked, masked_alpha, out_dir, base_name)
            # Need to skip the already saved tile textures
            count += 8
