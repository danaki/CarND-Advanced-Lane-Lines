import numpy as np
import cv2
import glob

class Camera:
    def __init__(self):
        ret, mtx, dist, rvecs, tvecs, chessboards = self.calibrate_camera(glob.glob('../camera_cal/calibration*.jpg'))

        self.mtx = mtx
        self.dist = dist

    def undistort(self, image, remove_hood=False):
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))

        dst = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        return dst

    def calibrate_camera(self, image_files):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        chessboards = []

        # Step through the list and search for chessboard corners
        for fname in image_files:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                chessboards.append(cv2.drawChessboardCorners(img, (9, 6), corners, ret))

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return ret, mtx, dist, rvecs, tvecs, chessboards


class Perspective:
    h = None
    w = None
    polygon = np.array([])
    M = []

    def __init__(self, h, w):
        self.h = h
        self.w = w

        x_center = self.w / 2

        # self.polygon = np.array([x_center - 115, self.h - 200,
        #                  x_center + 84, self.h - 200,
        #                  150 + x_center_shift / 2, self.h,
        #                  self.w - 150 + x_center_shift / 2, self.h], dtype = "float32").reshape(4, 2)

        self.polygon = np.array([x_center - 125, self.h - 200,
                         x_center + 125, self.h - 200,
                         x_center - 550, self.h,
                         x_center + 550, self.h], dtype = "float32").reshape(4, 2)

        dst = np.array([0, 0, self.h, 0, 0, self.w, self.h, self.w], dtype = "float32").reshape(4, 2)

        self.M = cv2.getPerspectiveTransform(self.polygon, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, self.polygon)

    def warp(self, image):
        return cv2.warpPerspective(image, self.M, (self.h, self.w))

    def unwarp(self, image, new_shape):
        return cv2.warpPerspective(image, self.Minv, new_shape)

    def transform(self, v):
        return cv2.perspectiveTransform(v, self.M)

    def untransform(self, v):
        return cv2.perspectiveTransform(v, self.Minv)

    def polygon_frame(self):
        p = self.polygon
        return np.dstack([p[0], p[1], p[3], p[2], p[0]]).T


class FeatureExtractor:
    # settings
    blur_kernel_size = 5
    abs_sobel_kernel_size = 7
    mag_sobel_kernel_size = 7
    dir_sobel_kernel_size = 7

    def __init__(self):
        pass

    def gaussian_blur(self, img):
        return cv2.GaussianBlur(img, (self.blur_kernel_size, self.blur_kernel_size), 0)

    def abs_sobel_thresh(self, img, dx, dy, thresh):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=self.abs_sobel_kernel_size)

        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return grad_binary

    def mag_thresh(self, img, thresh):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.mag_sobel_kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.mag_sobel_kernel_size)

        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255

        gradmag = (gradmag / scale_factor).astype(np.uint8)

        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        return mag_binary

    def dir_threshold(self, img, thresh):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.dir_sobel_kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.dir_sobel_kernel_size)

        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        dir = np.arctan2(abs_sobely, abs_sobelx)

        dir_binary = np.zeros_like(dir)
        dir_binary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1

        return dir_binary

    def color_threshold(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        yellow = cv2.inRange(hsv, (20, 100, 100), (50, 255, 255))

        sensitivity_1 = 68
        white = cv2.inRange(hsv, (0, 0, 255 - sensitivity_1), (255, 20, 255))

        sensitivity_2 = 60
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        white_2 = cv2.inRange(hls, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
        white_3 = cv2.inRange(hls, (200, 200, 200), (255, 255, 255))

        bit_layer = yellow | white | white_2 | white_3

        return bit_layer

    def color_mask(self, hsv, low, high):
        mask = cv2.inRange(hsv, low, high)

        return mask

    def apply_color_mask(self, hsv, img, low, high):
        mask = cv2.inRange(hsv, low, high)
        res = cv2.bitwise_and(img, img, mask=mask)

        return res

    def pipeline(self, img):
        img = self.gaussian_blur(img)

        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(img, 1, 0, (10, 255))
        grady = self.abs_sobel_thresh(img, 0, 1, (60, 255))

        mag_binary = self.mag_thresh(img, (40, 255))
        dir_binary = self.dir_threshold(img, (0.65, 1.05))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        color_binary = self.color_threshold(img)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        total_binary = np.zeros_like(combined)
        total_binary[(color_binary > 0) | (combined > 0)] = 1

        return total_binary


class LaneFinder:
    nwindows = 9
    margin = 100 # the width of the windows +/- margin
    minpix = 50 # minimum number of pixels found to recenter window

    image = None
    histogram = []
    pixels = []
    lane_inds = ([], []) # a tuple (left, right)
    fits = ([], []) # a tuple (left, right)
    curveads = ([], []) # a tuple (left, right, offset)

    def __init__(self, image):
        self.image = image
        self.histogram = self.histogram()
        self.pixels = self.nonzero()
        self.lane_inds = self.split_lanes()
        self.fits = self.fit_lines()
        self.curveads = self.calc_curverads()

    def histogram(self):
        return np.sum(self.image[self.image.shape[0]/2:,:], axis=0)

    def nonzero(self):
        nonzero = self.image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        return nonzeroy, nonzerox

    def split_lanes(self):
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(self.histogram.shape[0] / 2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(self.image.shape[0] / self.nwindows)

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        nonzeroy, nonzerox = self.pixels

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.image.shape[0] - (window + 1) * window_height
            win_y_high = self.image.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            #
            # if win_xleft_high > self.image.shape[1] / 2 - margin:
            #     win_xleft_high = self.image.shape[1] / 2 - margin
            #
            # if win_xright_low < self.image.shape[1] / 2 + margin:
            #     win_xright_low = self.image.shape[1] / 2 + margin
            #
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        return np.concatenate(left_lane_inds), np.concatenate(right_lane_inds)

    def extract_pixels(self):
        # Extract left and right line pixel positions
        leftx = self.pixels[1][self.lane_inds[0]]
        lefty = self.pixels[0][self.lane_inds[0]]
        rightx = self.pixels[1][self.lane_inds[1]]
        righty = self.pixels[0][self.lane_inds[1]]

        return leftx, lefty, rightx, righty

    def polyfit(self, X, y, degree):
        if len(X) == 0:
            return None

        return np.polyfit(X, y, degree, w=(np.max(X) - X) / np.max(X))

    def fit_lines(self):
        leftx, lefty, rightx, righty = self.extract_pixels()

        # Fit a second order polynomial to each
        left_fit = self.polyfit(lefty, leftx, 2)
        right_fit = self.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def calc_curverad(self, fit, y):
        ym_per_pix = 30 / 720 # meters per pixel in y dimension

        return ((1 + (2 * fit[0] * y * ym_per_pix + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

    def calc_curverads(self):
        leftx, lefty, rightx, righty = self.extract_pixels()

        y_eval = self.image.shape[0]
        left_fit = self.fits[0]
        right_fit = self.fits[1]

        ym_per_pix = 30 / 720 # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        bottom_left_fitx = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        bottom_right_fitx = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        lane_center = (bottom_left_fitx + bottom_right_fitx) / 2
        pixel_offset = lane_center - self.image.shape[1] / 2
        offset = pixel_offset * xm_per_pix

        return left_curverad, right_curverad, offset
