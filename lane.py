import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np

import math
# from settings import CALIB_FILE_NAME, PERSPECTIVE_FILE_NAME

def get_center_shift(coeffs, img_size, pixels_per_meter):
    return np.polyval(coeffs, img_size[1]/pixels_per_meter[1]) - (img_size[0]//2)/pixels_per_meter[0]

def get_curvature(coeffs, img_size, pixels_per_meter):
    return ((1 + (2*coeffs[0]*img_size[1]/pixels_per_meter[1] + coeffs[1])**2)**1.5) / np.absolute(2*coeffs[0])


#class that finds line in a mask
class LaneLineFinder:
    def __init__(self, img_size, pixels_per_meter, center_shift):
        self.lane_line_found = False
        self.poly_coeffs = np.zeros(3, dtype=np.float32)
        self.coeff_history = np.zeros((3, 7), dtype=np.float32)
        self.img_size = img_size
        self.pixels_per_meter = pixels_per_meter
        self.line_mask = np.ones((img_size[1], img_size[0]), dtype=np.uint8)
        self.other_line_mask = np.zeros_like(self.line_mask)
        self.line = np.zeros_like(self.line_mask)
        self.num_lost = 0
        # self.still_to_find = 1
        self.shift = center_shift
        self.first = True
        self.stddev = 0
        
    def reset_lane_line(self):
        self.lane_line_found = False
        self.poly_coeffs = np.zeros(3, dtype=np.float32)
        self.line_mask[:] = 1
        self.first = True

    def one_lost(self):
        # self.still_to_find = 5
        if self.lane_line_found:
            self.num_lost += 1
            if self.num_lost >= 7:
                self.reset_lane_line()

    def one_found(self):
        self.first = False
        self.num_lost = 0
        if not self.lane_line_found :
            # self.still_to_find -= 1
            # if self.still_to_find <= 0:
                print("LANEFOUND")
                self.lane_line_found = True

    def fit_lane_line(self, mask):
        y_coord, x_coord = np.where(mask)
        y_coord = y_coord.astype(np.float32)/self.pixels_per_meter[1]
        x_coord = x_coord.astype(np.float32)/self.pixels_per_meter[0]
        if len(y_coord) <= 150:
            coeffs = np.array([0, 0, (self.img_size[0]//2)/self.pixels_per_meter[0] + self.shift], dtype=np.float32)
        else:
            coeffs, v = np.polyfit(y_coord, x_coord, 2, rcond=1e-16, cov=True)
            self.stddev = 1 - math.exp(-5*np.sqrt(np.trace(v)))

        self.coeff_history = np.roll(self.coeff_history, 1)

        if self.first:
            self.coeff_history = np.reshape(np.repeat(coeffs, 7), (3, 7))
        else:
            self.coeff_history[:, 0] = coeffs

        value_x = get_center_shift(coeffs, self.img_size, self.pixels_per_meter)
        curve = get_curvature(coeffs, self.img_size, self.pixels_per_meter)

        if (self.stddev > 0.95) | (len(y_coord) < 150) | (math.fabs(value_x - self.shift) > math.fabs(0.5*self.shift))  \
                 | (curve < 15):

            self.coeff_history[0:2, 0] = 0
            self.coeff_history[2, 0] = (self.img_size[0]//2)/self.pixels_per_meter[0] + self.shift
            self.one_lost()
            print(self.stddev, len(y_coord), math.fabs(value_x-self.shift)-math.fabs(0.5*self.shift), curve)
        else:
            print("one_found")
            self.one_found()

        self.poly_coeffs = np.mean(self.coeff_history, axis=1)

    def get_line_points(self):
        y = np.array(range(0, self.img_size[1]+1, 10), dtype=np.float32)/self.pixels_per_meter[1]
        x = np.polyval(self.poly_coeffs, y)*self.pixels_per_meter[0]
        y *= self.pixels_per_meter[1]
        return np.array([x, y], dtype=np.int32).T

    def get_other_line_points(self):
        pts = self.get_line_points()
        pts[:, 0] = pts[:, 0] - 2*self.shift*self.pixels_per_meter[0]
        return pts

    def find_lane_line(self, mask, reset=False):
        n_segments = 16
        window_width = 30
        step = self.img_size[1]//n_segments

        if reset or (not self.lane_line_found) or self.first : # and self.still_to_find == 5) or self.first:
            self.line_mask[:] = 0
            n_steps = 4
            window_start = self.img_size[0]//2 + int(self.shift*self.pixels_per_meter[0]) - 3 * window_width
            window_end = window_start + 6*window_width
            sm = np.sum(mask[self.img_size[1]-4*step:self.img_size[1], window_start:window_end], axis=0)
            sm = np.convolve(sm, np.ones((window_width,))/window_width, mode='same')
            argmax = window_start + np.argmax(sm)
            shift = 0
            for last in range(self.img_size[1], 0, -step):
                first_line = max(0, last - n_steps*step)
                sm = np.sum(mask[first_line:last, :], axis=0)
                sm = np.convolve(sm, np.ones((window_width,))/window_width, mode='same')
                window_start = min(max(argmax + int(shift)-window_width//2, 0), self.img_size[0]-1)
                window_end = min(max(argmax + int(shift) + window_width//2, 0+1), self.img_size[0])
                new_argmax = window_start + np.argmax(sm[window_start:window_end])
                new_max = np.max(sm[window_start:window_end])
                if new_max <= 2:
                    new_argmax = argmax + int(shift)
                    shift = shift/2
                if last != self.img_size[1]:
                    shift = shift*0.25 + 0.75*(new_argmax - argmax)
                argmax = new_argmax
                cv2.rectangle(self.line_mask, (argmax-window_width//2, last-step), (argmax+window_width//2, last),
                              1, thickness=-1)
        else:
            self.line_mask[:] = 0
            points = self.get_line_points()
            if not self.lane_line_found :
                factor = 3
            else:
                factor = 2
            cv2.polylines(self.line_mask, [points], 0, 1, thickness=int(factor*window_width))

        self.line = self.line_mask * mask
        self.fit_lane_line(self.line)
        self.first = False
        if not self.lane_line_found :
            self.line_mask[:] = 1
        points = self.get_other_line_points()
        self.other_line_mask[:] = 0
        cv2.polylines(self.other_line_mask, [points], 0, 1, thickness=int(5*window_width))



if __name__ == "__main__":
    pass    
    # with open(CALIB_FILE_NAME, 'rb') as f:
    #     calib_data = pickle.load(f)
    # cam_matrix = calib_data["cam_matrix"]
    # dist_coeffs = calib_data["dist_coeffs"]
    # img_size = calib_data["img_size"]

    # with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
    #     perspective_data = pickle.load(f)

    # perspective_transform = perspective_data["perspective_transform"]
    # pixels_per_meter = perspective_data['pixels_per_meter']
    # orig_points = perspective_data["orig_points"]

    # input_dir = "test_images"
    # output_dir = "output_images"


    # for image_file in os.listdir(input_dir):
    #         if image_file.endswith("jpg"):
    #             # turn images to grayscale and find chessboard corners
    #             img = mpimg.imread(os.path.join(input_dir, image_file))
    #             lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, cam_matrix, dist_coeffs,
    #                 perspective_transform, pixels_per_meter, "warning.png")
    #             img = lf.process_image(img, True, show_period=1, blocking=False)





