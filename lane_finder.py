import cv2
import numpy as np
import math
from datetime import datetime
from matplotlib import pyplot as plt
from collections import deque
temp_dir = "images/detection/detect.jpg"
def compute_hls_white_yellow_binary(image,kernel_size =3):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB formats
    """
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # converted = cv2.GaussianBlur(converted, (kernel_size, kernel_size), 0)
    # yellow color mask  
    lower = np.uint8([ 20,   0, 100])
    upper = np.uint8([ 35, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # cv2.imwrite(temp_dir, cv2.bitwise_and(image,image,mask=yellow_mask))
   
    # white color mask
    lower = np.uint8([  0, 190,   0])
    upper = np.uint8([180, 255, 85])
    white_mask = cv2.inRange(converted, lower, upper)
    # cv2.imwrite(temp_dir, cv2.bitwise_and(image,image,mask=white_mask))
     # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    # cv2.imwrite(temp_dir, cv2.bitwise_and(image,image,mask=mask))
    return mask

# def mag_sobel(gray_img, kernel_size=3, thres=(0, 255)):
#     """
#     Computes sobel matrix in both x and y directions, merges them by computing the magnitude in both directions
#     and applies a threshold value to only set pixels within the specified range
#     """
#     sx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
#     sy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
#     sxy = np.sqrt(np.square(sx) + np.square(sy))
#     scaled_sxy = np.uint8(255 * sxy / np.max(sxy))
    
#     sxy_binary = np.zeros_like(scaled_sxy)
#     sxy_binary[(scaled_sxy >= thres[0]) & (scaled_sxy <= thres[1])] = 1
    
#     return sxy_binary

# def dir_sobel(gray_img, kernel_size=3, thres=(0, np.pi/2)):
#     """
#     Computes sobel matrix in both x and y directions, gets their absolute values to find the direction of the gradient
#     and applies a threshold value to only set pixels within the specified range
#     """
#     sx_abs = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size))
#     sy_abs = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size))
    
#     dir_sxy = np.arctan2(sx_abs, sy_abs)

#     binary_output = np.zeros_like(dir_sxy)
#     binary_output[(dir_sxy >= thres[0]) & (dir_sxy <= thres[1])] = 1
    
#     return binary_output


# def combined_sobels(sx_binary, sy_binary, sxy_magnitude_binary, gray_img, kernel_size=3, angle_thres=(0, np.pi/2)):
#     sxy_direction_binary = dir_sobel(gray_img, kernel_size=kernel_size, thres=angle_thres)
    
#     combined = np.zeros_like(sxy_direction_binary)
#     # Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels    
#     combined[(sx_binary == 1) | ((sy_binary == 1) & (sxy_magnitude_binary == 1) & (sxy_direction_binary == 1))] = 1
    
#     return combined

# def abs_sobel(gray_img, x_dir=True, kernel_size=3, thres=(0, 255)):
#     """
#     Applies the sobel operator to a grayscale-like (i.e. single channel) image in either horizontal or vertical direction
#     The function also computes the asbolute value of the resulting matrix and applies a binary threshold
#     """
#     sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) if x_dir else cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size) 
#     sobel_abs = np.absolute(sobel)
#     sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))
    
#     gradient_mask = np.zeros_like(sobel_scaled)
#     gradient_mask[(thres[0] <= sobel_scaled) & (sobel_scaled <= thres[1])] = 1
#     return gradient_mask

# def get_combined_binary_thresholded_img(undist_img):
#     """
#     Applies a combination of binary Sobel and color thresholding to an undistorted image
#     Those binary images are then combined to produce the returned binary image
#     """
#     undist_img_gray = cv2.cvtColor(undist_img, cv2.COLOR_RGB2LAB  )[:,:,0]
    

#     sx = abs_sobel(undist_img_gray, kernel_size=15, thres=(20, 120))
#     sy = abs_sobel(undist_img_gray, x_dir=False, kernel_size=15, thres=(20, 120))
#     sxy = mag_sobel(undist_img_gray, kernel_size=15, thres=(80, 200))
#     sxy_combined_dir = combined_sobels(sx, sy, sxy, undist_img_gray, kernel_size=15, angle_thres=(np.pi/4, np.pi/2))   
    
#     hls_w_y_thres = compute_hls_white_yellow_binary(undist_img)
#     combined_binary = np.zeros_like(hls_w_y_thres)
#     combined_binary[(sxy_combined_dir == 1) | (hls_w_y_thres == 1)] = 1


#     cv2.imwrite(temp_dir, sx*150)
#     cv2.imwrite(temp_dir, sy*150)
#     cv2.imwrite(temp_dir, sxy*150)
#     cv2.imwrite(temp_dir, sxy_combined_dir*150)
#     cv2.imwrite(temp_dir, hls_w_y_thres*150)
#     cv2.imwrite(temp_dir, combined_binary*150)
            
#     return combined_binary


def create_queue(length = 10):
    return deque(maxlen=length)



class LANE_LINE:
    def __init__(self):
        
        self.polynomial_coeff = None
        self.line_fit_x = None
        self.non_zero_x = []
        self.non_zero_y = []
        self.windows = []



class LANE_HISTORY:
    def __init__(self, queue_depth=3, test_points=[50, 300, 500, 700], poly_max_deviation_distance=150):
        self.lane_lines = create_queue(queue_depth)
        self.smoothed_poly = None
        self.test_points = test_points
        self.poly_max_deviation_distance = poly_max_deviation_distance
        self.lost = False
        self.lost_count = 0
        self.max_lost_count = queue_depth
    
    def append(self, lane_line, force=False):
        if len(self.lane_lines) == 0 or (self.lost_count > self.max_lost_count ) or force:
            self.lane_lines = create_queue(self.max_lost_count)
            self.smoothed_poly = None
            self.lane_lines.append(lane_line)
            self.get_smoothed_polynomial()
            self.lost =  False
            self.lost_count = 0
            print("**** RESET****")
            return True
        test_y_smooth = np.asarray(list(map(lambda x: np.polyval(self.smoothed_poly,x), self.test_points)))
        test_y_new = np.asarray(list(map(lambda x: np.polyval(lane_line.polynomial_coeff,x), self.test_points)))
        dist = np.absolute(test_y_smooth - test_y_new)
        max_dist = dist[np.argmax(dist)]
        if max_dist > self.poly_max_deviation_distance:
            print("**** MAX DISTANCE BREACHED ****")
            # print("y_smooth={0} - y_new={1} - distance={2} - max-distance={3}".format(test_y_smooth, test_y_new, max_dist, self.poly_max_deviation_distance))
            self.lost =  True
            self.lost_count += 1 
            return False  
        # self.lane_lines.append(lane_line)

        print("#####APPENDED######")
        self.get_smoothed_polynomial()
        self.lost =  False
        self.lost_count = 0
        return True
    
    def get_smoothed_polynomial(self):
        all_coeffs = np.asarray(list(map(lambda lane_line: lane_line.polynomial_coeff, self.lane_lines)))
        self.smoothed_poly = np.mean(all_coeffs, axis=0)
        return self.smoothed_poly



class LANE_DETECTION:
    """
    The AdvancedLaneDetectorWithMemory is a class that can detect lines on the road
    """
    UNWARPED_SIZE :(int,int)
    WRAPPED_WIDTH  :  int
    _pip_size=(int,int)
    _pip__x_offset=20
    _pip__y_offset=10
    img_dimensions=(int,int)
    temp_dir = "./images/detection/"
    windows_per_line = 15
    vanishing_point:(int,int)
    real_world_lane_size_meters=(32, 3.7)
    def __init__(self,  img ):
        self.objpts = None
        self.imgpts = None
        self.lane_roi = None
        # IMAGE PROPERTIES
        self.image =  img
        self.img_dimensions =  (self.image.shape[0], self.image.shape[1]) 
        self.UNWARPED_SIZE  = (int(self.img_dimensions[1]*1),int(self.img_dimensions[1]*0.95))
        self.WRAPPED_WIDTH =  int(self.img_dimensions[1]*0.1)
        self.calc_perspective()
        
        self.window_half_width = int(self.img_dimensions[1]*0.06)
        self.lb = int(0.20*self.img_dimensions[1])
        self.ub = int(0.35*self.img_dimensions[1])
        x =  np.linspace(0,self.ub -self.lb-1, self.ub -self.lb)
        self.parabola = -5*x*(x+self.lb -self.ub)
        self._pip_size = (int(self.image.shape[1] * 0.2), int(self.image.shape[0]*0.2))
        self.sliding_window_recenter_thres=int(self.window_half_width*0.1)
        # LANE PROPERTIES
        # We can pre-compute some data here
        # self.ym_per_px = self.real_world_lane_size_meters[0] / self.img_dimensions[0]
        # self.xm_per_px = self.real_world_lane_size_meters[1] / self.lane_width_px
        self.ploty = np.linspace(0, self.UNWARPED_SIZE[0] - 1, self.UNWARPED_SIZE[0])
        self.previous_left_lane_line = None
        self.previous_right_lane_line = None
        self.previous_left_lane_lines = LANE_HISTORY()
        self.previous_right_lane_lines = LANE_HISTORY()
        self.total_img_count = 0
        self.margin_red = 0.995
        
    def calc_perspective(self, verbose =  True):
        roi = np.zeros((self.img_dimensions[0], self.img_dimensions[1]), dtype=np.uint8) # 720 , 1280
        roi_points = np.array([[0, self.img_dimensions[0]*2//3],
                    [0, self.img_dimensions[0]],
                    [self.img_dimensions[1],self.img_dimensions[0]],
                    [self.img_dimensions[1], self.img_dimensions[0]*2//3],
                    [self.img_dimensions[1]*7//11,self.img_dimensions[0]//2],
                    [self.img_dimensions[1]*5//11,self.img_dimensions[0]//2]], dtype=np.int32)
        cv2.fillPoly(roi, [roi_points], 1)
        x = np.linspace(0,self.img_dimensions[0]-1,self.img_dimensions[0])
        grad= np.tile(5*x,self.img_dimensions[1]).reshape((self.img_dimensions[0], self.img_dimensions[1]))

        self.lane_roi = np.zeros((self.img_dimensions[0], self.img_dimensions[1]), dtype=np.uint8)
        Lhs = np.zeros((2,2), dtype= np.float32)
        Rhs = np.zeros((2,1), dtype= np.float32)
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mn_hsl = np.median(grey) #grey.median()
        edges = cv2.Canny(grey, int(mn_hsl*2), int(mn_hsl*.4))
        # edges = cv2.Canny(grey[:, :, 1], 500, 400)

        # cv2.imwrite(self.temp_dir+"mask.jpg", grey*roi)
        # cv2.imwrite(self.temp_dir+"mask.jpg", edges*roi)

        lines = cv2.HoughLinesP(edges*roi,rho = 5,theta = np.pi/180,threshold = 20,minLineLength = 150,maxLineGap = 20)

       
        # print(lines)
        for line in lines:
            
            for x1, y1, x2, y2 in line:
                normal = np.array([[-(y2-y1)], [x2-x1]], dtype=np.float32)
                normal /=np.linalg.norm(normal)
                point = np.array([[x1],[y1]], dtype=np.float32)
                outer = np.matmul(normal, normal.T)
                Lhs += outer
                Rhs += np.matmul(outer, point)
        self.vanishing_point= np.matmul(np.linalg.inv(Lhs),Rhs).reshape(2)
        top =self.vanishing_point[1] + 20
        bottom = self.img_dimensions[0]+200
        lane_roi_points = np.array([
                    [self.img_dimensions[1]*6//80, self.img_dimensions[0]],
                    [self.img_dimensions[1]*74//80,self.img_dimensions[0]],
                    [self.vanishing_point[0] + 2*self.img_dimensions[1]//25,self.vanishing_point[1] - 10],
                    [self.vanishing_point[0] - 2*self.img_dimensions[1]//25,self.vanishing_point[1] - 10]], dtype=np.int32)
        cv2.fillPoly(self.lane_roi , [lane_roi_points], 1)
        self.lane_roi =  self.lane_roi*grad

        def on_line(p1, p2, ycoord):
            return [p1[0]+ (p2[0]-p1[0])/float(p2[1]-p1[1])*(ycoord-p1[1]), ycoord]


        #define source and destination targets
        p1 = [self.vanishing_point[0] - self.WRAPPED_WIDTH/2, top]
        p2 = [self.vanishing_point[0] + self.WRAPPED_WIDTH/2, top]
        p3 = on_line(p2,self.vanishing_point, bottom)
        p4 = on_line(p1,self.vanishing_point, bottom)
        src_points = np.array([p1,p2,p3,p4], dtype=np.float32)
        # print(src_points,vanishing_point)
        dst_points = np.array([[0, 0], [self.UNWARPED_SIZE[0], 0],
                            [self.UNWARPED_SIZE[0], self.UNWARPED_SIZE[1]],
                            [0, self.UNWARPED_SIZE[1]]], dtype=np.float32)
        print(src_points,dst_points)
        self.trans_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inv_trans_mat = cv2.getPerspectiveTransform(dst_points,src_points)
        min_wid = 1000
        img = cv2.warpPerspective(self.image, self.trans_mat, self.UNWARPED_SIZE)
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        mask = grey[:,:,1]>128
        mask[:, :50]=0
        mask[:, -50:]=0
        x = np.linspace(0,mask.shape[0]-1,mask.shape[0])
        grad= np.tile(5*x,mask.shape[1]).reshape((mask.shape[0], mask.shape[1]))

        mask =  mask * grad
        histogram = np.sum(mask[mask.shape[0]//2:,:], axis=0)
        # delta1 =  histogram[:-1]-histogram[1:]
        # delta2 =  delta1[:-1]-delta1[1:]

        x1 = np.argmax(histogram[:histogram.shape[0]//2])
        x2 = histogram.shape[0]//2  +np.argmax(histogram[histogram.shape[0]//2 :])

        if (x2-x1<min_wid):
            min_wid = x2-x1
        self.px_per_xm = min_wid/self.real_world_lane_size_meters[1]
        self.xm_per_px =  1/self.px_per_xm
        if False :#self.camera.callibration_done :
            Lh = 1#np.linalg.inv(np.matmul(self.trans_mat, self.camera.cam_matrix))
        else:
            Lh = np.linalg.inv(self.trans_mat)
        self.px_per_ym = self.px_per_xm * np.linalg.norm(Lh[:,0]) / np.linalg.norm(Lh[:,1])
        self.ym_per_px =  1/self.px_per_ym
        self.perspective_done_at =  datetime.utcnow().timestamp()
        if verbose :       
            img_orig = cv2.polylines(self.image, [src_points.astype(np.int32)],True, (0,0,255), thickness=5)
            cv2.line(img, (int(x1), 0), (int(x1), self.UNWARPED_SIZE[1]), (255, 0, 0), 3)
            cv2.line(img, (int(x2), 0), (int(x2), self.UNWARPED_SIZE[1]), (0, 0, 255), 3)

            cv2.circle(img_orig,tuple(self.vanishing_point),10, color=(0,0,255), thickness=5)
            cv2.imwrite(self.temp_dir+"perspective1.jpg",img_orig)
            cv2.imwrite(self.temp_dir+"perspective2.jpg",img)
            return img_orig
        return
        
    
    def process_image(self, img):
        """
        Attempts to find lane lines on the given image and returns an image with lane area colored in green
        as well as small intermediate images overlaid on top to understand how the algorithm is performing
        """
        # First step - undistort the image using the instance's object and image points
        # undist_img = undistort_image(img, self.objpts, self.imgpts)
        undist_img =  img.copy()
        # Produce binary thresholded image from color and gradients
        # thres_img = get_combined_binary_thresholded_img(undist_img)
        thres_img=compute_hls_white_yellow_binary(undist_img)
        thresh_img_roi = thres_img * self.lane_roi
        # cv2.imwrite(self.temp_dir+"frame.jpg",undist_img* np.dstack((self.lane_roi>0,self.lane_roi>0,self.lane_roi>0)))
        # Create the undistorted and binary perspective transforms
        img_size = (undist_img.shape[1], undist_img.shape[0])
        undist_img_psp = cv2.warpPerspective(undist_img, self.trans_mat, img_size, flags=cv2.INTER_LINEAR)
        thres_img_psp = cv2.warpPerspective(thresh_img_roi, self.trans_mat, (self.UNWARPED_SIZE[1],self.UNWARPED_SIZE[0]))# img_size,)# flags=cv2.INTER_LINEAR)
        ll, rl = self.compute_lane_lines(thres_img_psp)
        lcr, rcr, lco = self.compute_lane_curvature(ll, rl)

        drawn_lines = self.draw_lane_lines(thres_img_psp, ll, rl)        
        # plt.imshow(drawn_lines)
        
        drawn_lines_regions = self.draw_lane_lines_regions(thres_img_psp, ll, rl)
        # plt.imshow(drawn_lines_regions)
        
        drawn_lane_area = self.draw_lane_area(thres_img_psp, undist_img, ll, rl)        
        # plt.imshow(drawn_lane_area)
        
        drawn_hotspots = self.draw_lines_hotspots(thres_img_psp, ll, rl)
        
        combined_lane_img = self.combine_images(drawn_lane_area, drawn_lines, drawn_lines_regions, drawn_hotspots, undist_img_psp)
        final_img = self.draw_lane_curvature_text(combined_lane_img, lcr, rcr, lco)
        
        self.total_img_count += 1
        self.previous_left_lane_line = ll
        self.previous_right_lane_line = rl
        
        return final_img
    
    def draw_lane_curvature_text(self, img, left_curvature_meters, right_curvature_meters, center_offset_meters):
        """
        Returns an image with curvature information inscribed
        """
        
        offset_y = self._pip_size[1] * 1 + self._pip__y_offset * 5
        offset_x = self._pip__x_offset
        
        template = "{0:17}{1:17}{2:17}"
        txt_header = template.format("Left Curvature", "Right Curvature", "Center Alignment") 
        # print(txt_header)
        txt_values = template.format("{:.0f}m".format(left_curvature_meters), 
                                     "{:.0f}m".format(right_curvature_meters),
                                     "{:.2f}m Right".format(center_offset_meters))
        if center_offset_meters < 0.0:
            txt_values = template.format("{:.0f}m".format(left_curvature_meters), 
                                     "{:.0f}m".format(right_curvature_meters),
                                     "{:.2f}m Left".format(math.fabs(center_offset_meters)))
            
        
        # print(txt_values)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, txt_header, (offset_x, offset_y), font, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, txt_values, (offset_x, offset_y + self._pip__y_offset * 5), font, 1, (255,255,255), 2, cv2.LINE_AA)
        
        return img
    
    def combine_images(self, lane_area_img, lines_img, lines_regions_img, lane_hotspots_img, psp_color_img):        
        """
        Returns a new image made up of the lane area image, and the remaining lane images are overlaid as
        small images in a row at the top of the the new image
        """
        small_lines = cv2.resize(lines_img, self._pip_size)
        small_region = cv2.resize(lines_regions_img, self._pip_size)
        small_hotspots = cv2.resize(lane_hotspots_img, self._pip_size)
        small_color_psp = cv2.resize(psp_color_img, self._pip_size)
                
        lane_area_img[self._pip__y_offset: self._pip__y_offset + self._pip_size[1], self._pip__x_offset: self._pip__x_offset + self._pip_size[0]] = small_lines
        
        start_offset_y = self._pip__y_offset 
        start_offset_x = 2 * self._pip__x_offset + self._pip_size[0]
        lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0]] = small_region
        
        start_offset_y = self._pip__y_offset 
        start_offset_x = 3 * self._pip__x_offset + 2 * self._pip_size[0]
        lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0]] = small_hotspots

        start_offset_y = self._pip__y_offset 
        start_offset_x = 4 * self._pip__x_offset + 3 * self._pip_size[0]
        # print(lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0],:].shape)
        # lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0],:] = small_color_psp
        
        
        return lane_area_img
    
        
    def draw_lane_area(self, warped_img, undist_img, left_line, right_line):
        """
        Returns an image where the inside of the lane has been colored in bright green
        """
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_line.line_fit_x, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.line_fit_x, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # cv2.imwrite(self.temp_dir+"frame.jpg", color_warp)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.inv_trans_mat, (undist_img.shape[1], undist_img.shape[0])) 
        # cv2.imwrite(self.temp_dir+"frame.jpg", newwarp)
        # cv2.imwrite(self.temp_dir+"frame.jpg", warped_img*125)
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
        # cv2.imwrite(self.temp_dir+"frame.jpg", result)
        return result
        
        
    def draw_lane_lines(self, warped_img, left_line, right_line):
        """
        Returns an image where the computed lane lines have been drawn on top of the original warped binary image
        """
        # Create an output image with 3 colors (RGB) from the binary warped image to draw on and  visualize the result
        out_img = np.dstack((warped_img, warped_img, warped_img))*255
        
        # Now draw the lines
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        pts_left = np.dstack((left_line.line_fit_x, ploty)).astype(np.int32)
        pts_right = np.dstack((right_line.line_fit_x, ploty)).astype(np.int32)

        cv2.polylines(out_img, pts_left, False,  (255, 140,0), 5)
        cv2.polylines(out_img, pts_right, False, (255, 140,0), 5)
        
        for low_pt, high_pt in left_line.windows:
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)

        for low_pt, high_pt in right_line.windows:            
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)           
        
        return out_img    
    
    def draw_lane_lines_regions(self, warped_img, left_line, right_line):
        """
        Returns an image where the computed left and right lane areas have been drawn on top of the original warped binary image
        """
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        margin = self.window_half_width
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        
        left_line_window1 = np.array([np.transpose(np.vstack([left_line.line_fit_x - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_line.line_fit_x + margin, 
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        
        right_line_window1 = np.array([np.transpose(np.vstack([right_line.line_fit_x - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_line.line_fit_x + margin, 
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Create RGB image from binary warped image
        region_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        # Draw the lane onto the warped blank image
        cv2.fillPoly(region_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(region_img, np.int_([right_line_pts]), (0, 255, 0))
        
        return region_img


    def draw_lines_hotspots(self, warped_img, left_line, right_line):
        """
        Returns a RGB image where the portions of the lane lines that were
        identified by our pipeline are colored in yellow (left) and blue (right)
        """
        out_img = np.dstack((warped_img, warped_img, warped_img))*255
        
        out_img[left_line.non_zero_y, left_line.non_zero_x] = [255, 255, 0]
        out_img[right_line.non_zero_y, right_line.non_zero_x] = [0, 0, 255]
        
        return out_img

    def compute_lane_curvature(self, left_line, right_line):
        """
        Returns the triple (left_curvature, right_curvature, lane_center_offset), which are all in meters
        """        
        ploty = self.ploty
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        
        leftx = left_line.line_fit_x
        rightx = right_line.line_fit_x
        
        # Fit new polynomials: find x for y in real-world space
        left_fit_cr = np.polyfit(ploty * self.ym_per_px, leftx * self.xm_per_px, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_px, rightx * self.xm_per_px, 2)
        
        # Now calculate the radii of the curvature
        left_curverad = int((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_px + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = int((1 + (2 *right_fit_cr[0] * y_eval * self.ym_per_px + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        
        # Use our computed polynomial to determine the car's center position in image space, then
        left_fit = left_line.polynomial_coeff
        right_fit = right_line.polynomial_coeff
        
        center_offset_img_space = (((left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]) + 
                   (right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2])) / 2) - self.vanishing_point[0]
        center_offset_real_world_m = int(center_offset_img_space * self.xm_per_px*100)/100.0
        
        # Now our radius of curvature is in meters        
        return left_curverad, right_curverad, center_offset_real_world_m
        
    def detect_lane_start(self, image):
        histx =   np.sum(image[image.shape[0]*2//3:,:], axis=0)
        histx = histx * self.parabola

        # fig, ax = plt.subplots(1, 3, figsize=(15,4))
        # ax[0].imshow(image, cmap='gray')
        # ax[0].axis("off")
        # ax[0].set_title("Binary Thresholded Perspective Transform Image")

        # ax[1].plot(histogram)
        # ax[2].plot(histx)
        # plt.show()
        return np.argmax(histx)
    
        
    def compute_lane_lines(self, warped_img):
        """
        Returns the tuple (left_lane_line, right_lane_line) which represents respectively the LANE_LINE instances for
        the computed left and right lanes, for the supplied binary warped image
        """
        margin = self.window_half_width
        margin15 = int(margin*1 )
        minpix = self.sliding_window_recenter_thres
        window_height = np.int(warped_img.shape[0]//self.windows_per_line)
        # Take a histogram of the bottom half of the image, summing pixel values column wise 
              

        # Set height of windows
        
        # Identify the x and y positions of all nonzero pixels in the image
        # NOTE: nonzero returns a tuple of arrays in y and x directions
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        total_non_zeros = len(nonzeroy)
        non_zero_found_pct = 0.0
        left_lane_inds = []
        right_lane_inds = []
        left_line = LANE_LINE()
        right_line = LANE_LINE()                 
        if self.previous_left_lane_line is not None and self.previous_right_lane_line is not None:
            left_lane_inds = ((nonzerox > (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                           + self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy 
                                           + self.previous_left_lane_line.polynomial_coeff[2] - margin15)) 
                              & (nonzerox < (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                            + self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy 
                                            + self.previous_left_lane_line.polynomial_coeff[2] + margin15))) 

            right_lane_inds = ((nonzerox > (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                           + self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy 
                                           + self.previous_right_lane_line.polynomial_coeff[2] - margin15)) 
                              & (nonzerox < (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                            + self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy 
                                            + self.previous_right_lane_line.polynomial_coeff[2] + margin15))) 
            
            non_zero_found_left = np.sum(left_lane_inds)
            non_zero_found_right = np.sum(right_lane_inds)
            non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros
            print("[Previous lane] Found pct={0}".format(non_zero_found_pct))
        if non_zero_found_pct < 0.8:
            histogram = np.sum(warped_img[warped_img.shape[0]*2//3:,:], axis=0)
            # histogram = histogram * self.parabola
            midpoint = self.UNWARPED_SIZE[0]//2
            leftx_base = midpoint-self.ub + self.detect_lane_start(warped_img[:,midpoint-self.ub :midpoint-self.lb])
            rightx_base = midpoint+self.lb + self.detect_lane_start(warped_img[:,midpoint+self.lb :midpoint+self.ub])

            # leftx_base = midpoint-ub+ np.argmax(histogram[midpoint-ub :midpoint-lb])
            # rightx_base = midpoint+lb + np.argmax(histogram[midpoint+lb:midpoint+ub])  
    
            leftx_current = leftx_base
            rightx_current = rightx_base 
            print("Non zeros found below thresholds, begining sliding window - pct={0}".format(non_zero_found_pct))
            left_lane_inds = []
            right_lane_inds = []
            left_centers = []
            right_centers = []
            left_center_idx = []
            right_center_idx =[]
            doleft = True
            doright = True
            # Step through the windows one by one
            for window in range(self.windows_per_line):
                # Identify window boundaries in x and y (and right and left)
                # We are moving our windows from the bottom to the top of the screen (highest to lowest y value)
                win_y_low = warped_img.shape[0] - (window + 1)* window_height
                win_y_high = warped_img.shape[0] - window * window_height

                # Defining our window's coverage in the horizontal (i.e. x) direction 
                # Notice that the window's width is twice the margin
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                if (win_xleft_low > 0) & (win_xleft_high < warped_img.shape[1]) & doleft:
                    left_line.windows.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
                    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                    if len(good_left_inds) > minpix:
                        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                        left_lane_inds.append(good_left_inds)
                        left_centers.append(leftx_current)
                        left_center_idx.append(window)
                    elif len(left_center_idx) > 3 :
                        left_coef = np.polyfit(np.array(left_center_idx),np.array(left_centers),2)
                        leftx_current = int(np.polyval(left_coef, window+1))
                else :
                    doleft = False

                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                if (win_xright_high > 0 )& (win_xright_low < warped_img.shape[1]) & doright:
                    right_line.windows.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])            
                    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                    if len(good_right_inds) > minpix:        
                        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                        right_lane_inds.append(good_right_inds)
                        right_centers.append(rightx_current)
                        right_center_idx.append(window)
                    elif len(right_center_idx) > 3 :
                        right_coef = np.polyfit(np.array(right_center_idx),np.array(right_centers),2)
                        rightx_current = int(np.polyval(right_coef, window+1))
                else :
                    doright = False

                margin =  int(margin * self.margin_red)
            if len(left_lane_inds)>0:
                left_lane_inds = np.concatenate(left_lane_inds)
            if len(right_lane_inds)>0:
                right_lane_inds = np.concatenate(right_lane_inds)
            
            non_zero_found_left = np.sum(left_lane_inds)
            non_zero_found_right = np.sum(right_lane_inds)
            non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros
            # print("[Sliding windows] Found pct={0}".format(non_zero_found_pct))
        else :
            x =  np.arange(len(right_lane_inds))
            right_lane_inds  =  x[right_lane_inds]
            x = np.arange(len(left_lane_inds))
            left_lane_inds = x[left_lane_inds]    
        if len(left_lane_inds)>0:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            left_fit = np.polyfit(lefty, leftx, 2)
            left_line.polynomial_coeff = left_fit
            if not self.previous_left_lane_lines.append(left_line):
                left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
                left_line.polynomial_coeff = left_fit
                # self.previous_left_lane_lines.append(left_line, force=True)
                # print("**** REVISED Poly left {0}".format(left_fit))  
            else:
                left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
                left_line.polynomial_coeff = left_fit  
                leftx =[]
                lefty =[]        
        else:
            left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
            left_line.polynomial_coeff = left_fit
            leftx =[]
            lefty =[] 
        
        if len(right_lane_inds)>0:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 
            right_fit = np.polyfit(righty, rightx, 2)
            right_line.polynomial_coeff = right_fit
            if not self.previous_right_lane_lines.append(right_line):
                right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
                right_line.polynomial_coeff = right_fit
                # self.previous_right_lane_lines.append(right_line, force=True)
                # print("**** REVISED Poly right {0}".format(right_fit))
            else:
                right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
                right_line.polynomial_coeff = right_fit
                rightx = []
                righty = []
        else:
            right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
            right_line.polynomial_coeff = right_fit
            rightx = []
            righty = []

        
        
        
        


        


    
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0] )
        
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)
        
        
        left_line.polynomial_coeff = left_fit
        left_line.line_fit_x = left_fitx
        left_line.non_zero_x = leftx  
        left_line.non_zero_y = lefty

        right_line.polynomial_coeff = right_fit
        right_line.line_fit_x = right_fitx
        right_line.non_zero_x = rightx
        right_line.non_zero_y = righty

        
        return (left_line, right_line)

if __name__ == "__main__":
    # img =  cv2.imread("./images/straight_lines1.jpg")
    # ld = LANE_DETECTION( img)
    # image =  cv2.imread("./images/test5.jpg")
    # center =  (image.shape[1]//4,image.shape[0]-100 )
    # cv2.circle(image,center,20, (66, 244, 238),-1)
    # proc_img = ld.process_image(image)
    # cv2.imwrite("./images/detection/frame.jpg",proc_img)

    video_reader =  cv2.VideoCapture("videos/harder_challenge_video.mp4") 
    fps =  video_reader.get(cv2.CAP_PROP_FPS)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_out = "videos/output10.mov"
    # cv2.VideoWriter_fourcc(*'MPEG')
    video_writer = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_w, frame_h))
    pers_frame_time = 0.1# seconds
    pers_frame = int(pers_frame_time *fps)
    video_reader.set(1,pers_frame)
    ret, image = video_reader.read()
    ld = LANE_DETECTION( image)
    pers_frame_time = 1# seconds
    pers_frame = int(pers_frame_time *fps)
    video_reader.set(1,pers_frame)
    ret, image = video_reader.read()
    proc_img = ld.process_image(image)
    cv2.imwrite("./images/detection/frame.jpg",proc_img)



