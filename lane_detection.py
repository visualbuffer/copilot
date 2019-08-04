import cv2
import numpy as np
import math
from datetime import datetime
from matplotlib import pyplot as plt
from collections import deque
from scipy.stats  import mode
temp_dir = "images/detection/detect.jpg"
def compute_hls_white_yellow_binary(image,kernel_size =3):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB formats
    """
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower = np.uint8([ 20, 50,   40])
    upper = np.uint8([35, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)   
    lower = np.uint8([ 0, 200,   0])
    upper = np.uint8([180, 255, 80])
    white_mask = cv2.inRange(converted, lower, upper)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return mask


def create_queue(length = 10):
    return deque(maxlen=length)



class LANE_LINE:
    def __init__(self):
        
        self.polynomial_coeff = None
        self.line_fit_x = None
        self.non_zero_x = np.array([])
        self.non_zero_y = np.array([])
        self.windows = []

    def purge_points(self, xlim):
        p  =  self.non_zero_x < xlim
        q =  self.non_zero_x > 0
        r = np.logical_and(p, q)
        self.non_zero_x = self.non_zero_x[r]
        self.non_zero_y = self.non_zero_y[r]
        return




class LANE_HISTORY:
    def __init__(self, queue_depth=12, test_points=[300, 500, 700], poly_max_deviation_distance=100, smoothing = 10):
        self.lane_lines = create_queue(queue_depth)
        self.smoothed_poly = None
        self.test_points = np.asarray(test_points)
        self.poly_max_deviation_distance = poly_max_deviation_distance
        self.lost = False
        self.lost_count = 0
        self.max_lost_count = queue_depth
        self.smoothing =  smoothing
    
    def addlane(self, lane_line, skip=True):
        status = "APPENDED | "
        if (not skip) and (len(self.lane_lines) == 0 or (self.lost_count > self.max_lost_count )) :
            # self.lane_lines = create_queue(self.max_lost_count)
            status ="RESET | "
            self.lane_lines.append(lane_line)
            self.get_smoothed_polynomial()
            self.lost =  False
            self.lost_count = 0
            print(status)
            return True, status
        test_y_smooth = np.asarray(list(map(lambda x: np.polyval(self.smoothed_poly,x), -self.test_points)))
        test_y_new = np.asarray(list(map(lambda x: np.polyval(lane_line.polynomial_coeff,x), -self.test_points)))
        dist = np.absolute(test_y_smooth - test_y_new)
        max_dist = dist[np.argmax(dist)]
        if max_dist > self.poly_max_deviation_distance:
            status = "BREACHED | "
            print(status)
            # print("y_smooth={0} - y_new={1} - distance={2} - max-distance={3}".format(test_y_smooth, test_y_new, max_dist, self.poly_max_deviation_distance))
            self.lost =  True
            self.lost_count += 1 
            return False , status 
        # self.lane_lines.append(lane_line)

        print(status)
        self.get_smoothed_polynomial()
        self.lost =  False
        self.lost_count = 0
        return True, status
    
    def get_smoothed_polynomial(self):
        all_coeffs = np.asarray(list(map(lambda lane_line: lane_line.polynomial_coeff, self.lane_lines)))
        self.smoothed_poly = np.mean(all_coeffs[-self.smoothing:,:], axis=0)
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
    windows_per_line = 27
    vanishing_point:(int,int)
    real_world_lane_size_meters=(32, 3.7)
    ego_vehicle_in_frame=False
    def __init__(self,  img,fps ):
        self.objpts = None
        self.fps=int(fps)
        self.imgpts = None
        self.lane_roi = None
        self.lane_width = create_queue(3*self.fps)
        # IMAGE PROPERTIES
        self.image =  img
        self.img_dimensions =  (self.image.shape[0], self.image.shape[1]) 
        self.UNWARPED_SIZE  = (int(self.img_dimensions[1]*1),int(self.img_dimensions[1]*0.95))
        self.WRAPPED_WIDTH =  int(self.img_dimensions[1]*0.08)
        self.window_half_width = int(self.img_dimensions[1]*0.07)
        self.lb = int(0.02*self.img_dimensions[1])
        self.ub = int(0.32*self.img_dimensions[1])
        x =  np.linspace(0,self.img_dimensions[1]-1, self.img_dimensions[1])
        self.parabola = -200*x*(x-self.img_dimensions[1])
        self._pip_size = (int(self.image.shape[1] * 0.2), int(self.image.shape[0]*0.2))
        self.sliding_window_recenter_thres=9
        x =  np.linspace(0,self.ub -self.lb-1, self.ub -self.lb)
        self.lanebola = -50*x*(x+self.lb -self.ub)
        self.calc_perspective()

        if self.ego_vehicle_in_frame :
            self.windows_range = range(int(self.windows_per_line*0.1), self.windows_per_line, 1)
        else:
            self.windows_range = range( self.windows_per_line)
        self.ploty = np.linspace(self.UNWARPED_SIZE[1] //2, self.UNWARPED_SIZE[1]- 1, self.UNWARPED_SIZE[1]//2)
        self.previous_left_lane_line = None
        self.previous_right_lane_line = None
        test = np.arange(0.6,1,0.05)*self.img_dimensions[0]
        test = test.astype(int)
        self.previous_left_lane_lines = LANE_HISTORY(test_points = test, queue_depth=self.fps//5)
        self.previous_right_lane_lines = LANE_HISTORY(test_points = test, queue_depth=self.fps//5)
        self.total_img_count = 0
        self.margin_red = 1# .995
        self.tot_key_pts = create_queue(self.fps*3//2)
        self.message = ""


    def calc_perspective(self, verbose =  True):
        roi = np.zeros((self.img_dimensions[0], self.img_dimensions[1]), dtype=np.uint8) # 720 , 1280
        roi_points = np.array([[0, self.img_dimensions[0]*12//13],
                    # [0, self.img_dimensions[0]],
                    # [self.img_dimensions[1],self.img_dimensions[0]],
                    [self.img_dimensions[1], self.img_dimensions[0]*12//13],
                    [self.img_dimensions[1]*13//23,self.img_dimensions[0]*13//23],
                    [self.img_dimensions[1]*11//23,self.img_dimensions[0]*13//23]], dtype=np.int32)
        cv2.fillPoly(roi, [roi_points], 1)
        x = np.linspace(0,self.img_dimensions[0]-1,self.img_dimensions[0])
        grad= np.tile(-5*x*(x-self.img_dimensions[1]),self.img_dimensions[1]).reshape((self.img_dimensions[0], self.img_dimensions[1]))

        self.lane_roi = np.zeros((self.img_dimensions[0], self.img_dimensions[1]), dtype=np.uint8)
        Lhs = np.zeros((2,2), dtype= np.float32)
        Rhs = np.zeros((2,1), dtype= np.float32)
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mn_hsl = np.median(grey) #grey.median()
        edges = cv2.Canny(grey, int(mn_hsl*2), int(mn_hsl*.4))
        # edges = cv2.Canny(grey[:, :, 1], 500, 400)

        cv2.imwrite(self.temp_dir+"mask.jpg", grey*roi)
        cv2.imwrite(self.temp_dir+"mask.jpg", edges*roi)

        lines = cv2.HoughLinesP(edges*roi,rho =19,theta = 3* np.pi/180,threshold = 7,minLineLength = 200,maxLineGap = 25)

        img2 =  self.image.copy()
        # print(lines)
        for line in lines:
           
            for x1, y1, x2, y2 in line:
                cv2.line(img2,(x1,y1),(x2,y2),(255,0,0),2)
                normal = np.array([[-(y2-y1)], [x2-x1]], dtype=np.float32)
                normal /=np.linalg.norm(normal)
                point = np.array([[x1],[y1]], dtype=np.float32)
                outer = np.matmul(normal, normal.T)
                Lhs += outer
                Rhs += np.matmul(outer, point)
        self.vanishing_point= np.matmul(np.linalg.inv(Lhs),Rhs).reshape(2)
        if abs(self.vanishing_point[0] - self.img_dimensions[1]//2) > 0.01 *self.img_dimensions[1] :
            print("ABSURD POSITION TRY OTHER PARAMETERS")
            self.vanishing_point[0] = self.img_dimensions[1]//2
        top =self.vanishing_point[1] + self.WRAPPED_WIDTH//12
        bottom = self.img_dimensions[0]+200
        lane_roi_points = np.array([
                    [self.img_dimensions[1]*6//80, self.img_dimensions[0]],
                    [self.img_dimensions[1]*74//80,self.img_dimensions[0]],
                    [self.vanishing_point[0] + 3*self.img_dimensions[1]//25,self.vanishing_point[1] - 10],
                    [self.vanishing_point[0] - 3*self.img_dimensions[1]//25,self.vanishing_point[1] - 10]], dtype=np.int32)
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
        self.trans_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inv_trans_mat = cv2.getPerspectiveTransform(dst_points,src_points)
        min_wid = 1000
        img = cv2.warpPerspective(self.image, self.trans_mat, self.UNWARPED_SIZE)
        mask = compute_hls_white_yellow_binary(img) 

        x = np.linspace(0,mask.shape[0]-1,mask.shape[0])


        # histx = histx * self.lanebola
        midpoint = self.UNWARPED_SIZE[0]//2
        print( midpoint,self.ub , self.detect_lane_start(mask[:,midpoint-self.ub :midpoint-self.lb]))
        x1 = midpoint-self.ub + self.detect_lane_start(mask[:,midpoint-self.ub :midpoint-self.lb])
        x2 = midpoint+self.lb + self.detect_lane_start(mask[:,midpoint+self.lb :midpoint+self.ub])

     
       

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
            cv2.imwrite(self.temp_dir+"perspective3.jpg",img2)
            return img_orig
        return
        
    
    def process_image(self, img):
        """
        Attempts to find lane lines on the given image and returns an image with lane area colored in green
        as well as small intermediate images overlaid on top to understand how the algorithm is performing
        """
        # First step - undistort the image using the instance's object and image points
        # undist_img = undistort_image(img, self.objpts, self.imgpts)
        self.message =""
        undist_img =  img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ll, rl , thres_img_psp= self.compute_lane_lines(undist_img)
        ll.purge_points(self.UNWARPED_SIZE[1])
        rl.purge_points(self.UNWARPED_SIZE[1])
        lcr, rcr, lco = self.compute_lane_curvature(ll, rl)
        out_img = np.dstack((thres_img_psp,thres_img_psp,thres_img_psp))

        drawn_lines = self.draw_lane_lines(out_img, ll, rl)        
        drawn_lines_regions = self.draw_lane_lines_regions(out_img, ll, rl)
        drawn_lane_area = self.draw_lane_area(out_img, undist_img, ll, rl)        
        drawn_hotspots = self.draw_lines_hotspots(out_img, ll, rl)
        combined_lane_img = self.combine_images(drawn_lane_area, drawn_lines, drawn_lines_regions,drawn_hotspots, img)
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
        cv2.putText(img, self.message, (offset_x, self.img_dimensions[0]-10), font, 1, (255,0,0), 1, cv2.LINE_AA)
        return img
    
    def combine_images(self, lane_area_img, lines_img, lines_regions_img,lane_hotspots_img, psp_color_img):        
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
        print(lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0],:].shape)
        lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0],:] = small_color_psp
        
        
        return lane_area_img
    
        
    def draw_lane_area(self, warped_img, undist_img, left_line, right_line):
        """
        Returns an image where the inside of the lane has been colored in bright green
        """
        color_warp = np.zeros_like(warped_img).astype(np.uint8)
        

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_line.line_fit_x, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.line_fit_x, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        newwarp = cv2.warpPerspective(color_warp, self.inv_trans_mat, (undist_img.shape[1], undist_img.shape[0])) 
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
        return result
        
        
    def draw_lane_lines(self, img, left_line, right_line):
        """
        Returns an image where the computed lane lines have been drawn on top of the original warped binary image
        """
        # Create an output image with 3 colors (RGB) from the binary warped image to draw on and  visualize the result
        # out_img = np.dstack((warped_img, warped_img, warped_img))*255
        
        # Now draw the lines
        out_img = img.copy()
        pts_left = np.dstack((left_line.line_fit_x, self.ploty)).astype(np.int32)
        pts_right = np.dstack((right_line.line_fit_x, self.ploty)).astype(np.int32)

        cv2.polylines(out_img, pts_left, False,  (255, 140,0), 5)
        cv2.polylines(out_img, pts_right, False, (255, 140,0), 5)
        
        for low_pt, high_pt in left_line.windows:
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)

        for low_pt, high_pt in right_line.windows:            
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)           
        return out_img    
    
    def draw_lane_lines_regions(self, img, left_line, right_line):
        """
        Returns an image where the computed left and right lane areas have been drawn on top of the original warped binary image
        """
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        warped_img = img.copy()
        margin = self.window_half_width

        
        left_line_window1 = np.array([np.transpose(np.vstack([left_line.line_fit_x - margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_line.line_fit_x + margin, 
                                      self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        
        right_line_window1 = np.array([np.transpose(np.vstack([right_line.line_fit_x - margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_line.line_fit_x + margin, 
                                      self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Create RGB image from binary warped image
        # region_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        # Draw the lane onto the warped blank image
        cv2.fillPoly(warped_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(warped_img, np.int_([right_line_pts]), (0, 255, 0))
        
        return warped_img


    def draw_lines_hotspots(self, img, left_line, right_line):
        """
        Returns a RGB image where the portions of the lane lines that were
        identified by our pipeline are colored in yellow (left) and blue (right)
        """
        # out_img = np.dstack((warped_img, warped_img, warped_img))*255
        out_img = img.copy()
        out_img[left_line.non_zero_y, left_line.non_zero_x] = [255, 255, 0]
        out_img[right_line.non_zero_y, right_line.non_zero_x] = [0, 0, 255]
        
        return out_img

    def compute_lane_curvature(self, left_line, right_line):
        """
        Returns the triple (left_curvature, right_curvature, lane_center_offset), which are all in meters
        """        
        y_eval = np.max(self.ploty)
        leftx = left_line.line_fit_x
        rightx = right_line.line_fit_x
        left_fit_cr = np.polyfit(-1*self.ploty * self.ym_per_px, leftx * self.xm_per_px, 2)
        right_fit_cr = np.polyfit(-1*self.ploty * self.ym_per_px, rightx * self.xm_per_px, 2)
        left_curverad = int((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_px + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = int((1 + (2 *right_fit_cr[0] * y_eval * self.ym_per_px + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        left_fit = left_line.polynomial_coeff
        right_fit = right_line.polynomial_coeff
        
        center_offset_img_space = (((left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]) + 
                   (right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2])) / 2) - self.vanishing_point[0]
        center_offset_real_world_m = int(center_offset_img_space * self.xm_per_px*100)/100.0       
        return left_curverad, right_curverad, center_offset_real_world_m
        
    def detect_lane_start(self, image):
        histx =   np.sum(image[image.shape[0]*3//4:,:], axis=0)
        histx = histx * self.lanebola
        return np.argmax(histx)
    
        
    def compute_lane_lines(self, img):
        """
        Returns the tuple (left_lane_line, right_lane_line) which represents respectively the LANE_LINE instances for
        the computed left and right lanes, for the supplied binary warped image
        """
        undst_img = compute_hls_white_yellow_binary(img)
        undst_img  = undst_img * self.lane_roi
        warped_img = cv2.warpPerspective(undst_img, self.trans_mat, (self.UNWARPED_SIZE[1],self.UNWARPED_SIZE[0]))
        margin = self.window_half_width
        margin15 = int(margin*1 )
        minpix = self.sliding_window_recenter_thres
        window_height = np.int(warped_img.shape[0]//self.windows_per_line)
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])   
        total_non_zeros = len(nonzeroy)
        self.tot_key_pts.append(total_non_zeros)
        skip = False
        if total_non_zeros /self.tot_key_pts[0] < 0.5 :
            skip = True
            self.message+="SKIPPED "
        non_zero_found_pct = 0.0
        left_lane_inds = []
        right_lane_inds = []
        left_line = LANE_LINE()
        right_line = LANE_LINE()                 
        if self.previous_left_lane_line is not None and self.previous_right_lane_line is not None:
            left_lane_inds = ((nonzerox > (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                           - self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy 
                                           + self.previous_left_lane_line.polynomial_coeff[2] - margin15)) 
                              & (nonzerox < (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                            - self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy 
                                            + self.previous_left_lane_line.polynomial_coeff[2] + margin15))) 

            right_lane_inds = ((nonzerox > (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                           - self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy 
                                           + self.previous_right_lane_line.polynomial_coeff[2] - margin15)) 
                              & (nonzerox < (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                            - self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy 
                                            + self.previous_right_lane_line.polynomial_coeff[2] + margin15))) 
            
            non_zero_found_left = np.sum(left_lane_inds)
            non_zero_found_right = np.sum(right_lane_inds)
            non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros
            print("[Previous lane] Found pct={0}".format(non_zero_found_pct),non_zero_found_left , non_zero_found_right,total_non_zeros)
        if non_zero_found_pct < 0.9:
            self.message += "| PCT "+ str(non_zero_found_pct)+" "
            midpoint = self.UNWARPED_SIZE[0]//2
            leftx_base = midpoint-self.ub + self.detect_lane_start(warped_img[:,midpoint-self.ub :midpoint-self.lb])
            rightx_base = midpoint+self.lb + self.detect_lane_start(warped_img[:,midpoint+self.lb :midpoint+self.ub])
            leftx_current = leftx_base
            rightx_current = rightx_base 
            self.lane_width.append(int(rightx_current - leftx_current))
            lane_width =  int(np.mean(self.lane_width))
            
            centerx_current = leftx_current + lane_width//2
            print("Non zeros found below thresholds, begining sliding window - pct={0}".format(non_zero_found_pct))
            left_lane_inds = []
            right_lane_inds = []
            centers = []
            center_idx = []
            leftx =[]
            rightx =[]
            lefty =[]
            righty =[]
            wleftx =[]
            wrightx=[]
            for window in self.windows_range:
                win_y_low = warped_img.shape[0] - (window + 1)* window_height
                win_y_high = warped_img.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                
                # if (win_xleft_low > 0) & (win_xleft_high < warped_img.shape[1]) & doleft:
                left_line.windows.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                right_line.windows.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])            
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                centerx =np.array([])


                if len(good_left_inds) > minpix:
                    leftx.extend(nonzerox[good_left_inds])
                    lefty.extend(-nonzeroy[good_left_inds])
                    
                    centerx = nonzerox[good_left_inds] + lane_width//2
                    rightx.extend(nonzerox[good_left_inds] + lane_width)
                    righty.extend(-nonzeroy[good_left_inds])
                    # centery = nonzeroy[good_left_inds]
                if len(good_right_inds) > minpix:
                    rightx.extend(nonzerox[good_right_inds])
                    righty.extend(-nonzeroy[good_right_inds])
                    centerx = np.append(centerx, nonzerox[good_right_inds] -lane_width//2)
                    leftx.extend(nonzerox[good_right_inds] - lane_width)
                    lefty.extend(-nonzeroy[good_right_inds])
                if (len(good_left_inds)> minpix )or (len(good_right_inds) > minpix ):
                    wleftx.extend([2000//(len(good_left_inds)+len(good_right_inds))]*(len(good_left_inds)+len(good_right_inds)))
                    

                # doleft =  True
                # if len(good_left_inds) > minpix:
                #     leftx.extend(nonzerox[good_left_inds])
                #     lefty.extend(nonzeroy[good_left_inds])
                #     wleftx.extend([2000//len(good_left_inds)]*len(good_left_inds))
                #     centerx = nonzerox[good_left_inds] +lane_width/2
                #     doleft  = False
                # if len(good_right_inds) > minpix:
                #     rightx.extend(nonzerox[good_right_inds])
                #     righty.extend(nonzeroy[good_right_inds])
                #     centerx = np.append(centerx, nonzerox[good_right_inds] -lane_width/2)
                #     wrightx.extend([2000//len(good_right_inds)]*len(good_right_inds))
                # elif len(good_left_inds) > minpix :
                #     rightx.extend(nonzerox[good_left_inds] + lane_width)
                #     righty.extend(nonzeroy[good_left_inds])
                #     wrightx.extend([2000//len(good_left_inds)]*len(good_left_inds))
                # if doleft & len(good_right_inds) > minpix:
                #     leftx.extend(nonzerox[good_right_inds] - lane_width)
                #     lefty.extend(nonzeroy[good_right_inds])
                #     wleftx.extend([2000//len(good_right_inds)]*len(good_right_inds))
                center_idx.append(window)
                if len(centerx) > minpix:    
                    centerx_current = np.int(mode(centerx)[0])
                    centers.append(centerx_current)
                    # center_idx.append(window)
                else :
                    centers.append(centerx_current)
                    if len(center_idx) > 5 :
                    
                        coef = np.polyfit(np.array(center_idx),np.array(centers),2)
                        centerx_current = int(np.polyval(coef, window+1))

                leftx_current = int(centerx_current -lane_width/2)
                rightx_current = int(centerx_current +lane_width/2)
                margin =  int(margin * self.margin_red)
            wrightx = wleftx
        else :
            x =  np.arange(len(right_lane_inds))
            right_lane_inds  =  x[right_lane_inds]
            x = np.arange(len(left_lane_inds))
            left_lane_inds = x[left_lane_inds] 
            lane_width =  int(np.mean(self.lane_width))

            # for window in self.windows_range :
            #     win_y_low = warped_img.shape[0] - (window + 1)* window_height
            #     win_y_high = warped_img.shape[0] - window * window_height
            #     good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            #                 (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            #     right_line.windows.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])            
            #     good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            #         (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            #     centerx =np.array([])

            leftx = np.array(nonzerox[left_lane_inds])
            lefty = -1*np.array(nonzeroy[left_lane_inds] )
            rightx = np.array(nonzerox[right_lane_inds])
            righty = -1* np.array(nonzeroy[right_lane_inds])
            leftx = np.concatenate((leftx, rightx-lane_width)) 
            lefty= np.concatenate((lefty, righty)) 
            rightx= np.concatenate((rightx, leftx+lane_width)) 
            righty= np.concatenate((righty, lefty)) 
            wleftx =None
            wrightx=None 

        if len(leftx)>minpix:
            left_fit = np.polyfit(lefty, leftx, 2,w=wleftx)
            left_line.polynomial_coeff = left_fit
            added,status = self.previous_left_lane_lines.addlane(left_line,skip = skip)
            self.message+= "LEFT" + status
            if not added:
                left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
                left_line.polynomial_coeff = left_fit
                # self.previous_left_lane_lines.append(left_line, force=True)
                # print("**** REVISED Poly left {0}".format(left_fit))  
            else:

                left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
                left_line.polynomial_coeff = left_fit         
        else:
            left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
            left_line.polynomial_coeff = left_fit
        
        if len(rightx)>minpix:
            right_fit = np.polyfit(righty, rightx, 2,w=wrightx)
            right_line.polynomial_coeff = right_fit
            added,status = self.previous_right_lane_lines.addlane(right_line,skip = skip)
            self.message+= "RIGHT" + status
            if not added:
                right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
                right_line.polynomial_coeff = right_fit
                # self.previous_right_lane_lines.append(right_line, force=True)
                # print("**** REVISED Poly right {0}".format(right_fit))
            else:
                right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
                right_line.polynomial_coeff = right_fit
        else:
            right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
            right_line.polynomial_coeff = right_fit
            rightx = []
            righty = []
        left_line.polynomial_coeff = left_fit
        left_line.line_fit_x = np.polyval(left_fit, -self.ploty )
        left_line.non_zero_x = np.array(leftx,np.int)  
        left_line.non_zero_y = -np.array(lefty,np.int)

        right_line.polynomial_coeff = right_fit
        right_line.line_fit_x = np.polyval(right_fit, -self.ploty )
        right_line.non_zero_x = np.array(rightx,np.int)
        right_line.non_zero_y =-np.array(righty,np.int)

        
        return (left_line, right_line,warped_img)

if __name__ == "__main__":
    # img =  cv2.imread("./images/straight_lines1.jpg")
    # ld = LANE_DETECTION( img)
    # image =  cv2.imread("./images/test5.jpg")
    # center =  (image.shape[1]//4,image.shape[0]-100 )
    # cv2.circle(image,center,20, (66, 244, 238),-1)
    # proc_img = ld.process_image(image)
    # cv2.imwrite("./images/detection/frame.jpg",proc_img)

    # video_reader =  cv2.VideoCapture("videos/harder_challenge_video.mp4") 
    video_reader =  cv2.VideoCapture("videos/challenge_video.mp4")
    fps =  video_reader.get(cv2.CAP_PROP_FPS)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_out = "videos/output10.mov"
    # cv2.VideoWriter_fourcc(*'MPEG')
    video_writer = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_w, frame_h))
    pers_frame_time = 10# seconds
    pers_frame = int(pers_frame_time *fps)
    video_reader.set(1,pers_frame)
    ret, image = video_reader.read()
    ld = LANE_DETECTION( image,fps)
    pers_frame_time = 4# seconds
    pers_frame = int(pers_frame_time *fps)
    video_reader.set(1,pers_frame)
    ret, image = video_reader.read()
    proc_img = ld.process_image(image)
    cv2.imwrite("./images/detection/frame.jpg",proc_img)



