import cv2
import numpy as np
import math
from datetime import datetime
from matplotlib import pyplot as plt
from collections import deque
from scipy.stats  import mode
from scipy.optimize import curve_fit
from yolo_model import BoundBox

temp_dir = "images/detection/detect.jpg"

WHITE = (255, 255, 255)
YELLOW = (66, 244, 238)
GREEN = (80, 220, 60)
LIGHT_CYAN = (255, 255, 224)
DARK_BLUE = (139, 0, 0)
GRAY = (128, 128, 128)
RED = (0,0,255)
ORANGE =(0,165,255)
BLACK =(0,0,0)
vehicles = [1,2,3,5,6,7,8]
animals =[15,16,17,18,19,21,22,23,]
humans =[0]
obstructions =  humans + animals + vehicles
classes = [#
    'Ped','bicycle','car','motorbike','aeroplane','bus',\
    'train','truck','boat','traffic light','fire hydrant','stop sign',\
    'parking meter','bench','bird','cat','dog','horse',\
    'sheep','cow','elephant', 'bear','zebra','giraffe',\
    'backpack','umbrella','handbag','tie','suitcase','frisbee',\
    'skis','snowboard','sports ball','kite','baseball bat',\
    'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass',\
    'cup','fork','knife','spoon','bowl','banana',\
    'apple','sandwich','orange','broccoli','carrot','hot dog',\
    'pizza','donut','cake','chair','sofa','pottedplant',\
    'bed','diningtable','toilet','tvmonitor','laptop','mouse',\
    'remote','keyboard','cell phone','microwave','oven','toaster',\
    'sink','refrigerator','book','clock','vase','scissors',\
    'teddy bear','hair drier','toothbrush' ]





def create_queue(length = 10):
    return deque(maxlen=length)

def polyfunc(x, a2, a1, a0):
     return a2*x*x + a1*x + a0

class OBSTACLE(BoundBox):
        xmax :int
        xmin :int
        ymin :int
        ymax :int
        xmid  :int
        ymid  :int
        lane : str
        x :  int
        y : int
        tracker = None
        position  : [int,int]
        PERIOD = 4
        __count = 0
    
        def __init__(self,box: BoundBox, _id, v_updt =5) :
            self.col_time:float =999.0
            self._id = _id
            
            self.position_hist = create_queue(v_updt)
            # self.position_hist.append(dst)
            self.update_coord(box)
            self.update_score(box)
            self.velocity = np.zeros((2))
            self.score=box.score
            self.label = box.label
    
        def update_obstacle(self, dst,  fps, n=5) :
            self.position = dst
            if self.lane == "my" :
                self.col_time = min(int(dst[1]/(self.velocity[1]+0.001)*18/5),99)
            else :
                self.col_time   = None
            if (self.__count > self.position_hist.maxlen):
                self.velocity = ((self.position-self.position_hist[0] ) * fps / self.position_hist.maxlen *5/18 ).astype(int)
            self.__count += 1
            self.position_hist.append(dst)

        def update_coord(self,box):
            self.xmax = box.xmax
            self.xmin =  box.xmin
            self.ymin  =  box.ymin
            self.ymax =  box.ymax
            self.xmid = int((box.xmax+box.xmin)/2)
            self.ymid = int((box.ymax+box.ymin)/2)
            self.position =  np.mean(self.position_hist, axis = 0)
    
        def update_score(self,box):     
            self.score=box.score
            self.label = box.label 
    
        def update_box(self,box):
            self.update_coord(box)
            self.update_score(box)


class LANE_LINE:
    def __init__(self):
        
        self.polynomial_coeff = []
        self.line_fit_x = []
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

    def fitpoly(self, P0, ):
        self.polynomial_coeff,_ = curve_fit(polyfunc,self.non_zero_y, self.non_zero_x, p0=P0)#, sigma=sigma)
        return




class LANE_HISTORY:
    def __init__(self, queue_depth=12,poly_col=np.array([1,1,1]), test_points=[300, 500, 700], poly_max_deviation_distance=50, smoothing = 10, ploty =np.array([])):
        self.lane_lines = create_queue(queue_depth)
        self.smoothed_poly = poly_col
        self.test_points = np.asarray(test_points)
        self.poly_max_deviation_distance = poly_max_deviation_distance
        self.lost = False
        self.lost_count = 0
        self.max_lost_count = queue_depth
        self.smoothing =  smoothing
        self.ploty =ploty
        self.fitx =  None
        self.appended = 0 
        self.breached = 0
        self.reset = 0
      
    
    def addlane(self, lane_line :LANE_LINE,):
        status = "APPENDED | "   
        lane_line.fitpoly(P0 = self.smoothed_poly ) 
        # dist = abs( np.polyval(self.smoothed_poly,-self.ploty[-1]) - np.polyval(lane_line.polynomial_coeff, -self.ploty[-1]) )
        # skip =  dist  >  self.poly_max_deviation_distance 
        # if skip and (self.fitx is not None) :
        #     status = "SKP_BRCH | "
        #     lane_line.polynomial_coeff = self.smoothed_poly
        #     lane_line.line_fit_x =  self.fitx
        #     self.lost =  True
        #     self.lost_count += 1 
        #     self.breached +=1
        #     return False , status, lane_line 

        if  (len(self.lane_lines) == 0 or (self.lost_count > self.max_lost_count )) :  
            status ="RESET | "
            self.lane_lines.append(lane_line)
            lane_line.polynomial_coeff = self.get_smoothed_polynomial()
            lane_line.line_fit_x =  np.polyval(lane_line.polynomial_coeff  ,  -self.ploty)  
            self.fitx = lane_line.line_fit_x
            self.lost =  False
            self.lost_count = 0
            self.reset +=1

            return True, status, lane_line
        test_y_smooth = np.asarray(list(map(lambda x: np.polyval(self.smoothed_poly,x), -self.test_points)))
        test_y_new = np.asarray(list(map(lambda x: np.polyval(lane_line.polynomial_coeff,x), -self.test_points)))
        dist = np.absolute(test_y_smooth - test_y_new)
        max_dist = dist[np.argmax(dist)]
        if max_dist > self.poly_max_deviation_distance:
            status = "BREACHED | "
            lane_line.polynomial_coeff = self.smoothed_poly
            lane_line.line_fit_x =  self.fitx
            self.lost =  True
            self.lost_count += 1 
 
            self.breached +=1
            return False , status, lane_line 

        self.lane_lines.append(lane_line)
        lane_line.polynomial_coeff = self.get_smoothed_polynomial()
        lane_line.line_fit_x =  np.polyval(lane_line.polynomial_coeff  ,  -self.ploty)  
        self.fitx = lane_line.line_fit_x
        self.lost =  False
        self.lost_count = 0
        self.appended +=1
        return True, status, lane_line
    
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
    windows_per_line = 40
    vanishing_point:(int,int)
    real_world_lane_size_meters=(32, 3.7)
    ego_vehicle_in_frame=False
    font = cv2.FONT_HERSHEY_SIMPLEX
    def __init__(self,  img,fps,
            yellow_lower = np.uint8([ 20, 50,   40]),
            yellow_upper = np.uint8([35, 255, 255]),
            white_lower = np.uint8([ 0, 200,   0]),
            white_upper = np.uint8([180, 255, 100]), 
            lane_start=[0.35,0.75] , 
            verbose = 3):
        self.message = ""
        self.verbose  =  verbose
        self.objpts = None
        self.count =  0
        self.fps=int(fps)
        self.imgpts = None
        self.lane_roi = None
        ## LANE DETECTION PROPERTIES
        self.yellow_lower =  yellow_lower
        self.yellow_upper =  yellow_upper
        self.white_lower =  white_lower
        self.white_upper = white_upper 
        self.lane_change = True
        # IMAGE PROPERTIES
        self.image =  img
        self.font_sz = 4e-4 * self.image.shape[0]
        self.img_dimensions =  (self.image.shape[0], self.image.shape[1]) 
        self.UNWARPED_SIZE  = (int(self.img_dimensions[1]*1),int(self.img_dimensions[1]*1))
        self.WRAPPED_WIDTH =  int(self.img_dimensions[1]*0.15)
        self.window_half_width = int(self.img_dimensions[1]*0.075)

        x =  np.linspace(0,self.img_dimensions[1]-1, self.img_dimensions[1])
        self.parabola = -200*x*(x-self.img_dimensions[1])
        self._pip_size = (int(self.image.shape[1] * 0.2), int(self.image.shape[0]*0.2))
        self.sliding_window_recenter_thres=9
        # x =  np.linspace(0, self.window_half_width*2-1, self.window_half_width*2)
        # self.lanebola = -50*x*(x-self.window_half_width*2)
        self.leftx = create_queue(self.fps//4)
        self.rightx = create_queue(self.fps//4)
        self.calc_perspective(lane_start=lane_start)
        self.ndirect = 0
        self.nskipped = 0
        self.max_gap = 0
        self.coef = np.array([1,1,1])
        if self.ego_vehicle_in_frame :
            self.windows_range = range(int(self.windows_per_line*0.05), self.windows_per_line, 1)
            self.window_offset = int(self.windows_per_line*0.05)
        else:
            self.windows_range = range( self.windows_per_line)
            self.window_offset = 0 
        self.ploty = np.linspace(int(self.UNWARPED_SIZE[1]*0.45), self.UNWARPED_SIZE[1]- 1, int(self.UNWARPED_SIZE[1]*0.4))
        self.previous_left_lane_line = LANE_LINE()
        self.previous_right_lane_line = LANE_LINE()
        test = np.arange(0.3,1,0.1)*self.UNWARPED_SIZE[1]
        test = test.astype(int)
        self.left_line_history = LANE_HISTORY(test_points = test, queue_depth=self.fps//3, ploty = self.ploty)
        self.right_line_history = LANE_HISTORY(test_points = test, queue_depth=self.fps//3, ploty = self.ploty)
        self.total_img_count = 0
        self.margin_red = 1# .995
        
        

    def compute_bounds(self, image):
        avg = np.average(image[image.shape[0]//4:image.shape[0]*3//4,image.shape[1]*2//3 : image.shape[1],1])
        l_rel =  max(min((avg/100)**3.0,1.35),0.5)
        self.yellow_lower[1] = int(l_rel*50)
        self.white_lower[1] = int(l_rel *180)
        self.message = self.message+"WHITE" + str(self.white_lower[1])
        return

    def compute_mask(self, image):
        """
        Returns a binary thresholded image produced retaining only white and yellow elements on the picture
        The provided image should be in RGB formats
        """
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        if self.count % (self.fps) == 0 :
            self.compute_bounds(converted)
            
        yellow_mask = cv2.inRange(converted, self.yellow_lower, self.yellow_upper)   
        white_mask = cv2.inRange(converted, self.white_lower, self.white_upper)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return mask

    def calc_perspective(self,  lane_start=[0.25,0.75]):
        roi = np.zeros((self.img_dimensions[0], self.img_dimensions[1]), dtype=np.uint8) # 720 , 1280
        roi_points = np.array([[0, self.img_dimensions[0]*8//9],
                    [0, self.img_dimensions[0]],
                    [self.img_dimensions[1], self.img_dimensions[0]],
                    [self.img_dimensions[1], self.img_dimensions[0]*8//9],
                    [self.img_dimensions[1]*15//23,self.img_dimensions[0]*13//23],
                    [self.img_dimensions[1]*9//23,self.img_dimensions[0]*13//23]], dtype=np.int32)
        cv2.fillPoly(roi, [roi_points], 1)
        # x = np.linspace(0,self.img_dimensions[0]-1,self.img_dimensions[0])
        # grad= np.tile(-5*x*(x-self.img_dimensions[1]),self.img_dimensions[1]).reshape((self.img_dimensions[0], self.img_dimensions[1]))

        self.lane_roi = np.zeros((self.img_dimensions[0], self.img_dimensions[1]), dtype=np.uint8)
        Lhs = np.zeros((2,2), dtype= np.float32)
        Rhs = np.zeros((2,1), dtype= np.float32)
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mn_hsl = np.median(grey) #grey.median()
        edges = cv2.Canny(grey, int(mn_hsl*2), int(mn_hsl*.4))
        # edges = cv2.Canny(grey[:, :, 1], 500, 400)
        # plt.imshow(edges, cmap="gray")
        # plt.show()
        # cv2.imwrite(self.temp_dir+"mask.jpg", grey*roi)
        # cv2.imwrite(self.temp_dir+"mask.jpg", edges*roi)

        lines = cv2.HoughLinesP(edges*roi,rho =20,theta = 2* np.pi/180,threshold = 7,minLineLength = self.img_dimensions[0]//3,maxLineGap = self.img_dimensions[0]//15)

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
            print("ABSURD POSITION TRY OTHER PARAMETERS",self.vanishing_point , "FOR",self.img_dimensions )
            self.vanishing_point[0] = self.img_dimensions[1]//2
        if abs(self.vanishing_point[1] - self.img_dimensions[0]*0.55) > 0.1 *self.img_dimensions[0] :
            print("ABSURD POSITION TRY OTHER PARAMETERS",self.vanishing_point , "FOR",self.img_dimensions )
            self.vanishing_point[1] = int(self.img_dimensions[0]*0.55)
        top =self.vanishing_point[1] + int(self.WRAPPED_WIDTH*0.15)
        bottom = self.img_dimensions[0]*7//6
        

        def on_line(p1, p2, ycoord):
            return [p1[0]+ (p2[0]-p1[0])/float(p2[1]-p1[1])*(ycoord-p1[1]), ycoord]

        p1 = [self.vanishing_point[0] - self.WRAPPED_WIDTH/2, top]
        p2 = [self.vanishing_point[0] + self.WRAPPED_WIDTH/2, top]
        p3 = on_line(p2,self.vanishing_point, bottom)
        p4 = on_line(p1,self.vanishing_point, bottom)
        src_points = np.array([p1,p2,p3,p4], dtype=np.float32)
        dst_points = np.array([[0, 0], [self.UNWARPED_SIZE[0], 0],
                            [self.UNWARPED_SIZE[0], self.UNWARPED_SIZE[1]],
                            [0, self.UNWARPED_SIZE[1]]], dtype=np.float32)
        self.trans_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inv_trans_mat = cv2.getPerspectiveTransform(dst_points,src_points)
        min_wid = 1000
        img = cv2.warpPerspective(self.image, self.trans_mat, self.UNWARPED_SIZE)
        mask = self.compute_mask(img) 
        
        # x = np.linspace(0,mask.shape[0]-1,mask.shape[0])
        x1 = int(self.UNWARPED_SIZE[0]*lane_start[0])
        x2 = int(self.UNWARPED_SIZE[0]*lane_start[1]) 
        span = self.UNWARPED_SIZE[0]//5 
        x1 = x1-span + self.detect_lane_start(mask[:,x1-span :x1+span])
        x2 = x2-span + self.detect_lane_start(mask[:,x2-span :x2+span])
        self.leftx.append(x1)
        self.rightx.append(x2)
        self.previous_centers = np.ones(self.windows_per_line)*(x1+x2)//2
        lane_roi_points = np.array([
                    [self.img_dimensions[1]*9//80, self.img_dimensions[0]],
                    [self.img_dimensions[1]*71//80,self.img_dimensions[0]],
                    [self.vanishing_point[0] + 3*self.img_dimensions[1]//25,self.vanishing_point[1] - 10],
                    [self.vanishing_point[0] - 3*self.img_dimensions[1]//25,self.vanishing_point[1] - 10]], dtype=np.int32)
        cv2.fillPoly(self.lane_roi , [lane_roi_points], 1)
        # self.lane_roi =  self.lane_roi*grad

       

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
        if self.verbose  > 2:       
            img_orig = cv2.polylines(self.image, [src_points.astype(np.int32)],True, (0,0,255), thickness=5)
            cv2.line(img, (int(x1), 0), (int(x1), self.UNWARPED_SIZE[1]), (255, 0, 0), 3)
            cv2.line(img, (int(x2), 0), (int(x2), self.UNWARPED_SIZE[1]), (0, 0, 255), 3)

            cv2.circle(img_orig,tuple(self.vanishing_point),10, color=(0,0,255), thickness=5)
            cv2.imwrite(self.temp_dir+"perspective1.jpg",img_orig)
            cv2.imwrite(self.temp_dir+"perspective2.jpg",img)
            cv2.imwrite(self.temp_dir+"perspective3.jpg",img2)
            return img_orig
        return

    def calculate_position(self, box: OBSTACLE):
            pos = np.array((box.xmax/2+box.xmin/2, box.ymax)).reshape(1, 1, -1)
            dst =  cv2.perspectiveTransform(pos, self.trans_mat).reshape(2)
            box.x =  int(dst[0])
            box.y = -int(dst[1])
            left= np.polyval(self.left_line_history.smoothed_poly ,box.y) - box.x
            right= np.polyval(self.right_line_history.smoothed_poly ,box.y) - box.x
            status = "my"
            if left < 0 and right <0:
                status = "left"
            elif right>0 and left >0 :
                status = "right"
            box.lane = status
            dst =  np.array([dst[0]/self.px_per_xm,(self.UNWARPED_SIZE[1]-dst[1])/self.px_per_ym])
            box.update_obstacle(dst,self.fps)

            
            return box
        # else:
            
        #     return np.array([0,0])    
    def put_text(self, overlay,text, coord, color=WHITE):
        sz = self.font_sz*50
        rect_ht = int(sz *1.1)
        rect_wd = int(len(text)*sz*0.5)
        p1 = (coord[0], coord[1]+2)
        p2 = (coord[0]+rect_wd, coord[1]-rect_ht)
        cv2.rectangle(overlay, p1, p2,  (0, 0, 0),-1)
        cv2.putText(overlay, text,   coord,  self.font, self.font_sz*1.25, color, 1, cv2.LINE_AA)

        return 
    
    def process_image(self, img,obstacles :[OBSTACLE] =[], alpha=1, beta=0.3, gamma=0):
        """
        Attempts to find lane lines on the given image and returns an image with lane area colored in green
        as well as small intermediate images overlaid on top to understand how the algorithm is performing
        """
        # First step - undistort the image using the instance's object and image points
        # undist_img = undistort_image(img, self.objpts, self.imgpts)
        off =  int(100*self.font_sz)
        self.message = "["+ str(self.count)+"]"
        overlay =  img.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ll, rl , thres_img_psp= self.compute_lane_lines(overlay)
        # overlay = thres_img_psp.copy() 
        for i in range(len(obstacles)):
            obstacles[i] =  self.calculate_position(obstacles[i])
            box =  obstacles[i]
            past=[box.xmin,box.ymin,box.xmax,box.ymax]
            color = WHITE
            t1 = classes[obstructions[box.label]] +" ["+str(int(box.position[1])) + "m]" 
            t2 = "("+str(int(box.score*100))+"%) ID: " +str(box._id)
            b1= "Lane " + box.lane  + " " + str(int(box.velocity[1]))+"kmph"

            
            pt1 = (box.xmin, box.ymin-off)
            pt2 =  (box.xmin, box.ymin)
            pb1 = (box.xmin, box.ymax+off)
            if (box.lane == "my") and (box.col_time < 99) : 
                color = RED
                b3 = "Col "+str(int(box.col_time))+"s"
                pb3 =  (box.xmin, box.ymax+2*off)
                self.put_text(overlay, b3,   pb3, color = color)
            
            self.put_text(overlay, t1,   pt1, color = color)
            self.put_text(overlay, t2,   pt2, color = color)
            self.put_text(overlay, b1,   pb1, color = color)
            
            past_center =  (int(past[0]/2+past[2]/2), past[3])
            
            color = ORANGE if box.velocity[1] < 0  else GREEN
            cv2.rectangle(overlay, (box.xmin,box.ymin), (box.xmax,box.ymax), color,2)
            cv2.circle(overlay,past_center,1, GRAY,2)
        img =  cv2.addWeighted(img, alpha, overlay, beta, gamma)
        




        
        ll.purge_points(self.UNWARPED_SIZE[1])
        rl.purge_points(self.UNWARPED_SIZE[1])
        out_img = np.dstack((thres_img_psp,thres_img_psp,thres_img_psp))
        img = self.draw_lane_area(out_img, img, ll, rl)  
        self.previous_left_lane_line = ll
        self.previous_right_lane_line = rl
        lcr, rcr, lco = self.compute_lane_curvature(ll, rl)
        if self.verbose >2 : 
            
            drawn_lines = self.draw_lane_lines(out_img, ll, rl)        
            # drawn_lines_regions = self.draw_lane_lines_regions(out_img, ll, rl)
            drawn_hotspots = self.draw_lines_hotspots(out_img, ll, rl, obstacles)
            img = self.combine_images(img, drawn_lines,drawn_hotspots)
            img = self.draw_lane_curvature_text(img, lcr, rcr, lco)
        self.total_img_count += 1
        
        return img
    
    def draw_lane_curvature_text(self, img, left_curvature_meters, right_curvature_meters, center_offset_meters):
        """
        Returns an image with curvature information inscribed
        """
        sz = self.font_sz*3
        offset_y = self._pip_size[1] * 1 + self._pip__y_offset * 5
        offset_x = self._pip__x_offset
        
        template = "{0:17}{1:17}{2:17}"
        txt_header = template.format("Left ", "Right ", "Center") 
        # print(txt_header)
        txt_values = template.format("{:.0f}m".format(left_curvature_meters), 
                                     "{:.0f}m".format(right_curvature_meters),
                                     "{:.2f}m Right".format(center_offset_meters))
        if center_offset_meters < 0.0:
            txt_values = template.format("{:0f}m".format(left_curvature_meters), 
                                     "{:0f}m".format(right_curvature_meters),
                                     "{:.1f}m Left".format(math.fabs(center_offset_meters)))
            
        

        cv2.putText(img, txt_header, (offset_x, offset_y), self.font, sz, BLACK, 1, cv2.LINE_AA)
        cv2.putText(img, txt_values, (offset_x, offset_y + self._pip__y_offset * 2), self.font, sz, BLACK, 2, cv2.LINE_AA)
        cv2.putText(img, self.message, (offset_x, self.img_dimensions[0]-10), self.font, sz, BLACK, 1, cv2.LINE_AA)
        return img
    
    def combine_images(self, lane_area_img, lines_img,lane_hotspots_img):        
        """
        Returns a new image made up of the lane area image, and the remaining lane images are overlaid as
        small images in a row at the top of the the new image
        """

     
        small_lines = cv2.resize(lines_img, self._pip_size)
        # small_region = cv2.resize(lines_regions_img, self._pip_size)
        small_hotspots = cv2.resize(lane_hotspots_img, self._pip_size)
        # small_color_psp = cv2.resize(psp_color_img, self._pip_size)
                
        lane_area_img[self._pip__y_offset: self._pip__y_offset + self._pip_size[1], self._pip__x_offset: self._pip__x_offset + self._pip_size[0]] = small_lines
        
        # start_offset_y = self._pip__y_offset 
        # start_offset_x = 2 * self._pip__x_offset + self._pip_size[0]
        # lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0]] = small_region
        
        start_offset_y = self._pip__y_offset 
        start_offset_x = 2 * self._pip__x_offset + 1 * self._pip_size[0]
        lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0]] = small_hotspots

        # start_offset_y = self._pip__y_offset 
        # start_offset_x = 4 * self._pip__x_offset + 3 * self._pip_size[0]
        # lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0],:] = small_color_psp
        
        
        return lane_area_img
    
        
    def draw_lane_area(self, warped_img, undist_img, left_line, right_line):
        """
        Returns an image where the inside of the lane has been colored in bright green
        """
        color_warp = np.zeros_like(warped_img).astype(np.uint8)
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
        out_img = img.copy()
        # try: 
        pts_left = np.dstack((left_line.line_fit_x, self.ploty)).astype(np.int32)
        pts_right = np.dstack((right_line.line_fit_x, self.ploty)).astype(np.int32)

        cv2.polylines(out_img, pts_left, False,  (255, 140,0), 5)
        cv2.polylines(out_img, pts_right, False, (255, 140,0), 5)
        
        for low_pt, high_pt in left_line.windows:
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)

        for low_pt, high_pt in right_line.windows:            
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)  
        # except:
        #     print("LANE LOST IN THIS FRAME") 
        #     plt.imshow(out_img)
        #     plt.show()        
        return out_img    
    
    def draw_lane_lines_regions(self, img, left_line, right_line):
        """
        Returns an image where the computed left and right lane areas have been drawn on top of the original warped binary image
        """
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
        cv2.fillPoly(warped_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(warped_img, np.int_([right_line_pts]), (0, 255, 0))
        
        return warped_img


    def draw_lines_hotspots(self, img, left_line, right_line, obstacles:[OBSTACLE] =  []):
        """
        Returns a RGB image where the portions of the lane lines that were
        identified by our pipeline are colored in yellow (left) and blue (right)
        """
        # out_img = np.dstack((warped_img, warped_img, warped_img))*255
        sz = self.font_sz*12
        out_img = img.copy()
        for i in range(len(left_line.non_zero_x)) :
            cv2.circle( out_img,( left_line.non_zero_x[i],-left_line.non_zero_y[i]), 5, (255, 255, 0), -1)
        for i in range(len(right_line.non_zero_x)) :
            cv2.circle( out_img,( right_line.non_zero_x[i],-right_line.non_zero_y[i]), 5, (0, 0, 255), -1)
        cv2.line(out_img, (int(np.average(self.leftx)), 0), (int(np.average(self.leftx)), self.UNWARPED_SIZE[1]), (255, 255, 0), 4)
        cv2.line(out_img, (int(np.average(self.rightx)), 0), (int(np.average(self.rightx)), self.UNWARPED_SIZE[1]), (0, 0, 255), 4)
        for i in range(len(obstacles)):
            box =  obstacles[i]
            cv2.putText(out_img,str(box._id),(box.x, -box.y), self.font, sz, GREEN, 8, cv2.LINE_AA)    
        return out_img

    def compute_lane_curvature(self, left_line, right_line):
        """
        Returns the triple (left_curvature, right_curvature, lane_center_offset), which are all in meters
        """        
        y_eval = -np.max(self.ploty)
        left_fit = self.left_line_history.smoothed_poly# np.polyfit(-1*self.ploty * self.ym_per_px, leftx * self.xm_per_px, 2)
        right_fit =self.right_line_history.smoothed_poly# np.polyfit(-1*self.ploty * self.ym_per_px, rightx * self.xm_per_px, 2)
        left_curverad = int((1 + (2 * left_fit[0] * y_eval * self.ym_per_px + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
        right_curverad = int((1 + (2 *right_fit[0] * y_eval * self.ym_per_px + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
        # left_fit = left_line.polynomial_coeff
        # right_fit = right_line.polynomial_coeff
        half_width  =    int(np.mean(self.rightx) -np.mean(self.leftx))//2
        center_offset_img_space = (((left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]) + 
                   (right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2])) / 2) - self.vanishing_point[0]
        self.lane_change = False
        if abs(center_offset_img_space) > half_width :
            self.lane_change = True
            if center_offset_img_space < 0:
                print("\n\rLANE CHANGE TO RIGHT\033[F")
                self.left_line_history = self.right_line_history
                self.previous_left_lane_line  = self.previous_right_lane_line
                self.right_line_history = LANE_HISTORY(test_points = self.left_line_history.test_points, queue_depth=self.left_line_history.max_lost_count, ploty = self.ploty)
                self.previous_right_lane_line =  LANE_LINE()
                self.leftx =  self.rightx
                self.rightx  =  create_queue(length = self.leftx.maxlen)
                self.rightx.append(np.mean(self.leftx) +  2*half_width)
            elif np.mean(self.rightx)>2*half_width:
                print("\n\rLANE CHANGE TO LEFT\033[F")
                self.right_line_history = self.left_line_history
                self.previous_right_lane_line  = self.previous_left_lane_line
                self.left_line_history = LANE_HISTORY(test_points = self.right_line_history.test_points, queue_depth=self.right_line_history.max_lost_count, ploty = self.ploty)
                self.previous_left_lane_line =  LANE_LINE()
                self.rightx =  self.leftx
                self.leftx  =  create_queue(length = self.rightx.maxlen)
                self.leftx.append(np.mean(self.rightx) -  2*half_width)
            self.previous_centers = np.ones(self.windows_per_line)*(np.mean(self.leftx)+np.mean(self.rightx))//2
        center_offset_real_world_m = int(center_offset_img_space * self.xm_per_px*100)/100.0       

        return left_curverad, right_curverad, center_offset_real_world_m
        
    def detect_lane_start(self, image):
        histx =   np.sum(image[image.shape[0]*4//5:,:], axis=0)
        # histx = histx * self.lanebola
        # print(np.argmax(histx))
        return np.argmax(histx)
    
        
    def compute_lane_lines(self, img):
        """
        Returns the tuple (left_lane_line, right_lane_line) which represents respectively the LANE_LINE instances for
        the computed left and right lanes, for the supplied binary warped image
        """
        left_line = self.previous_left_lane_line
        right_line = self.previous_right_lane_line
        left_line.windows = []
        right_line.windows = []
        undst_img  = cv2.bitwise_and(img, img, mask =  self.lane_roi )
        pp_img = cv2.warpPerspective(undst_img, self.trans_mat, (self.UNWARPED_SIZE[1],self.UNWARPED_SIZE[0]))
        warped_img = self.compute_mask(pp_img)

        margin = self.window_half_width
        minpix = self.sliding_window_recenter_thres
        window_height = np.int(warped_img.shape[0]//self.windows_per_line)
        x1_av =  int(np.average(self.leftx))
        x2_av  = int(np.average(self.rightx))
        x1 = min(max(x1_av,self.window_half_width), self.UNWARPED_SIZE[0]-self.window_half_width*2)
        x2 = max( min(x2_av,self.UNWARPED_SIZE[0]-self.window_half_width-1), x1+self.window_half_width*2)
        leftx_base = x1-self.window_half_width + self.detect_lane_start(warped_img[:,x1-self.window_half_width :x1+self.window_half_width])
        rightx_base = x2-self.window_half_width + self.detect_lane_start(warped_img[:,x2-self.window_half_width :x2+self.window_half_width])
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])  
        if (rightx_base - leftx_base > self.UNWARPED_SIZE[0]//4) \
            and (rightx_base - leftx_base < self.UNWARPED_SIZE[0]//2)\
                and (not self.lane_change)\
                    and  (leftx_base > 0):
            if abs(x1_av - leftx_base) < margin : 
                self.leftx.append(leftx_base)
            if abs(x2_av - rightx_base) < margin : 
                self.rightx.append(rightx_base)
        leftx_current = int(np.mean(self.leftx))
        rightx_current = int(np.mean(self.rightx))
        lane_width= int(rightx_current - leftx_current)                
        centerx_current = leftx_current + lane_width//2
        centers = []
        center_idx = []
        all_centers = []

        leftx =[]
        rightx =[]
        lefty =[]
        righty =[]

        # leftx =[leftx_current]
        # rightx =[rightx_current]
        # lefty =[-(warped_img.shape[0] - self.windows_range[0] * window_height)]
        # righty =[-(warped_img.shape[0] - self.windows_range[0]* window_height)]
        curve_compute =  0
        self.max_gap =  0 
        gap  = 0
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

            s = 0
            if len(good_left_inds) > minpix:
                x = mode(nonzerox[good_left_inds])[0]
                y = mode(nonzeroy[good_left_inds])[0]
                leftx.extend(x)
                lefty.extend(-y)
                s +=1
                centerx = x + lane_width//2
                rightx.extend(x + lane_width)
                righty.extend(-y)
                # centery = nonzeroy[good_left_inds]
            if len(good_right_inds) > minpix:
                s += 1
                x = mode(nonzerox[good_right_inds])[0]
                y = mode(nonzeroy[good_right_inds])[0]

                rightx.extend(x)
                righty.extend(-y)
                centerx = np.append(centerx, x-lane_width//2)
                leftx.extend(x - lane_width)
                lefty.extend(-y)


            # self.max_gap += s > 0
            
            if s > 0:    
                centerx_current = int(np.average(centerx))
                centers.append(centerx_current)
                center_idx.append(window)
                
                gap = 0
            else :
                gap +=1
                if len(center_idx) > 5 :
                    if curve_compute% 5 == 0 :
                        self.coef,_ =curve_fit(polyfunc,np.array(center_idx),np.array(centers), p0=self.coef)
                    centerx_current = int(np.polyval(self.coef, window+1))
                    curve_compute += 1
                else :
                    centerx_current = self.previous_centers[window-self.window_offset]
            self.max_gap =  max(self.max_gap, gap)
            all_centers.append(centerx_current)
            leftx_current = int(centerx_current -lane_width/2)
            rightx_current = int(centerx_current +lane_width/2)
            margin =  int(margin * self.margin_red)
        if (self.max_gap > self.windows_per_line//2) and (not self.lane_change):
            self.previous_left_lane_line.windows = []
            self.previous_right_lane_line.windows = []
            self.message+="SKIPPED  "+ str(self.max_gap)
            left_line =  self.previous_left_lane_line
            right_line =  self.previous_right_lane_line
            self.count+=1
            self.nskipped+=1
            self.compute_bounds(cv2.cvtColor(pp_img, cv2.COLOR_BGR2HLS))
            return (left_line, right_line,warped_img)
            
        self.previous_centers = all_centers
        left_line.non_zero_x = np.array(leftx,np.int)  
        left_line.non_zero_y =  np.array(lefty,np.int)
        right_line.non_zero_x = np.array(rightx,np.int)
        right_line.non_zero_y =  np.array(righty,np.int)

        if len(leftx)>5:
            left_ok,status,left_line = self.left_line_history.addlane(left_line)
            self.message+= "LEFT" +   status  
            if not left_ok :
                self.compute_bounds(cv2.cvtColor(pp_img, cv2.COLOR_BGR2HLS)) 
        else:
            left_line.polynomial_coeff  = self.left_line_history.smoothed_poly 
            left_line.line_fit_x =  np.polyval(left_line.polynomial_coeff  ,  -self.ploty) 
            # self.compute_bounds(cv2.cvtColor(pp_img, cv2.COLOR_BGR2HLS)) 
                 
        if len(rightx)>5:
            right_ok,status,right_line = self.right_line_history.addlane(right_line)
            self.message+= "RIGHT" + status
            if not right_ok :
                self.compute_bounds(cv2.cvtColor(pp_img, cv2.COLOR_BGR2HLS)) 
        else:
            right_line.polynomial_coeff = self.right_line_history.smoothed_poly
            right_line.line_fit_x =  np.polyval(right_line.polynomial_coeff  ,  -self.ploty)  
            # self.compute_bounds(cv2.cvtColor(pp_img, cv2.COLOR_BGR2HLS)) 
        self.count+=1
        return (left_line, right_line,warped_img)

if __name__ == "__main__":


    # video_reader =  cv2.VideoCapture("videos/nice_road.mp4") 
    video_reader =  cv2.VideoCapture("videos/challenge_video.mp4")
    # video_reader =  cv2.VideoCapture("videos/harder_challenge_video.mp4")
    fps =  video_reader.get(cv2.CAP_PROP_FPS)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_out = "videos/output30.mov"
    # cv2.VideoWriter_fourcc(*'MPEG')
    video_writer = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_w, frame_h))
    pers_frame_time = 14#180# seconds
    pers_frame = int(pers_frame_time *fps)
    video_reader.set(1,pers_frame)
    ret, image = video_reader.read()
    ld = LANE_DETECTION( image,fps)
    # pers_frame_time = 130# seconds
    pers_frame = int(pers_frame_time *fps)
    video_reader.set(1,pers_frame)
    ret, image = video_reader.read()
    proc_img = ld.process_image(image)
    cv2.imwrite("./images/detection/frame.jpg",proc_img)



