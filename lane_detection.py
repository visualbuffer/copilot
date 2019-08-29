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
BLUE = (255,0,0)
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



class LANE_HISTORY:
    def __init__(self,fps, queue_depth=12,poly_col=np.array([1,1,1]), test_points=[300, 500, 700], poly_max_deviation_distance=50, smoothing = 10, ploty =np.array([])):
        self.fps =fps
        self.test_points = np.asarray(test_points)
        self.poly_max_deviation_distance = poly_max_deviation_distance
        self.lost = False
        
        self.max_lost_count = queue_depth
        self.lost_count = self.max_lost_count + 10
        self.smoothing =  smoothing
        self.ploty =ploty
        self.leftFit =  None      # LEFT FIT POINTS
        self.rightFit =  None     # RIGHT FIT POINTS
        self.leftx = create_queue(self.fps//4)
        self.rightx =  create_queue(self.fps//4)
        self.width = None  
        self.previous_centers =  None
        self.current_coef = None
        self.smoothed_poly = poly_col
        self.poly_history = create_queue(queue_depth)
        self.y = None
        self.x = None
        self.appended = 0 
        self.breached = 0
        self.reset = 0
        self.curvature = 0 
        self.centerx = 0
        self.lane_offset = 0 
        self.left_windows = []
        self.right_windows=[]


    def compute_lane_points(self) :  
        self.leftFit =  self.previous_centers - self.width//2
        self.rightFit = self.previous_centers + self.width//2

    def compute_curvature(self, alpha, beta):
        y_eval = -np.max(self.ploty)
        lp =  self.smoothed_poly
        self.curvature =  int(((beta**2 + (2 * lp[0] * y_eval * alpha**2 + \
                    lp[1]*alpha)**2)**1.5)/(np.absolute(2 * lp[0]*(alpha*beta)**2)))
        return


    def compute_offset(self):
        y_eval = -np.max(self.ploty)
        lp =  self.smoothed_poly
        self.lane_offset = lp[0] * y_eval**2 + lp[1] * y_eval + lp[2] -  self.centerx
        if abs(self.lane_offset) > self.width //2 :
            if self.lane_offset < 0 :
                print("\n\rLANE CHANGE TO RIGHT\033[F")
                self.poly_history = create_queue(self.poly_history.maxlen)
                self.leftx =  self.rightx
                self.rightx  =  create_queue(length = self.rightx.maxlen)
                self.rightx.append(int(np.mean(self.leftx) +  self.width  ))
                self.previous_centers =  self.previous_centers+self.width
            else :
                print("\n\rLANE CHANGE TO LEFT\033[F")
                self.poly_history = create_queue(self.poly_history.maxlen)
                self.rightx =  self.leftx
                self.leftx  =  create_queue(length = self.leftx.maxlen)
                self.leftx.append(int(np.mean(self.rightx) -  self.width  ))
                self.previous_centers =  self.previous_centers-self.width
        else:
            self.leftx.append(self.previous_centers[0] - self.width//2)
            self.rightx.append(self.previous_centers[0] + self.width//2)
        return

    def addlane(self, y,x):
        status = "APPENDED | "   
        self.y =  y
        self.x =  x
        self.current_coef,_ = curve_fit(polyfunc,self.y, self.x, p0=self.smoothed_poly)
        if  (self.lost_count > self.max_lost_count ) :  
            status ="RESET | "
            
            self.get_smoothed_polynomial()
            self.lost =  False
            self.lost_count = 0
            self.reset +=1
            return True, status

        test_y_smooth = np.asarray(list(map(lambda x: np.polyval(self.smoothed_poly,x), -self.test_points)))
        test_y_new = np.asarray(list(map(lambda x: np.polyval(self.current_coef,x), -self.test_points)))
        dist = np.absolute(test_y_smooth - test_y_new)
        max_dist = dist[np.argmax(dist)]
        if max_dist > self.poly_max_deviation_distance:
            status = "BREACHED | "
            self.lost =  True
            self.lost_count += 1 
 
            self.breached +=1
            return False , status 

        self.get_smoothed_polynomial()
        self.lost =  False
        self.lost_count = 0
        self.appended +=1
        return True, status
    
    def get_smoothed_polynomial(self):
        self.poly_history.append(self.current_coef)
        all_coeffs = np.asarray(list(self.poly_history))
        self.smoothed_poly = np.mean(all_coeffs[-self.smoothing:,:], axis=0)
        self.previous_centers =  np.asarray([np.polyval(self.smoothed_poly,-x) for x in self.ploty], dtype=int)
        self.compute_lane_points()
        self.compute_offset()
        return self.smoothed_poly

    def calculate_position(self,x,y):
        position =  np.polyval(self.smoothed_poly ,y) - x 
        status =  "right"
        if position < - self.width//2 :
            status =  "left"
        elif position < self.width//2:
            status =  "my"
        return status




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
    windows_per_line = 30
    vanishing_point:(int,int)
    real_world_lane_size_meters=(32, 3.7)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom = 0
    def __init__(self,  img,fps,
            yellow_lower = np.uint8([ 20, 50,   50]),
            yellow_upper = np.uint8([35, 255, 255]),
            white_lower = np.uint8([ 0, 200,   0]),
            white_upper = np.uint8([180, 255, 100]), 
            lum_factor = 130,
            max_gap_th = 2/5,
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
        self.lane_change = False
        # self.lane_width = 0
        # IMAGE PROPERTIES
        self.lum_factor =  lum_factor
       
        self.image =  img
        self.font_sz = 4e-4 * self.image.shape[0]
        self.img_dimensions =  (self.image.shape[0], self.image.shape[1]) 
        self.UNWARPED_SIZE  = (360,360)#(int(self.img_dimensions[1]*0.5),int(self.img_dimensions[1]*0.5))
        self.WRAPPED_WIDTH =  int(self.img_dimensions[1]*0.15)
        self.margin = int(self.UNWARPED_SIZE[1]*0.08)
        self.window_height = np.int(self.UNWARPED_SIZE[1]//self.windows_per_line)
        self._pip_size = (int(self.image.shape[1] * 0.2), int(self.image.shape[0]*0.2))
        self.minpix=self.window_height
        self.maxpix = int(self.margin * self.window_height *0.5)


        self.n_gap_skip = 0
        self.max_gap = 0

        self.windows_range = range( self.windows_per_line)
        self.window_offset = 0 
        self.ploty = np.linspace(int(self.UNWARPED_SIZE[1]*0.45), self.UNWARPED_SIZE[1]- 1, int(self.UNWARPED_SIZE[1]*0.4), dtype=int)

        test = np.arange(0.3,1,0.1)*self.UNWARPED_SIZE[1]
        test = test.astype(int)
        self.lane = LANE_HISTORY(self.fps,test_points = test, queue_depth=self.fps//3, ploty = self.ploty)
        self.max_gap_th =  max_gap_th * self.windows_per_line
        self.calc_perspective(lane_start=lane_start)
        

    def compute_bounds(self, image):
        lx = int( max( np.mean(self.lane.leftx),0))
        rx =  int(max(min(np.mean(self.lane.rightx), image.shape[0]),max(image.shape[0]//4,lx)))

        avg = np.average(image[lx:rx,\
            image.shape[1]//2 : image.shape[1]-self.bottom,1])
        l_rel =  max(min((avg/self.lum_factor)**2,1.3),0.45)
        self.yellow_lower[1] = int(l_rel*30)
        self.white_lower[1] = int(l_rel *170)
        self.message = self.message+"WHITE" + str(self.white_lower[1])
        return

    def compute_mask(self, image):
        """
        Returns a binary thresholded image produced retaining only white and yellow elements on the picture
        The provided image should be in RGB formats
        """
        # cv2.imwrite(self.temp_dir+"temp.jpg", image)
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        if self.count % (self.fps*4) == 0 :
            self.compute_bounds(converted)
            
        yellow_mask = cv2.inRange(converted, self.yellow_lower, self.yellow_upper)   
        white_mask = cv2.inRange(converted, self.white_lower, self.white_upper)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        # t2 =  cv2.bitwise_and(image, image, mask = mask)
        # cv2.imwrite(self.temp_dir+"temp2.jpg", t2)
        # print(self.white_lower[1])
        return mask

    def calc_perspective(self,  lane_start=[0.25,0.75]):
        roi = np.zeros((self.img_dimensions[0], self.img_dimensions[1]), dtype=np.uint8) # 720 , 1280
        roi_points = np.array([[0, self.img_dimensions[0]*7//9],
                [0, self.img_dimensions[0]],
                [self.img_dimensions[1], self.img_dimensions[0]],
                [self.img_dimensions[1], self.img_dimensions[0]*7//9],
                [self.img_dimensions[1]*45//99,self.img_dimensions[0]//2],
                [self.img_dimensions[1]*45//99,self.img_dimensions[0]//2]], dtype=np.int32)
        cv2.fillPoly(roi, [roi_points], 1)
        self.lane_roi = np.zeros((self.img_dimensions[0], self.img_dimensions[1]), dtype=np.uint8)
        Lhs = np.zeros((2,2), dtype= np.float32)
        Rhs = np.zeros((2,1), dtype= np.float32)
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mn_hsl = np.median(grey)
        edges = cv2.Canny(grey, int(mn_hsl*1.5), int(mn_hsl*.3))
        lines = cv2.HoughLinesP(edges*roi,rho =self.img_dimensions[0]//20,\
                theta = 2* np.pi/180,\
                threshold = self.img_dimensions[0]//80,\
                minLineLength = self.img_dimensions[0]//3,\
                maxLineGap = self.img_dimensions[0]//15)

        img2 =  self.image.copy()
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img2,(x1,y1),(x2,y2),(255,0,0),2)
                normal = np.array([[-(y2-y1)], [x2-x1]], dtype=np.float32)
                normal /=np.linalg.norm(normal)
                point = np.array([[x1],[y1]], dtype=np.float32)
                outer = np.matmul(normal, normal.T)
                Lhs += outer
                Rhs += np.matmul(outer, point)
        self.vanishing_point = np.matmul(np.linalg.inv(Lhs),Rhs).reshape(2)
        orig_points=self.vanishing_point.copy() 
        if abs(self.vanishing_point[0] - self.img_dimensions[1]//2) > 0.07 *self.img_dimensions[1] :
            print("ABSURD X POSITION TRY OTHER PARAMETERS",self.vanishing_point[0] , "in ==> ",self.img_dimensions )
            self.vanishing_point[0] = self.img_dimensions[1]//2
        if abs(self.vanishing_point[1] - self.img_dimensions[0]*0.57) > 0.1 *self.img_dimensions[0] :
            print("ABSURD Y POSITION TRY OTHER PARAMETERS",self.vanishing_point[0] , "in ==>",self.img_dimensions )
            self.vanishing_point[1] = int(self.img_dimensions[0]*0.57)
        top =self.vanishing_point[1] + int(self.WRAPPED_WIDTH*0.15)
        self.bottom  = int(0.02*self.img_dimensions[0])
        bottom = self.img_dimensions[0]+self.bottom
        

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
        x1 = int(self.UNWARPED_SIZE[0]*lane_start[0])
        x2 = int(self.UNWARPED_SIZE[0]*lane_start[1])
        self.lane.leftx.append(x1)
        self.lane.rightx.append(x2)
        mask = self.compute_mask(img) 
        # x = np.linspace(0,mask.shape[0]-1,mask.shape[0])
        span = self.UNWARPED_SIZE[0]//5 
        x1 = x1-span + self.detect_lane_start(mask[:,x1-span :x1+span])
        x2 = x2-span + self.detect_lane_start(mask[:,x2-span :x2+span])
        self.lane.leftx.append(x1)
        self.lane.rightx.append(x2)
        
        self.lane.width = x2 -x1
        self.lane.previous_centers = np.ones(self.windows_per_line)*(x1+x2)//2
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

        pos = np.array((self.vanishing_point[0], bottom )).reshape(1, 1, -1)
        dst =  cv2.perspectiveTransform(pos, self.trans_mat).reshape(2)
        self.lane.centerx = dst[0]
        print("PERSPECTIVE TRANSFORMATION MATRIX COMPUTED")
        if self.verbose  > 1:       
            img_orig = cv2.polylines(self.image, [src_points.astype(np.int32)],True, (0,0,255), thickness=5)
            cv2.line(img, (int(x1), 0), (int(x1), self.UNWARPED_SIZE[1]), (255, 0, 0), 3)
            cv2.line(img, (int(x2), 0), (int(x2), self.UNWARPED_SIZE[1]), (0, 0, 255), 3)

            cv2.circle(img_orig,tuple(self.vanishing_point),10, color=RED, thickness=5)
            cv2.circle(img_orig,tuple(orig_points),10, color=GRAY, thickness=4)
            cv2.imwrite(self.temp_dir+"vanishing_point.jpg",img_orig)
            cv2.imwrite(self.temp_dir+"lane_width.jpg",img)
            cv2.imwrite(self.temp_dir+"perspective_lines.jpg",img2)
            cv2.imwrite(self.temp_dir+"mask.jpg",mask)
            img = cv2.bitwise_and(img, img, mask =  mask )
            cv2.imwrite(self.temp_dir+"masked_regions.jpg",img)
            cv2.imwrite(self.temp_dir+"edges.jpg",edges*roi)
            return img_orig
        return

    def calculate_position(self, box: OBSTACLE):
            pos = np.array((box.xmax/2+box.xmin/2, box.ymax)).reshape(1, 1, -1)
            dst =  cv2.perspectiveTransform(pos, self.trans_mat).reshape(2)
            box.x =  int(dst[0])
            box.y = -int(dst[1])
            box.lane  = self.lane.calculate_position(box.x,box.y)
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
    
    def process_image(self, img,obstacles :[OBSTACLE] =[], alpha=.7, beta=0.3, gamma=0):
        """
        Attempts to find lane lines on the given image and returns an image with lane area colored in green
        as well as small intermediate images overlaid on top to understand how the algorithm is performing
        """
        off =  int(100*self.font_sz)
        self.message = "["+ str(self.count)+"]"
        overlay =  img.copy()
        thres_img_psp= self.compute_lane_lines(overlay)
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
            if (box.lane == "my") and (box.velocity[1] < 0) : 
                color = RED
                if box.col_time > -4 :
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
        out_img = np.dstack((thres_img_psp,thres_img_psp,thres_img_psp))
        img = self.draw_lane_area(out_img, img)  
        if self.verbose >2 : 
            
            drawn_lines = self.draw_lane_lines(out_img)        
            drawn_hotspots = self.draw_lines_hotspots(out_img, obstacles)
            img = self.combine_images(img, drawn_lines,drawn_hotspots)
            img = self.draw_lane_curvature_text(img,)
            
        
        return img
    
    def draw_lane_curvature_text(self, img):
        """
        Returns an image with curvature information inscribed
        """
        sz = self.font_sz*3
        offset_y = self._pip_size[1] * 1 + self._pip__y_offset * 5
        offset_x = self._pip__x_offset
        
        template = "{0:17}{1:17}"
        txt_header = template.format("Curvature ", "Offset") 
        # print(txt_header)
        txt_values = template.format("{:d}m".format(self.lane.curvature), 
                                     "{:.2f}m Left".format(self.lane.lane_offset*self.xm_per_px))
        if self.lane.lane_offset < 0.0:
            txt_values = template.format("{:d}m".format(self.lane.curvature), 
                                     "{:.2f}m Right".format(self.lane.lane_offset*self.xm_per_px))
            
         

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
        small_hotspots = cv2.resize(lane_hotspots_img, self._pip_size)           
        lane_area_img[self._pip__y_offset: self._pip__y_offset + self._pip_size[1], self._pip__x_offset: self._pip__x_offset + self._pip_size[0]] = small_lines
        start_offset_y = self._pip__y_offset 
        start_offset_x = 2 * self._pip__x_offset + 1 * self._pip_size[0]
        lane_area_img[start_offset_y: start_offset_y + self._pip_size[1], start_offset_x: start_offset_x + self._pip_size[0]] = small_hotspots


        
        return lane_area_img
    
        
    def draw_lane_area(self, warped_img, undist_img):
        """
        Returns an image where the inside of the lane has been colored in bright green
        """
        color_warp = np.zeros_like(warped_img).astype(np.uint8)
        pts_left = np.array([np.transpose(np.vstack([self.lane.leftFit, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.lane.rightFit, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]),GREEN)
        newwarp = cv2.warpPerspective(color_warp, self.inv_trans_mat, (undist_img.shape[1], undist_img.shape[0])) 
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
        return result
        
        
    def draw_lane_lines(self, img):
        """
        Returns an image where the computed lane lines have been drawn on top of the original warped binary image
        """
        out_img = img.copy()

        for low_pt, high_pt in self.lane.left_windows:
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 1)

        for low_pt, high_pt in self.lane.right_windows:            
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 1) 
    
        return out_img    
    
    


    def draw_lines_hotspots(self, img,  obstacles:[OBSTACLE] =  []):
        """
        Returns a RGB image where the portions of the lane lines that were
        identified by our pipeline are colored in yellow (left) and blue (right)
        """
        sz = self.font_sz*12
        out_img = img.copy()
        lx =  self.lane.x - self.lane.width//2
        rx = self.lane.x + self.lane.width//2 

        pts_left = np.dstack((self.lane.leftFit, self.ploty)).astype(np.int32)
        pts_right = np.dstack((self.lane.rightFit, self.ploty)).astype(np.int32)
        pts_cntr = np.dstack((self.lane.rightFit - self.lane.width//2, self.ploty)).astype(np.int32)
        cv2.polylines(out_img, pts_left, False, BLUE, 2)
        cv2.polylines(out_img, pts_right, False, RED, 2)
        cv2.polylines(out_img, pts_cntr, False, YELLOW, 15)
        for i in range(len(self.lane.x)) :
            cv2.circle( out_img,( lx[i],-self.lane.y[i]), 4, BLUE, -1)
            cv2.circle( out_img,( rx[i],-self.lane.y[i]), 4, RED, -1)
        for i in range(len(obstacles)):
            box =  obstacles[i]
            color = GREEN
            if (box.col_time) and (box.lane == "my") and (box.col_time < 0) and (box.col_time>-4) :
                color = RED
            cv2.putText(out_img,str(box._id),(box.x, -box.y), self.font, sz, color, 8, cv2.LINE_AA)     
        return out_img

  
        
    def detect_lane_start(self, image):
        histx =   np.sum(image[image.shape[0]*4//5:,:], axis=0)
        return np.argmax(histx)
    
        
    def compute_lane_lines(self, img):
        """
        Returns the tuple (left_lane_line, right_lane_line) which represents respectively the LANE_LINE instances for
        the computed left and right lanes, for the supplied binary warped image
        """

        self.lane.left_windows = []
        self.lane.right_windows = []
        undst_img  = cv2.bitwise_and(img, img, mask =  self.lane_roi )
        pp_img = cv2.warpPerspective(undst_img, self.trans_mat, (self.UNWARPED_SIZE[1],self.UNWARPED_SIZE[0]))
        warped_img = self.compute_mask(pp_img)


        
        x1_av =  int(np.average(self.lane.leftx))
        x2_av  = int(np.average(self.lane.rightx))
        x1 = min(max(x1_av,self.margin), self.UNWARPED_SIZE[0]-self.margin*2)
        x2 = max( min(x2_av,self.UNWARPED_SIZE[0]-self.margin-1), x1+self.margin*2)
        leftx_current = x1-self.margin + self.detect_lane_start(warped_img[:,x1-self.margin :x1+self.margin])
        rightx_current = x2-self.margin + self.detect_lane_start(warped_img[:,x2-self.margin :x2+self.margin])
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])  
        self.lane.width= int(x2_av - x1_av)                
        centerx_current = (x2_av - x1_av) //2
        pointx = []
        pointy=[]
        center_idx = []
        curve_compute =  0
        self.max_gap =  0 
        gap  = 0
        for window in self.windows_range:
            win_y_low = warped_img.shape[0] - (window + 1)* self.window_height
            win_y_high = warped_img.shape[0] - window * self.window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            self.lane.left_windows.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            self.lane.right_windows.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])            
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            centerx =np.array([])
            centery =np.array([])
            s = 0
            if (len(good_left_inds) > self.minpix) and (len(good_left_inds) < self.maxpix):
                x = mode(nonzerox[good_left_inds])[0]
                y = mode(nonzeroy[good_left_inds])[0]
                s +=1
                centerx = x + self.lane.width//2
                centery = -y 
            if (len(good_right_inds) > self.minpix) and  (len(good_right_inds) < self.maxpix):
                s += 1
                x = mode(nonzerox[good_right_inds])[0]
                y = mode(nonzeroy[good_right_inds])[0]
                centerx = np.append(centerx, x-self.lane.width//2)
                centery =  np.append(centery,-y)

            
            if s > 0:    
                centerx_current = int(np.average(centerx))
                pointx.append(centerx_current)
                pointy.append(int(np.average(centery)))
                
                gap = 0
            else :
                gap +=1
                if len(center_idx) > 5 :
                    if curve_compute% 5 == 0 :
                        self.coef,_ =curve_fit(polyfunc,pointy,np.array(pointx), p0=self.coef)
                    centerx_current = int(np.polyval(self.coef, (window+1)*self.window_height))
                    curve_compute += 1
                else :
                    centerx_current = self.lane.previous_centers[window-self.window_offset]
            self.max_gap =  max(self.max_gap, gap)
            leftx_current = int(centerx_current -self.lane.width/2)
            rightx_current = int(centerx_current +self.lane.width/2)
        if (not self.lane_change): 
            if (self.max_gap > self.max_gap_th) and (self.count>0) :
                self.lane.left_windows = []
                self.lane.right_windows = []
                self.message+="SKIPPED  "+ str(self.max_gap)
                self.count+=1
                self.n_gap_skip+=1
                self.compute_bounds(cv2.cvtColor(pp_img, cv2.COLOR_BGR2HLS))
                return warped_img
            status , message =  self.lane.addlane(pointy,np.array(pointx) )
            
            self.message+= message
            if not status : 
                self.compute_bounds(cv2.cvtColor(pp_img, cv2.COLOR_BGR2HLS)) 
            else:
                self.lane.compute_curvature(self.px_per_ym,self.px_per_xm)
        self.count+=1
        return warped_img

if __name__ == "__main__":


    video_reader =  cv2.VideoCapture("videos/nice_road.mp4") 

    fps =  video_reader.get(cv2.CAP_PROP_FPS)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    pers_frame_time = 398#180# seconds
    pers_frame = int(pers_frame_time *fps)
    video_reader.set(1,pers_frame)
    ret, image = video_reader.read()
    ld = LANE_DETECTION( image,fps, 
                        yellow_lower = np.uint8([ 20, 50,   110]),
                        yellow_upper = np.uint8([35, 255, 255]),
                        white_lower = np.uint8([ 0, 140,   0]), 
                        white_upper = np.uint8([255, 255, 100]), 
                        lum_factor = 110,
                        lane_start=[0.2,0.5])






