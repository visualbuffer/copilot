from camera import CAMERA
from yolo_model import BoundBox,  YOLO 
from utils.bbox import bbox_iou 
from lane import LaneLineFinder, get_center_shift, get_curvature
import numpy as np
import cv2
from datetime import datetime
from PIL import Image

# yolo_detector =  YOLO(score =  0.3, iou =  0.5, gpu_num = 0)
WHITE = (255, 255, 255)
YELLOW = (66, 244, 238)
GREEN = (80, 220, 60)
LIGHT_CYAN = (255, 255, 224)
DARK_BLUE = (139, 0, 0)
GRAY = (128, 128, 128)
RED = (0,0,255)
ORANGE =(0,165,255)

vehicles = [1,2,3,5,6,7,8]
animals =[15,16,17,18,19,21,22,23,]
humans =[0]
obstructions =  humans + animals + vehicles
classes = [#
    'person','bicycle','car','motorbike','aeroplane','bus',\
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


class OBSTACLE(BoundBox):
    xmax :int
    xmin :int
    ymin :int
    ymax :int
    xmid  :int
    ymid  :int
    lane : str
    tracker = None
    position  : [int,int]
    PERIOD = 5
    __count = 0

    def __init__(self,box: BoundBox,dst, _id) :
        self.col_time:float =999.0
        self._id = _id
        self.update_box(box)
        self.history : np.ndarray = []
        self.position_hist = []
        self.velocity = np.zeros((2))
        self.position = dst

    def update_obstacle(self, box: BoundBox, dst,  fps) :
        self.position_hist.append((self.xmin, self.ymin, self.xmax,self.ymax))
        self.update_box(box)
        old_loc = self.position
        self.history.append(old_loc)
        self.col_time = min(dst[1]/(self.velocity[1]+0.001),99)
        if self.__count % self.PERIOD == 0 :
            self.velocity = (old_loc-dst ) * fps/self.PERIOD     
        self.__count += 1
        
        
    def update_box(self,box):
        self.xmax = box.xmax
        self.xmin =  box.xmin
        self.ymin  =  box.ymin
        self.ymax =  box.ymax
        self.xmid = int((box.xmax+box.xmin)/2)
        self.ymid = int((box.ymax+box.ymin)/2)

        
        
  
    
    
class TRAFFIC_LIGHTS(OBSTACLE) :
  def __init__(self) :
    return None
  
  def detect_status(self):
    return None
    
class TRAFFIC_SIGNS(OBSTACLE):
  def __init__(self) :
    return None
  
  def decipher(self):
    return None


class FRAME :
    fps:float
    UNWARPED_SIZE :(int,int)
    LANE_WIDTH :int
    WRAPPED_WIDTH  :  int
    camera : CAMERA
    yolo : classmethod
    PERSP_PERIOD =  100000
    YOLO_PERIOD = 0.5 # SECONDS
    _defaults = {
        "id": 0,
        "first": True,
        "speed": 0,
        "n_objects" :0,
         "camera" : CAMERA(),
        "image" : [],
      
        "LANE_WIDTH" :  3.66,
        "fps" :22
        }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"  

    def __init__(self, **kwargs):
        # calc pers => detect cars and dist > detect lanes
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.speed =  self.get_speed()
        ### IMAGE PROPERTIES
        self.image : np.ndarray
        if  self.image.size ==0 :
            raise ValueError("No Image") 
        self.temp_dir = './images/detection/'
        self.size : (int , int) =  (self.image.shape[0] ,  self.image.shape[1] )
        self.UNWARPED_SIZE  = (int(self.size[1]*0.5),int(self.size[0]*2))
        self.WRAPPED_WIDTH =  int(self.UNWARPED_SIZE[0]*1.25)
        self.trans_mat  = None
        self.inv_trans_mat  = None
        self.pixels_per_meter = [0,0]
        self.perspective_done_at = 0
        self.img_shp =  (self.image.shape[1], self.image.shape[0] )
        self.area =  self.img_shp[0]*self.img_shp[1]
        # self.image =  self.camera.undistort(self.image)
        ### OBJECT DETECTION AND TRACKING
        self.yolo =  YOLO()
        self.first_detect = True
        self.obstacles :[OBSTACLE] =[]
        self.__yp = int(self.YOLO_PERIOD*self.fps)
        ### LANE FINDER 
        self.lane_found = False
        self.count = 0
        # self.cam_matrix = cam_matrix
        # self.dist_coeffs = dist_coeffs
        # self.img_size = img_size
        # self.UNWARPED_SIZE = UNWARPED_SIZE
        self.mask = np.zeros((self.UNWARPED_SIZE[1], self.UNWARPED_SIZE[0], 3), dtype=np.uint8)
        self.roi_mask = np.ones((self.UNWARPED_SIZE[1], self.UNWARPED_SIZE[0], 3), dtype=np.uint8)
        self.total_mask = np.zeros_like(self.roi_mask)
        self.warped_mask = np.zeros((self.UNWARPED_SIZE[1], self.UNWARPED_SIZE[0]), dtype=np.uint8)
        # self.trans_mat = transform_matrix
        self.lane_count = 0


        self.left_line = LaneLineFinder(self.UNWARPED_SIZE, self.pixels_per_meter, -1.8288)  # 6 feet in meters
        self.right_line = LaneLineFinder(self.UNWARPED_SIZE, self.pixels_per_meter, 1.8288)


    def perspective_tfm(self ,  pos) : 
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > self.PERSP_PERIOD :
            self.calc_perspective()
        
        return cv2.perspectiveTransform(pos, self.trans_mat)
        #cv2.warpPerspective(image, self.trans_mat, self.UNWARPED_SIZE)
  
    def calc_perspective(self, verbose =  True):
        roi = np.zeros((self.size[0], self.size[1]), dtype=np.uint8) # 720 , 1280
        roi_points = np.array([[0, self.size[0]],[self.size[1],self.size[0]],
                    [self.size[1]//2+100,-0*self.size[0]],
                     [self.size[1]//2-100,-0*self.size[0]]], dtype=np.int32)
        cv2.fillPoly(roi, [roi_points], 1)
        
        Lhs = np.zeros((2,2), dtype= np.float32)
        Rhs = np.zeros((2,1), dtype= np.float32)
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mn_hsl = np.median(grey) #grey.median()
        edges = cv2.Canny(grey, int(mn_hsl*4), int(mn_hsl*3))
        # edges = cv2.Canny(grey[:, :, 1], 500, 400)
        edges2 = edges*roi
        cv2.imwrite(self.temp_dir+"mask.jpg", edges2)
        lines = cv2.HoughLinesP(edges*roi,rho = 4,theta = np.pi/180,threshold = 4,minLineLength = 80,maxLineGap = 40)

       
        # print(lines)
        for line in lines:
            
            for x1, y1, x2, y2 in line:
                cv2.line(edges2, (x1,y1),(x2,y2), (255, 0, 0), 1)
                normal = np.array([[-(y2-y1)], [x2-x1]], dtype=np.float32)
                normal /=np.linalg.norm(normal)
                point = np.array([[x1],[y1]], dtype=np.float32)
                outer = np.matmul(normal, normal.T)
                Lhs += outer
                Rhs += np.matmul(outer, point)
        vanishing_point = np.matmul(np.linalg.inv(Lhs),Rhs)
        top = vanishing_point[1] + 50
        bottom = self.size[1]-100
        
        def on_line(p1, p2, ycoord):
            return [p1[0]+ (p2[0]-p1[0])/float(p2[1]-p1[1])*(ycoord-p1[1]), ycoord]


        #define source and destination targets
        p1 = [vanishing_point[0] - self.WRAPPED_WIDTH/2, top]
        p2 = [vanishing_point[0] + self.WRAPPED_WIDTH/2, top]
        p3 = on_line(p2, vanishing_point, bottom)
        p4 = on_line(p1, vanishing_point, bottom)
        src_points = np.array([p1,p2,p3,p4], dtype=np.float32)
        # print(src_points,vanishing_point)
        dst_points = np.array([[0, 0], [self.UNWARPED_SIZE[0], 0],
                            [self.UNWARPED_SIZE[0], self.UNWARPED_SIZE[1]],
                            [0, self.UNWARPED_SIZE[1]]], dtype=np.float32)

        self.trans_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inv_trans_mat = cv2.getPerspectiveTransform(dst_points,src_points)
        min_wid = 1000
        img = cv2.warpPerspective(self.image, self.trans_mat, self.UNWARPED_SIZE)
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        mask = grey[:,:,1]>128
        mask[:, :50]=0
        mask[:, -50:]=0
        mom = cv2.moments(mask[:,:self.UNWARPED_SIZE[0]//2].astype(np.uint8))
        x1 = mom["m10"]/mom["m00"]
        mom = cv2.moments(mask[:,self.UNWARPED_SIZE[0]//2:].astype(np.uint8))
        x2 = self.UNWARPED_SIZE[0]//2 + mom["m10"]/mom["m00"]

        if (x2-x1<min_wid):
            min_wid = x2-x1
        self.pixels_per_meter[0] = min_wid/self.LANE_WIDTH
        if False :#self.camera.callibration_done :
            Lh = np.linalg.inv(np.matmul(self.trans_mat, self.camera.cam_matrix))
        else:
            Lh = np.linalg.inv(self.trans_mat)
        self.pixels_per_meter[1] = self.pixels_per_meter[0] * np.linalg.norm(Lh[:,0]) / np.linalg.norm(Lh[:,1])
        self.perspective_done_at =  datetime.utcnow().timestamp()
        if verbose :       
            img_orig = cv2.polylines(self.image, [src_points.astype(np.int32)],True, (0,0,255), thickness=5)
            cv2.line(img, (int(x1), 0), (int(x1), self.UNWARPED_SIZE[1]), (255, 0, 0), 3)
            cv2.line(img, (int(x2), 0), (int(x2), self.UNWARPED_SIZE[1]), (0, 0, 255), 3)

            cv2.circle(img_orig,tuple(vanishing_point),10, color=(0,0,255), thickness=5)
      
            cv2.imwrite(self.temp_dir+"perspective1.jpg",img_orig)
            cv2.imwrite(self.temp_dir+"perspective2.jpg",img)
            # cv2.imshow(cv2.hconcat((img_orig, cv2.resize(img, img_orig.shape))))
        return
    
    def get_speed(self):
        return 30
    
    # def detect_objects(self, image, plot = False):
    #     if self.count% self.__yp == 0 :
    #         boxes= self.yolo.make_predictions(image,obstructions = obstructions,plot=True) 
    #         boxes =  self.map_frame2frame
    #         image  =  cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #         n_obs =  len(self.obstacles)
    #         for i in range(len(boxes)):
    #             tracker = cv2.TrackerKCF_create()# cv2.TrackerMIL_create()#  # Note: Try comparing KCF with MIL
    #             box = boxes[i]
    #             dst =  self.calculate_position(box)
    #             bbox = (box.xmin, box.ymin, box.xmax-box.xmin, box.ymax-box.ymin)
    #             # print(bbox)
    #             success = tracker.init(image, bbox )
    #             if success :
    #                 self.trackers.append(tracker)
    #                 obstacle =  OBSTACLE(box, i+n_obs,  dst)
    #                 self.obstacles.append(obstacle)
    #         if not self.first_detect :
    #             self.update_trackers(image)
    #     else:
   
    #         self.update_trackers(image,plot)
    #     return
    def determine_lane(self, box:OBSTACLE):
        points =np.array( [box.xmid, box.ymid], dtype='float32').reshape(1,1,2)
        new_points = cv2.perspectiveTransform(points,self.inv_trans_mat)
        new_points =  new_points.reshape(2)
        left  = np.polyval(self.left_line.poly_coeffs,new_points[0]) - new_points[1]
        right = np.polyval(self.right_line.poly_coeffs,new_points[0]) - new_points[1]

        left2  = np.polyval(self.left_line.poly_coeffs,new_points[1]) - new_points[0]
        right2 = np.polyval(self.right_line.poly_coeffs,new_points[1]) - new_points[0]
        status = "my"
        if left < 0 and right <0:
            status = "left"
        elif right>0 and left >0 :
            status = "right"
        print(box._id,status, left, right, "|", left2, right2)
        return status

    def calculate_position(self, box: BoundBox):
        if (self.perspective_done_at > 0):
            pos = np.array((box.xmax/2+box.xmin/2, box.ymax)).reshape(1, 1, -1)
            dst = self.perspective_tfm(pos).reshape(2)
            dst =  np.array([dst[0]/self.pixels_per_meter[0],(self.UNWARPED_SIZE[1]-dst[1])/self.pixels_per_meter[1]])
            return dst
        else:
            
            return np.array([0,0])
    

    @staticmethod
    def corwh2box(corwh):
        box=BoundBox( int(corwh[0]), int(corwh[1]), int(corwh[0] + corwh[2]), int(corwh[1] + corwh[3]))
        return box

    def tracker2object(self, boxes : [OBSTACLE], th =  0.5) : 
        n_b = len(boxes)
        n_o =  len(self.obstacles)
        iou_mat =  np.zeros((n_o,n_b))
        for i in range(n_o):
            for j in range(n_b):
                iou_mat[i,j] =  bbox_iou(self.obstacles[i],boxes[j])
        count =  min(n_b,n_o)
        used = []
        while count >0 :
            r,k  = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
            if iou_mat[r,k] > th :
                used.append(k)
                obstacle  = self.obstacles[r]
                box = boxes[k]
                obstacle.update_box(box)
                self.obstacles[r] =  obstacle
            iou_mat[r,:] =  -99
            iou_mat[:,k] =  -99
            count = count -1
        idx = range(n_b)
        idx =  [elem for elem in idx if elem not in used]
        for i, c in enumerate(idx):
            dst  =  self.calculate_position(boxes[c])
            obstacle = OBSTACLE(boxes[c],dst,i+n_o)
            self.obstacles.append(obstacle)
        return
    
    def update_trackers(self, img,plot = False):
        image = img.copy()
        self.find_lane( img, distorted=False, reset=False)
        for n, obs in enumerate(self.obstacles):

            success, corwh = obs.tracker.update(image)
            # print("tracking", corwh ,  self.obstacles[n].xmin,self.obstacles[n].ymin,self.obstacles[n].xmax,self.obstacles[n].ymax)
            if not success :
                del self.obstacles[n]

                continue
            box = self.corwh2box(corwh)
            dst = self.calculate_position( box)  
            self.obstacles[n].update_obstacle(box, dst, self.fps)
        
        if self.count% self.__yp == 0 :
            boxes= self.yolo.make_predictions(image,obstructions = obstructions,plot=True) 
            self.tracker2object(boxes)
            image  =  cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            n_obs =  len(self.obstacles)
            for i in range(n_obs):
                tracker = cv2.TrackerKCF_create()# cv2.TrackerMIL_create()#  # Note: Try comparing KCF with MIL
                box = self.obstacles[i]
                bbox = (box.xmin, box.ymin, box.xmax-box.xmin, box.ymax-box.ymin)
                # print(bbox)
                success = tracker.init(image, bbox )
                if success :
                    self.obstacles[i].tracker=tracker

        
        self.count +=1
        for i in range(len(self.obstacles)):
            lane = self.determine_lane(self.obstacles[i])
            self.obstacles[i].lane =  lane

        if plot and self.count>1: 
           self.draw_lane_weighted(img)
        return

    def warp(self, img):
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > self.PERSP_PERIOD :
            self.calc_perspective()
        return cv2.warpPerspective(img, self.trans_mat, self.UNWARPED_SIZE, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)

    def unwarp(self, img):
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > self.PERSP_PERIOD :
            self.calc_perspective()
        return cv2.warpPerspective(img, self.trans_mat, self.img_shp, flags=cv2.WARP_FILL_OUTLIERS +
                                                                     cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)
    def equalize_lines(self, alpha=0.9):
        mean = 0.5 * (self.left_line.coeff_history[:, 0] + self.right_line.coeff_history[:, 0])
        self.left_line.coeff_history[:, 0] = alpha * self.left_line.coeff_history[:, 0] + \
                                             (1-alpha)*(mean - np.array([0,0, 1.8288], dtype=np.uint8))
        self.right_line.coeff_history[:, 0] = alpha * self.right_line.coeff_history[:, 0] + \
                                              (1-alpha)*(mean + np.array([0,0, 1.8288], dtype=np.uint8))

    def find_lane(self, img, distorted=False, reset=False):
        # undistort, warp, change space, filter
        # save =  "detecetion.jpg"
        image =  img.copy()
        if distorted:
            image = self.camera.undistort(image)
        if reset:
            self.left_line.reset_lane_line()
            self.right_line.reset_lane_line()

        image = self.warp(image)

        # cv2.imwrite(self.temp_dir+save,image)
        img_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        img_hls = cv2.medianBlur(img_hls, 5)
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img_lab = cv2.medianBlur(img_lab, 5)

        big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        greenery = (img_lab[:, :, 2].astype(np.uint8) > 130) & cv2.inRange(img_hls, (0, 0, 50), (35, 190, 255))

     
        road_mask = np.logical_not(greenery).astype(np.uint8) & (img_hls[:, :, 1] < 250)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, small_kernel)
        road_mask = cv2.dilate(road_mask, big_kernel)

     
        img2, contours, hierarchy = cv2.findContours(road_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        biggest_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area>biggest_area:
                biggest_area = area
                biggest_contour = contour
        road_mask = np.zeros_like(road_mask)
        cv2.fillPoly(road_mask, [biggest_contour],  1)

   
        self.roi_mask[:, :, 0] = (self.left_line.line_mask | self.right_line.line_mask) & road_mask
        self.roi_mask[:, :, 1] = self.roi_mask[:, :, 0]
        self.roi_mask[:, :, 2] = self.roi_mask[:, :, 0]
        
      
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
        black = cv2.morphologyEx(img_lab[:,:, 0], cv2.MORPH_TOPHAT, kernel)
        lanes = cv2.morphologyEx(img_hls[:,:,1], cv2.MORPH_TOPHAT, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        lanes_yellow = cv2.morphologyEx(img_lab[:, :, 2], cv2.MORPH_TOPHAT, kernel)

        self.mask[:, :, 0] = cv2.adaptiveThreshold(black, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -6)
        self.mask[:, :, 1] = cv2.adaptiveThreshold(lanes, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
        self.mask[:, :, 2] = cv2.adaptiveThreshold(lanes_yellow, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                   13, -1.5)
        self.mask *= self.roi_mask
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.total_mask = np.any(self.mask, axis=2).astype(np.uint8)
        self.total_mask = cv2.morphologyEx(self.total_mask.astype(np.uint8), cv2.MORPH_ERODE, small_kernel)

        left_mask = np.copy(self.total_mask)
        right_mask = np.copy(self.total_mask)
        if self.right_line.lane_line_found :
            left_mask = left_mask & np.logical_not(self.right_line.line_mask) & self.right_line.other_line_mask
        if self.left_line.lane_line_found :
            right_mask = right_mask & np.logical_not(self.left_line.line_mask) & self.left_line.other_line_mask
        print("LEFT")
        self.left_line.find_lane_line(left_mask, reset)
        print("RIGHT")
        self.right_line.find_lane_line(right_mask, reset)
        self.lane_found = self.left_line.lane_line_found and self.right_line.lane_line_found

        if self.lane_found:
            self.equalize_lines(0.875)


    def draw_lane_weighted(self, image, thickness=5, alpha=1, beta=0.6, gamma=0):
        for i , box in enumerate(self.obstacles):
            past=[box.xmin,box.ymin,box.xmax,box.ymax]
            pos_lbl =  str(int(box.velocity[0]))+"," + str(int(box.velocity[1]))
            center =  (int(past[0]/2+past[2]/2), past[3])
            centr_lbl = str(box._id)+"|"+box.lane
            color = ORANGE if box.velocity[1] > 0  else GREEN
            cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), color, 1)
            cv2.circle(image,center,2, DARK_BLUE,2)
            cv2.circle(image,(int(box.xmin/2+box.xmax/2),box.ymax),2, color,2)
            cv2.putText(image, pos_lbl, 
                        (box.xmin+5, box.ymid),  cv2.FONT_HERSHEY_SIMPLEX, 
                        5e-4 * image.shape[0],   YELLOW, 1)
            cv2.putText(image, centr_lbl, 
                        (box.xmid, box.ymin),  cv2.FONT_HERSHEY_SIMPLEX, 
                        5e-4 * image.shape[0], LIGHT_CYAN  , 1)


        left_line = self.left_line.get_line_points()
        right_line = self.right_line.get_line_points()
        both_lines = np.concatenate((left_line, np.flipud(right_line)), axis=0)
        lanes = np.zeros((self.UNWARPED_SIZE[1], self.UNWARPED_SIZE[0], 3), dtype=np.uint8)
        center_line =  (left_line +  right_line)//2
        if self.lane_found:
            # input("Press Enter to continue...")
            # cv2.fillPoly(lanes, [both_lines.astype(np.int32)], (0, 255, 0))
            cv2.fillPoly(lanes, [both_lines.astype(np.int32)], LIGHT_CYAN)
            cv2.polylines(lanes, [left_line.astype(np.int32)], False,RED ,thickness=5 )
            cv2.polylines(lanes, [right_line.astype(np.int32)],False,  DARK_BLUE, thickness=5)
            cv2.polylines(lanes, [center_line.astype(np.int32)],False,  ORANGE, thickness=5)
            mid_coef = 0.5 * (self.left_line.poly_coeffs + self.right_line.poly_coeffs)
            curve = get_curvature(mid_coef, img_size=self.UNWARPED_SIZE, pixels_per_meter=self.left_line.pixels_per_meter)
            shift = get_center_shift(mid_coef, img_size=self.UNWARPED_SIZE,
                                     pixels_per_meter=self.left_line.pixels_per_meter)
            cv2.putText(image, "Road curvature: {:6.2f}m".format(curve), (420, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                        thickness=5, color=(255, 255, 255))
            cv2.putText(image, "Road curvature: {:6.2f}m".format(curve), (420, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                        thickness=3, color=(0, 0, 0))
            cv2.putText(image, "Car position: {:4.2f}m".format(shift), (460, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                        thickness=5, color=(255, 255, 255))
            cv2.putText(image, "Car position: {:4.2f}m".format(shift), (460, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                        thickness=3, color=(0, 0, 0))
            lanes_unwarped = self.unwarp(lanes)
            overlay = cv2.addWeighted(image, alpha, lanes_unwarped, beta, gamma)
            cv2.imwrite(self.temp_dir+"detection.jpg", overlay)
        else:
            # # warning_shape = self.warning_icon.shape
            # corner = (10, (image.shape[1]-warning_shape[1])//2)
            # patch = image[corner[0]:corner[0]+warning_shape[0], corner[1]:corner[1]+warning_shape[1]]
            # # patch[self.warning_icon[:, :, 3] > 0] = self.warning_icon[self.warning_icon[:, :, 3] > 0, 0:3]
            # image[corner[0]:corner[0]+warning_shape[0], corner[1]:corner[1]+warning_shape[1]]=patch
            cv2.putText(image, "Lane lost!", (550, 170), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                        thickness=5, color=(255, 255, 255))
            cv2.putText(image, "Lane lost!", (550, 170), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5,
                        thickness=3, color=(0, 0, 0))
            cv2.imwrite(self.temp_dir+"detection.jpg", image)
        
        return 
    
     
    
    
    def vehicle_speed(self) :
        return
def main():
    import os
    files =  os.listdir("./images/")
    files  = [f for f in files if f[-3:]=="jpg"]
    files.sort(reverse=True)

    frame  = FRAME( image=cv2.imread("./images/"+files[0]))
    frame.calc_perspective()
    # frame.find_lane(cv2.imread("./images/"+files[10]), plot=True)
    # frame.detect_objects(cv2.imread("./images/"+files[0]))
    for i, f in enumerate(files[0:72]):
        # frame.find_lane(cv2.imread("./images/"+f),plot=True)
       frame.update_trackers(cv2.imread("./images/"+f),plot=True)
    #    cv2.waitKey(1)
       input("Press Enter to continue...")
    
main()
