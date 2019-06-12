from camera import CAMERA
from yolo_model import BoundBox,  YOLO 
from utils.bbox import bbox_iou 
from lane_detection import LANE_DETECTION
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
        self.update_coord(box)
        self.update_score(box)
        self.history : np.ndarray = []
        self.position_hist = []
        self.velocity = np.zeros((2))
        self.position = dst
        self.score=box.score
        self.label = box.label

    def update_obstacle(self, box: BoundBox, dst,  fps) :
        self.position_hist.append((self.xmin, self.ymin, self.xmax,self.ymax))
        self.update_coord(box)
        old_loc = self.position
        self.history.append(old_loc)
        self.col_time = min(dst[1]/(self.velocity[1]+0.001),99)
        if self.__count % self.PERIOD == 0 :
            self.velocity = (old_loc-dst ) * fps/self.PERIOD     
        self.__count += 1

    def update_coord(self,box):
        self.xmax = box.xmax
        self.xmin =  box.xmin
        self.ymin  =  box.ymin
        self.ymax =  box.ymax
        self.xmid = int((box.xmax+box.xmin)/2)
        self.ymid = int((box.ymax+box.ymin)/2)

    def update_score(self,box):     
        self.score=box.score
        self.label = box.label 

    def update_box(self,box):
        self.update_coord(box)
        self.update_score(box)
        
        
        
  
    
    
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
        self.lane = LANE_DETECTION(self.image)
        self.temp_dir = './images/detection/'
        self.size : (int , int) =  (self.image.shape[0] ,  self.image.shape[1] )
        # self.UNWARPED_SIZE  = (int(self.size[0]),int(self.size[1]*.4))
        # self.WRAPPED_WIDTH =  int(self.UNWARPED_SIZE[0]*1.25)
        # self.trans_mat  = None
        # self.inv_trans_mat  = None
        # self.pixels_per_meter = [0,0]
        self.perspective_done_at = datetime.utcnow().timestamp()
        self.img_shp =  (self.image.shape[1], self.image.shape[0] )
        self.area =  self.img_shp[0]*self.img_shp[1]
        # self.image =  self.camera.undistort(self.image)
        ### OBJECT DETECTION AND TRACKING
        self.yolo =  YOLO()
        self.first_detect = True
        self.obstacles :[OBSTACLE] =[]
        self.__yp = int(self.YOLO_PERIOD*self.fps)
        ### LANE FINDER 
        # self.lane_found = False
        self.count = 0
        # self.mask = np.zeros((self.UNWARPED_SIZE[1], self.UNWARPED_SIZE[0], 3), dtype=np.uint8)
        # self.roi_mask = np.ones((self.UNWARPED_SIZE[1], self.UNWARPED_SIZE[0], 3), dtype=np.uint8)
        # self.total_mask = np.zeros_like(self.roi_mask)
        # self.warped_mask = np.zeros((self.UNWARPED_SIZE[1], self.UNWARPED_SIZE[0]), dtype=np.uint8)
        # self.lane_count = 0


        # self.left_line = LaneLineFinder(self.UNWARPED_SIZE, self.pixels_per_meter, -1.8288)  # 6 feet in meters
        # self.right_line = LaneLineFinder(self.UNWARPED_SIZE, self.pixels_per_meter, 1.8288)


    def perspective_tfm(self ,  pos) : 
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > self.PERSP_PERIOD :
            self.lane = LANE_DETECTION(self.image)
        
        return cv2.perspectiveTransform(pos, self.lane.inv_trans_mat)
        #cv2.warpPerspective(image, self.trans_mat, self.UNWARPED_SIZE)
  
    
    def get_speed(self):
        return 30
    

    def determine_lane(self, box:OBSTACLE):
        points =np.array( [box.xmid, box.ymid], dtype='float32').reshape(1,1,2)
        new_points = cv2.perspectiveTransform(points,self.lane.trans_mat)
        new_points =  new_points.reshape(2)
        # left  = np.polyval(self.lane.previous_left_lane_line.polynomial_coeff,new_points[0]) - new_points[1]
        # right = np.polyval(self.lane.previous_right_lane_line.polynomial_coeff,new_points[0]) - new_points[1]
        # points =  points.reshape(2)
        # left2  = np.polyval(self.lane.previous_left_lane_line.polynomial_coeff,points[1]) - points[0]
        # right2 = np.polyval(self.lane.previous_right_lane_line.polynomial_coeff,points[1]) - points[0]
        left= np.polyval(self.lane.previous_left_lane_line.polynomial_coeff,new_points[1]) - new_points[0]
        right= np.polyval(self.lane.previous_right_lane_line.polynomial_coeff,new_points[1]) - new_points[0]
        status = "my"
        if left < 0 and right <0:
            status = "right"
        elif right>0 and left >0 :
            status = "left"
        print(box._id,status, left, right)
        return status

    def calculate_position(self, box: BoundBox):
        if (self.perspective_done_at > 0):
            pos = np.array((box.xmax/2+box.xmin/2, box.ymax)).reshape(1, 1, -1)
            dst = self.perspective_tfm(pos).reshape(2)
            dst =  np.array([dst[0]/self.lane.px_per_xm,(self.lane.UNWARPED_SIZE[1]-dst[1])/self.lane.px_per_ym])
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
        self.lane.process_image( img)
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
            self.lane = LANE_DETECTION(self.image)
        return cv2.warpPerspective(img, self.lane.trans_mat, self.lane.UNWARPED_SIZE, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)

    def unwarp(self, img):
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > self.PERSP_PERIOD :
            self.lane = LANE_DETECTION(self.image)
        return cv2.warpPerspective(img, self.lane.trans_mat, self.img_shp, flags=cv2.WARP_FILL_OUTLIERS +
                                                                     cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)


    @staticmethod
    def put_text(overlay,text, coord, color=WHITE):
        ft_sz = 5e-4 * overlay.shape[0]
        sz = ft_sz*25
        font =  cv2.FONT_HERSHEY_SIMPLEX
        rect_ht = int(sz *1.2)
        rect_wd = int(len(text)*sz*0.8)
        p1 = (coord[0], coord[1])
        p2 = (coord[0]+rect_wd, coord[1]-rect_ht)
        cv2.rectangle(overlay, p1, p2,  (0, 0, 0),-1)
        # cv2.putText(overlay, text,   coord,  font, ft_sz, (0, 0, 0), 5)
        cv2.putText(overlay, text,   coord,  font, ft_sz, color, 1)

        return 

    def draw_lane_weighted(self, image, thickness=5, alpha=1, beta=0.8, gamma=0):
        overlay = image.copy()
        font =  cv2.FONT_HERSHEY_COMPLEX_SMALL
        for i , box in enumerate(self.obstacles):
            past=[box.xmin,box.ymin,box.xmax,box.ymax]

            t1 = classes[obstructions[box.label]] +" ["+str(int(box.position[1])) + "m]" 
            t2 = "("+str(int(box.score*100))+"%) ID: " +str(box._id)
            b1= "Lane "+ box.lane
            b2 = str(int(box.velocity[1]))+"m/s"
            b3 = "Col "+str(int(box.col_time))+"s"
            pt1 = (box.xmin, box.ymin-10)
            pt2 =  (box.xmin, box.ymin)
            pb1 = (box.xmin, box.ymax+10)
            pb2 = (box.xmin, box.ymax+20)
            pb3 =  (box.xmin, box.ymax+30)
            self.put_text(overlay, t1,   pt1)
            self.put_text(overlay, t2,   pt2)
            self.put_text(overlay, b1,   pb1)
            self.put_text(overlay, b2,   pb2)
            self.put_text(overlay, b3,   pb3)

            # centr_lbl = classes[obstructions[box.label]] +"|"+ str(int(box.score*100))+"%" 
            # top_lbl = str(box._id)+"|"+box.lane
            # bot_lbl = str(int(box.col_time)) +"s | " + str(int(box.position[0])) + " m | " +str(int(box.velocity[0])) + " m/s"
            past_center =  (int(past[0]/2+past[2]/2), past[3])
            # font_thk =1
            
            color = ORANGE if box.velocity[1] > 0  else GREEN
            cv2.rectangle(overlay, (box.xmin,box.ymin), (box.xmax,box.ymax), color,2)
            cv2.circle(overlay,past_center,1, GRAY,2)
            # self.put_text(overlay, top_lbl,   (box.xmin+5, box.ymin))
            # self.put_text(overlay, centr_lbl,   (box.xmin, box.ymid))
            # self.put_text(overlay, bot_lbl,   (box.xmin, box.ymax),)
            # cv2.putText(overlay, top_lbl, 
            #             (box.xmin+5, box.ymin),  font, 
            #             6e-4 * image.shape[0],   YELLOW, font_thk)
            # cv2.putText(overlay, centr_lbl, 
            #             (box.xmin, box.ymid),  font, 
            #             6e-4 * image.shape[0], LIGHT_CYAN  , font_thk)
            # cv2.putText(overlay, bot_lbl, 
            #             (box.xmin, box.ymax),  font, 
            #             6e-4 * image.shape[0], LIGHT_CYAN  , font_thk)
        image =  cv2.addWeighted(image, alpha, overlay, beta, gamma)
        cv2.imwrite(self.temp_dir+"detect.jpg", image)
        # left_line = self.left_line.get_line_points()
        # right_line = self.right_line.get_line_points()
        # both_lines = np.concatenate((left_line, np.flipud(right_line)), axis=0)
        # lanes = np.zeros((self.UNWARPED_SIZE[1], self.UNWARPED_SIZE[0], 3), dtype=np.uint8)
        # center_line =  (left_line +  right_line)//2
        # if 1:#self.lane_found:
        #     # cv2.fillPoly(lanes, [both_lines.astype(np.int32)], LIGHT_CYAN)
        #     # cv2.polylines(lanes, [left_line.astype(np.int32)], False,RED ,thickness=5 )
        #     # cv2.polylines(lanes, [right_line.astype(np.int32)],False,  DARK_BLUE, thickness=5)
        #     # cv2.polylines(lanes, [center_line.astype(np.int32)],False,  ORANGE, thickness=5)

        #     # lanes_unwarped = self.unwarp(lanes)
        #     overlay = cv2.addWeighted(image, alpha, lanes_unwarped, beta, gamma)
        #     cv2.imwrite(self.temp_dir+"detection.jpg", overlay)
        # else:
        #     cv2.putText(image, "Lane lost!", (550, 170), font, fontScale=2.5,
        #                 thickness=5, color=(255, 255, 255))
        #     cv2.putText(image, "Lane lost!", (550, 170), font, fontScale=2.5,
        #                 thickness=3, color=(0, 0, 0))
        #     cv2.imwrite(self.temp_dir+"detection.jpg", image)
        
        return 
    
     
    
    
    def vehicle_speed(self) :
        return
def main():
    import os
    files =  os.listdir("./images/from_video/")
    files  = [f for f in files if f[-3:]=="jpg"]
    files.sort()
    # files =["test1.jpg","test4.jpg","test6.jpg","test5.jpg",]

    frame  = FRAME( image=cv2.imread("./images/from_video/"+files[-1]))
    # frame.find_lane(cv2.imread("./images/from_video/"+files[0]), plot=True)
    # frame.detect_objects(cv2.imread("./images/"+files[0]))
    for i, f in enumerate(files[0:72]):
        # frame.find_lane(cv2.imread("./images/"+f),plot=True)
       frame.update_trackers(cv2.imread("./images/from_video/"+f),plot=True)
    #    cv2.waitKey(1)
       input("Press Enter to continue...")
    
main()
