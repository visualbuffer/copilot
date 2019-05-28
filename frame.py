from camera import CAMERA
from yolo_model import YOLO ,  BoundBox
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
RED = (255,0,0)
ORANGE =(255,165,0)

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
    PERIOD = 5
    __count = 0

    def __init__(self,box: BoundBox, _id,  postion) :
        self.col_time:float =999.0
        self._id = _id
        self.__update_box(box)
        self.position = postion
        self.history : np.ndarray = []
        self.position_hist = []
        self.velocity = np.zeros((2))


    def update_obstacle(self, box: BoundBox, dst, fps) :
        self.position_hist.append((self.xmin, self.ymin, self.xmax,self.ymax))
        self.__update_box(box)
        old_loc = self.position
        self.history.append(old_loc)
        self.position = dst
        if self.__count % self.PERIOD == 0 :
            self.velocity = (old_loc-dst ) * fps/self.PERIOD
        self.__count += 1
        self.col_time = min(dst[1]/(self.velocity[1]+0.001),99)
        
    def __update_box(self,box):
        self.xmax = box.xmax
        self.xmin =  box.xmin
        self.ymin  =  box.ymin
        self.ymax =  box.ymax
        self.xmid = int((box.xmax+box.xmin)/2)
        self.ymid = int((box.ymax+box.ymin)/2)

class VEHICLE(OBSTACLE) :
  def __init__(self) :
    self.numplate=None
    self.rx = None
    self.ry = None
    self.vx = 0
    self.first = True
    
  def detect_number_plate(self):
    return None
  
  def detect_position(self) :
    return None
  
  def detect_velocity(self) :
    return None
    
  
    
    
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


    _defaults = {
        "id": 0,
        "first": True,
        "speed": 0,
        "n_objects" :0,
         "camera" : CAMERA(),
        "image" : [],
        "UNWARPED_SIZE" : (500, 600),
        "WRAPPED_WIDTH" :  530,
        "LANE_WIDTH" :  3.5,
        "fps" :22
        }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"  

    def __init__(self, **kwargs):
        self.fps:float
        self.UNWARPED_SIZE :(int,int)
        self.LANE_WIDTH :int
        self.WRAPPED_WIDTH  :  int
        self.camera : CAMERA
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.speed =  self.get_speed()
        self.image : np.ndarray
        if  self.image.size ==0 :
            raise ValueError("No Image") 
        # self.image =  self.camera.undistort(self.image)
        self.temp_dir = './images/detection/'
        self.size : (int , int) =  (self.image.shape[0] ,  self.image.shape[1] )
        self.M  = None
        self.pix_per_meter_x = 0
        self.pix_per_meter_y = 0
        self.perspective_done_at = 0
        self.yolo =  YOLO()
        self.first_detect = True
        self.trackers = []
        self.obstacles :[OBSTACLE] =[]
        self.img_shp =  self.image.shape
        self.area =  self.img_shp[0]*self.img_shp[1]

    def perspective_tfm(self ,  pos) : 
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > 10000 :
            self.calc_perspective()
        
        return cv2.perspectiveTransform(pos, self.M)
        #cv2.warpPerspective(image, self.M, self.UNWARPED_SIZE)
  
    def calc_perspective(self, verbose =  False):
        roi = np.zeros((self.size[0], self.size[1]), dtype=np.uint8) # 720 , 1280
        roi_points = np.array([[0, self.size[0]-150],[self.size[1],self.size[0]-150],
                    [self.size[1]//2+100,self.size[0]//2 -200],
                     [self.size[1]//2-100,self.size[0]//2 -200]], dtype=np.int32)
        cv2.fillPoly(roi, [roi_points], 1)
        Lhs = np.zeros((2,2), dtype= np.float32)
        Rhs = np.zeros((2,1), dtype= np.float32)
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mn_hsl = np.median(grey) #grey.median()
        edges = cv2.Canny(grey, int(mn_hsl*4), int(mn_hsl*3))
        # edges = cv2.Canny(grey[:, :, 1], 500, 400)
        edges2 = edges*roi
        lines = cv2.HoughLinesP(edges*roi,rho = 4,theta = np.pi/180,threshold = 5,minLineLength = 90,maxLineGap = 30)

       
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
        top = vanishing_point[1] + 60
        bottom = self.size[1]-35
        
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

        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        min_wid = 1000
        img = cv2.warpPerspective(self.image, self.M, self.UNWARPED_SIZE)
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
        self.pix_per_meter_x = min_wid/self.LANE_WIDTH
        if False :#self.camera.callibration_done :
            Lh = np.linalg.inv(np.matmul(self.M, self.camera.cam_matrix))
        else:
            Lh = np.linalg.inv(self.M)
        self.pix_per_meter_y = self.pix_per_meter_x * np.linalg.norm(Lh[:,0]) / np.linalg.norm(Lh[:,1])
        self.perspective_done_at =  datetime.utcnow().timestamp()
        if verbose : 
            cv2.imwrite("edges.jpg",edges2)         
            img_orig = cv2.polylines(self.image, [src_points.astype(np.int32)],True, (0,0,255), thickness=5)
            cv2.line(img, (int(x1), 0), (int(x1), self.UNWARPED_SIZE[1]), (255, 0, 0), 3)
            cv2.line(img, (int(x2), 0), (int(x2), self.UNWARPED_SIZE[1]), (0, 0, 255), 3)

            cv2.circle(img_orig,tuple(vanishing_point),10, color=(0,0,255), thickness=5)
            cv2.imwrite("POLY.jpg",img_orig)
            # cv2.imwrite("INPUT.jpg",img)
            # cv2.imshow(cv2.hconcat((img_orig, cv2.resize(img, img_orig.shape))))
        return
    
    def get_speed(self):
        return 30
    
    def detect_objects(self, image):
        boxes= self.yolo.make_predictions(image,obstructions = obstructions,plot=True) 
        image  =  cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        n_obs =  len(self.obstacles)
        for i in range(len(boxes)):
            tracker = cv2.TrackerKCF_create()# cv2.TrackerMIL_create()#  # Note: Try comparing KCF with MIL
            box = boxes[i]
            dst =  self.calculate_position(box)
            bbox = (box.xmin, box.ymin, box.xmax-box.xmin, box.ymax-box.ymin)
            print(bbox)
            success = tracker.init(image, bbox )
            if success :
                self.trackers.append(tracker)
                obstacle =  OBSTACLE(box, i+n_obs,  dst)
                self.obstacles.append(obstacle)
        if not self.first_detect :
            self.update_trackers(image)
        return

    def calculate_position(self, box: BoundBox):
        if (self.perspective_done_at > 0):
            pos = np.array((box.xmax/2+box.xmin/2, box.ymax)).reshape(1, 1, -1)
            dst = self.perspective_tfm(pos).reshape(2)
            dst =  np.array([dst[0]/self.pix_per_meter_x,(self.UNWARPED_SIZE[1]-dst[1])/self.pix_per_meter_y])
            return dst
        else:
            return np.array([0,0])
    

    @staticmethod
    def corwh2box(corwh):
        box=BoundBox( int(corwh[0]), int(corwh[1]), int(corwh[0] + corwh[2]), int(corwh[1] + corwh[3]))
        return box

    
    def update_trackers(self, image,plot = False):
        for n, tracker in enumerate(self.trackers):

            success, corwh = tracker.update(image)
            print("tracking", corwh ,  self.obstacles[n].xmin,self.obstacles[n].ymin,self.obstacles[n].xmax,self.obstacles[n].ymax)
            if not success :
                del self.obstacles[n]
                del self.trackers[n]
                continue
            box = self.corwh2box(corwh)
            dst = self.calculate_position( box)  
            self.obstacles[n].update_obstacle(box, dst, self.fps)
        if plot: 
            for i , box in enumerate(self.obstacles):
                past=box.position_hist[0]
                pos_lbl =  str(int(box.velocity[0]))+"," + str(int(box.velocity[1]))
                center =  (int(past[0]/2+past[2]/2), past[3])
                color = ORANGE if box.velocity[1] > 0  else GREEN
                cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), color, 1)
                cv2.circle(image,center,2, DARK_BLUE,2)
                cv2.circle(image,(int(box.xmin/2+box.xmax/2),box.ymax),2, color,2)
                cv2.putText(image, pos_lbl, 
                            (box.xmin+5, box.ymid),  cv2.FONT_HERSHEY_SIMPLEX, 
                            5e-4 * image.shape[0],   YELLOW, 1)
                cv2.putText(image, str(int(box.col_time))+"|" + str(int(box.position[1])), 
                            (box.xmid, box.ymin),  cv2.FONT_HERSHEY_SIMPLEX, 
                            5e-4 * image.shape[0], LIGHT_CYAN  , 1)
            cv2.imwrite(self.temp_dir+"updated.jpg", image)   
        return 

    


    def remove_tracked(self) :
        return
    
    def find_distance(self) : 
        
        return
    
    def map_to_previous(self) : 
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
    frame.detect_objects(cv2.imread("./images/"+files[0]))
    for i, f in enumerate(files[1:72]):
        frame.update_trackers(cv2.imread("./images/"+f),plot=True)
        # cv2.waitKey(1)
        input("Press Enter to continue...")
    
main()
