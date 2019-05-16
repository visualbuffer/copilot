from camera import CAMERA
from yolo import YOLO
import cv2
from datetime import datetime

yolo_detector =  YOLO(score =  0.3, iou =  0.5, gpu_num = 0)


class FRAME :


    _defaults = {
        "id": 0,
        "first": True,
        "speed": 0,
        "n_objects" :0,
        "camera" : CAMERA(),
        "image":None
        "UNWARPED_SIZE" : [ 500, 600],
        "WRAPPED_WIDTH" :  530
        "lane_width" :  3.5
        }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"  
        if not cls.image :
            return "No image !!"

    def __init__(self, **kwargs):
        self.UNWARPED_SIZEs :[float,float]
        self.camera : CAMERA
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.speed =  self.getSpeed()
        self.image =  self.camera.undistort(self.image)
        self.size =  self.image.shape
        self.M  = None
        self.pix_per_meter_x = 0
        self.pix_per_meter_y = 0
        self.perspective_done_at = 0

    def perspective_tfm(self ,  image) : 
        now  = datetime.utcnow()
        if now - self.perspective_done_at > 10000 :
            self.calc_perspective()
        return cv2.warpPerspective(image, self.M, self.UNWARPED_SIZEs)
  
    def calc_perspective(self, verbose =  False):
        Lhs = np.zeros((2,2), dtype= np.float32)
        Rhs = np.zeros((2,1), dtype= np.float32)
        img_hsl = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
        edges = cv2.Canny(img_hsl[:, :, 1], 200, 100)
        lines = cv2.HoughLinesP(edges*roi, 0.5, np.pi/180, 20, None, 180, 120)
        for line in lines:
            for x1, y1, x2, y2 in line:
                normal = np.array([[-(y2-y1)], [x2-x1]], dtype=np.float32)
                normal /=np.linalg.norm(normal)
                point = np.array([[x1],[y1]], dtype=np.float32)
                outer = np.matmul(normal, normal.T)
                Lhs += outer
                Rhs += np.matmul(outer, point)
                cv2.line(img, (x1,y1), (x2, y2),(255, 0, 0), thickness=2)
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

        dst_points = np.array([[0, 0], self.UNWARPED_SIZEs[0], 0],
                            [self.UNWARPED_SIZEs[0], self.UNWARPED_SIZEs[1]],
                            [0, self.UNWARPED_SIZEs[1]]], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        min_wid = 1000
        img = cv2.warpPerspective(self.image, self.M, self.UNWARPED_SIZEs)
        img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        mask = img_hsl[:,:,1]>128
        mask[:, :50]=0
        mask[:, -50:]=0
        mom = cv2.moments(mask[:,:self.UNWARPED_SIZE[0]//2].astype(np.uint8))
        x1 = mom["m10"]/mom["m00"]
        mom = cv2.moments(mask[:,self.UNWARPED_SIZE[0]//2:].astype(np.uint8))
        x2 = self.UNWARPED_SIZE[0]//2 + mom["m10"]/mom["m00"]

        if (x2-x1<min_wid):
            min_wid = x2-x1
        self.pix_per_meter_x = min_wid/(12* meter_per_foot)
        Lh = np.linalg.inv(np.matmul(M, self.camera.cam_matrix))
        self.pix_per_meter_y = self.pix_per_meter_x * np.linalg.norm(Lh[:,0]) / np.linalg.norm(Lh[:,1])
        if verbose :          
            img_orig = cv2.polylines(self.image, [src_points.astype(np.int32)],True, (0,0,255), thickness=5)
            cv2.line(img, (int(x1), 0), (int(x1), self.UNWARPED_SIZE[1]), (255, 0, 0), 3)
            cv2.line(img, (int(x2), 0), (int(x2), self.UNWARPED_SIZE[1]), (0, 0, 255), 3)
            cv2.imshow(cv2.hconcat((img_orig, img)))
        return
    
    def get_speed(self)
        return 30
    
    def detect_objects():
        return
    
    def find_distance() : 
        
        return
    
    def map_to_previous() : 
        return
    
    
    def vehicle_speed() :
        return

class DETECTION:
  def __init__(self , bbox, score, category,_id, image) :
    self.bbox = bbox
    self.score =  score
    self.category =  category
    self._id = _id
    self.image = image
    

class VEHICLE(DETECTION) :
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
    
  
    
    
class TRAFFIC_LIGHTS(DETECTION)
  def __init__(self) :
    return None
  
  def detect_status(self):
    return None
    
class TRAFFIC_SIGNS(DETECTION)
  def __init__(self) :
    return None
  
  def decipher(self):
    return None