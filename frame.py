from camera import CAMERA
from yolo_model import BoundBox,  YOLO 
from utils.bbox import bbox_iou 
from lane_detection import LANE_DETECTION, OBSTACLE,obstructions,create_queue, plt
# from lane_finder import LANE_DETECTION
import numpy as np
import cv2
from datetime import datetime
# from PIL import Image
# from matplotlib import pyplot as plt
# yolo_detector =  YOLO(score =  0.3, iou =  0.5, gpu_num = 0)




        
        
        
  
    
    
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
    YOLO_PERIOD = 1 # SECONDS
    _defaults = {
        "id": 0,
        "first": True,
        "speed": 0,
        "n_objects" :0,
         "camera" : CAMERA(),
        "image" : [],
      
        "LANE_WIDTH" :  3.66,
        "fps" :22,
        'verbose' :  True
        }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"  

    def __init__(self, **kwargs):
        # calc pers => detect cars and dist > detect lanes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.speed =  self.get_speed()
        ### IMAGE PROPERTIES
        self.image : np.ndarray
        if  self.image.size ==0 :
            raise ValueError("No Image") 
        self.font_sz = 4e-4 * self.image.shape[0]
        self.lane = LANE_DETECTION(self.image, self.fps,verbose=self.verbose)
        self.temp_dir = './images/detection/'
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
        self.count = 0


    def perspective_tfm(self ,  pos) : 
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > self.PERSP_PERIOD :
            self.lane = LANE_DETECTION(self.image,self.fps)
        return cv2.perspectiveTransform(pos, self.lane.trans_mat)

  
    
    def get_speed(self):
        return 30
    


    
    def process_and_plot(self,image):
        self.update_trackers(image)
        lane_img = self.lane.process_image( image, self.obstacles)
        return lane_img

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
        idmax = 0
        obstacles =[]
        while count >0 :
            r,k  = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
            if iou_mat[r,k] > th :
                used.append(k)
                obstacle  = self.obstacles[r]
                box = boxes[k]
                if idmax < obstacle._id :
                    idmax = obstacle._id 
                obstacle.update_box(box)
                obstacles.append(obstacle)
            iou_mat[r,:] =  -99
            iou_mat[:,k] =  -99
            count = count -1
        idx = range(n_b)
        idx =  [elem for elem in idx if elem not in used]
        self.obstacles = obstacles
        for i, c in enumerate(idx):
            # dst  =  self.calculate_position(boxes[c])
            obstacle = OBSTACLE(boxes[c],i+idmax+1)
            self.obstacles.append(obstacle)
        return
    
    def update_trackers(self, img):
        image = img.copy()
        for n, obs in enumerate(self.obstacles):

            success, corwh = obs.tracker.update(image)
            # print("tracking", corwh ,  self.obstacles[n].xmin,self.obstacles[n].ymin,self.obstacles[n].xmax,self.obstacles[n].ymax)
            if not success :
                del self.obstacles[n]

                continue
            box = self.corwh2box(corwh)
            # dst = self.calculate_position( box)  
            self.obstacles[n].update_coord(box)
        
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

           
        return



    def warp(self, img):
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > self.PERSP_PERIOD :
            self.lane = LANE_DETECTION(self.image,self.fps)
        return cv2.warpPerspective(img, self.lane.trans_mat, self.lane.UNWARPED_SIZE, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)

    def unwarp(self, img):
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > self.PERSP_PERIOD :
            self.lane = LANE_DETECTION(self.image,self.fps)
        return cv2.warpPerspective(img, self.lane.trans_mat, self.img_shp, flags=cv2.WARP_FILL_OUTLIERS +
                                                                     cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)



    
     
    
    
    def vehicle_speed(self) :
        return
        
if __name__ == "__main__":
    from tqdm import tqdm
    # video_reader =  cv2.VideoCapture("videos/harder_challenge_video.mp4") 
    # video_reader =  cv2.VideoCapture("videos/challenge_video.mp4") 
    video_reader =  cv2.VideoCapture("videos/nice_road.mp4") 
    fps =  video_reader.get(cv2.CAP_PROP_FPS)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_out = "videos/output10.mov"
    # cv2.VideoWriter_fourcc(*'MPEG')
    video_writer = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_w, frame_h))
    pers_frame_time = 180# seconds
    pers_frame = int(pers_frame_time *fps)
    video_reader.set(1,pers_frame)
    ret, image = video_reader.read()
    frame = FRAME(image=image, fps =  fps, verbose =  True)
    frames = nb_frames
    t0  = 180 #0  # sec
    t1 =  2000 # frames/fps #sec
    dur = t1 -t0
    video_reader.set(1,t0*fps)
    start = datetime.utcnow().timestamp()
    for i in tqdm(range(int(t0*fps), int(t1*fps))):
        status, image = video_reader.read()
        if  status :
            try : 
                procs_img = frame.process_and_plot(image)
                video_writer.write(procs_img) 
            except :
                print("TGO EXEPTION TO PROCES THE IMAGE")
    stop =datetime.utcnow().timestamp()
    print(stop - start, "[s] Processing time for ", dur, " [s] at ", fps, " FPS")
    lh = frame.lane.left_line_history
    rh = frame.lane.right_line_history
    print(lh.reset, lh.breached, lh.appended)
    print(rh.reset, rh.breached, rh.appended)
    print(frame.lane.ndirect,frame.lane.nskipped, frame.lane.count)
    video_reader.release()
    video_writer.release() 
    cv2.destroyAllWindows()


