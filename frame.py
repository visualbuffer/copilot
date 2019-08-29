from camera import CAMERA
from yolo_model import BoundBox,  YOLO 
from utils.bbox import bbox_iou 
from lane_detection import LANE_DETECTION, OBSTACLE,obstructions,create_queue, plt
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm


class FRAME :
    fps:float
    camera : CAMERA
    yolo : classmethod
    PERSP_PERIOD =  100000
    YOLO_PERIOD = 2 # SECONDS
    verbose = 3

    yellow_lower = np.uint8([ 20, 50,   50]),
    yellow_upper = np.uint8([35, 255, 255]),
    white_lower = np.uint8([ 0, 200,   0]),
    white_upper = np.uint8([180, 255, 100]), 
    lum_factor = 150,
    max_gap_th = 2/5,
    lane_start=[0.35,0.75]  

    ego_vehicle_offset = 0
    time =  datetime.utcnow().timestamp()
    l_gap_skipped = 0
    l_breached = 0 
    l_reset = 0 
    l_appended = 0

    
    n_gap_skipped = 0
    n_breached = 0 
    n_reset = 0 
    n_appended = 0
    _defaults = {
        "id": 0,
        "first": True,
        "speed": 0,
        "n_objects" :0,
         "camera" : CAMERA(),
        "image" : [],
        "LANE_WIDTH" :  3.66,
        "fps" :22,
        "ego_vehicle_offset" : 0,
        'verbose' :  3,
        'YOLO_PERIOD' : 2,
        "yellow_lower" : np.uint8([ 20, 50,   50]),
        "yellow_upper" : np.uint8([35, 255, 255]),
        "white_lower" : np.uint8([ 0, 200,   0]),
        "white_upper" : np.uint8([180, 255, 100]), 
        "lum_factor" : 150,
        "max_gap_th" : 2/5,
        "lane_start":[0.35,0.75] , 
        "verbose" : 3
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
        # if  self.image.size ==0 :
        #     raise ValueError("No Image") 
        self.img_shp :  (int, int)
        self.area : int
        
        self.temp_dir = './images/detection/'
        self.perspective_done_at = datetime.utcnow().timestamp()
       
        # self.image =  self.camera.undistort(self.image)
        ### OBJECT DETECTION AND TRACKING
        self.yolo =  YOLO()
        self.first_detect = True
        self.obstacles :[OBSTACLE] =[]
        self.__yp = int(self.YOLO_PERIOD*self.fps)
        ### LANE FINDER 
        self.count = 0
        self.lane :LANE_DETECTION = None



    def perspective_tfm(self ,  pos) : 
        now  = datetime.utcnow().timestamp()
        if now - self.perspective_done_at > self.PERSP_PERIOD :
            self.lane = LANE_DETECTION(self.image,self.fps,verbose=self.verbose)
        return cv2.perspectiveTransform(pos, self.lane.trans_mat)

    def determine_stats(self):
        n = 30
        t  = datetime.utcnow().timestamp()
        dt = int(t - self.time)
        if self.count % (self.fps * n) == 0:
            
            self.n_gap_skipped = int((self.lane.n_gap_skip - self.l_gap_skipped) *100 / (self.fps * n))
            self.n_appended = int((self.lane.lane.appended - self.l_appended) *100 / (self.fps * n))
            self.n_breached = int((self.lane.lane.breached - self.l_breached) *100 / (self.fps * n))
            self.n_reset = int((self.lane.lane.reset - self.l_reset) *100 / (self.fps * n))

           
            self.l_gap_skipped = self.lane.n_gap_skip 
            self.l_appended = self.lane.lane.appended 
            self.l_breached = self.lane.lane.breached
            self.l_reset = self.lane.lane.reset 
            print("SKIPPED {:d}% BREACHED {:d}% RESET {:d}% APPENDED {:d}% | Time {:d}s , Processing FPS {:.2f} vs Desired FPS {:.2f}  "\
                .format(self.n_gap_skipped, self.n_breached, self.n_reset, self.n_appended,\
                    dt, self.fps * n / dt, self.fps ))
            self.time=t
    def get_speed(self):
        return 30
    


    
    def process_and_plot(self,image):
        self.update_trackers(image)
        lane_img = self.lane.process_image( image, self.obstacles)
        self.determine_stats()
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
                tracker = cv2.TrackerKCF_create()# 
                # tracker = cv2.TrackerMIL_create()#  # Note: Try comparing KCF with MIL
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

    def process_video(self, file_path, fps_factor,\
            video_out = "videos/output11.mov",pers_frame_time =14,\
            t0  =None , t1 =None ):

        
        video_reader =  cv2.VideoCapture(file_path) 
        fps_actual =  video_reader.get(cv2.CAP_PROP_FPS)

        self.fps =  fps_actual//fps_factor
        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h_out =  int(frame_h*(1-self.ego_vehicle_offset))
        print("{:s} WIDTH {:d} HEIGHT {:d} FPS {:.2f} DUR {:.1f} s".format(\
            file_path,frame_w,frame_h,fps_actual,nb_frames//fps_actual
            ))

        video_writer = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),self.fps, (frame_w, frame_h_out))
        #180# 310# seconds
        pers_frame = int(pers_frame_time *fps_actual)
        video_reader.set(1,pers_frame)
        _, self.image = video_reader.read()
        self.image = self.image[:frame_h_out,:,:]
        self.img_shp =  (self.image.shape[1], self.image.shape[0] )
        # self.ego_vehicle_offset = self.img_shp[0]*int(1-self.ego_vehicle_offset)
        self.area =  self.img_shp[0]*self.img_shp[1]
        self.lane = LANE_DETECTION(self.image, self.fps,\
            verbose=self.verbose, 
            yellow_lower =self.yellow_lower,
            yellow_upper = self.yellow_upper,
            white_lower = self.white_lower,
            white_upper = self.white_upper, 
            lum_factor = self.lum_factor,
            max_gap_th = self.max_gap_th,
            lane_start=self.lane_start ,
        )
        t1  =  t1 if t1 is not None else nb_frames/fps_actual 
        t0 = t0 if t0 is not None else pers_frame_time
        video_reader.set(1,t0*fps_actual)
        for i in tqdm(range(int(t0*fps_actual), int(t1*fps_actual)),mininterval=3):
            status, image = video_reader.read()

            if  status and (i % fps_factor == 0 ) :
                image = image[:frame_h_out,:,:]
                try : 
                     procs_img = self.process_and_plot(image)
                     video_writer.write(procs_img) 
                except :
                     print("\n\rGOT EXEPTION TO PROCES THE IMAGE\033[F", self.count)
                     l1 =  self.lane.white_lower[1]
                     self.lane.compute_bounds(image)
                     print(l1,"->",self.lane.white_lower[1])
        print("SKIPPED {:d} BREACHED {:d} RESET {:d} APPENDED {:d} | Total {:d} ".\
            format(self.lane.n_gap_skip, self.lane.lane.breached,\
                self.lane.lane.reset,self.lane.lane.appended, self.count))
        print("SAVED TO ", video_out)
        video_reader.release()
        video_writer.release() 
        cv2.destroyAllWindows()


    
     
    
    
    def vehicle_speed(self) :
        return
        
if __name__ == "__main__":
    
    
    # file_path =  "videos/challenge_video.mp4"         # 145
    # file_path =  "videos/challenge_video_edit.mp4"    #145
    # file_path =  "videos/harder_challenge_video.mp4"  
    # file_path =  "videos/nice_road.mp4"               #110 62
    file_path =  "videos/us-highway.mp4"               #118 143
    # file_path =  "videos/nh60.mp4"                      # 118 18                   
    video_out = "videos/output11.mov"
    frame =  FRAME( 
        ego_vehicle_offset = .15,
        yellow_lower = np.uint8([ 20, 50,   100]),
        yellow_upper = np.uint8([35, 255, 255]),
        white_lower = np.uint8([ 0, 200,   0]),
        white_upper = np.uint8([180, 255, 100]), 
        lum_factor = 118,
        max_gap_th = 0.45,
        YOLO_PERIOD = .25,
        lane_start=[0.35,0.75] , 
        verbose = 3)
    frame.process_video(file_path, 1,\
            video_out = video_out,pers_frame_time =144,\
            t0  =144 , t1 =150)#None)
    


