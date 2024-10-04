import glob
import cv2
import yaml

def threshold(img, params):
    return cv2.threshold(img, params['threshold'], 255, cv2.THRESH_BINARY)[1]


class SwabDetector:
    def __init__(self,path='images/*.jpg'):
        self.imgs=glob.glob(path)
        self.processed_imgs = {}
        self.window_name = 'image'
        self.config_file = 'config.yaml'
        self.config = {}

        self.current_img = 0
        self.curr_params = {}
    
        assert len(self.imgs) > 0, "No images found in the images directory"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.load_values()

        self.pipelines = []

        self.init_trackbars()

    def load_values(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f) 

        if self.config is None:
            self.config = {}

    def init_trackbars(self):
        self.add_trackbar('img_idx', (0, len(self.imgs)-1), 0)

        for k, v in self.config.items():
            self.add_trackbar(k, (v['min'],v['max']), v['value'])

    def add_trackbar(self, trackbar_name, rng, def_val):
        cv2.createTrackbar(trackbar_name,self.window_name, rng[0], rng[1], lambda x: None)
        cv2.setTrackbarPos(trackbar_name,self.window_name, def_val)

    def update_value(self, trackbar_name, val):
        self.config.update(trackbar_name, val)
        # save values to config file
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)

    
    def run(self):
        while True:
            img_path = self.imgs[self.current_img]
            img = cv2.imread(img_path)


            for p in self.pipelines:
                img = p(img, self.config)

            cv2.imshow(self.window_name, img)
    
            current_input = cv2.waitKey(1) & 0xFF
            if current_input == ord('q'):
                break

            elif current_input == ord(']'):
                self.current_img +=1

            elif current_input == ord('['):
                self.current_img -=1

            if self.current_img < 0:
                self.current_img = len(self.imgs) - 1
            elif self.current_img >= len(self.imgs):
                self.current_img = 0


if __name__ == '__main__':
    sd = SwabDetector()
    sd.pipelines.append(threshold)
    sd.run()

