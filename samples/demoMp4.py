import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Process,Queue

# import coco
from coco import coco
# import utils
from mrcnn import utils
from mrcnn import model as modellib
 
import cv2
import colorsys
 
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

IMAGES_BATCH = 21

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    # IMAGES_PER_GPU = 1
    IMAGES_PER_GPU = IMAGES_BATCH
 
config = InferenceConfig()
 
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
 
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
 
cap = cv2.VideoCapture("test.mp4")
#cap = cv2.VideoCapture(0)

# height = 720
# width  = 1280 
# height = 800
# width  = 800
# height = 360
# width  = 640
# height = 180
# width  = 320
height = 600
width  = 600


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
 
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
 
def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
 
    colors = random_colors(N)
 
    masked_image = image.copy()
    for i in range(N):
        color = colors[i]
 
        # Bounding box
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        camera_color = (color[0] * 255, color[1] * 255, color[2] * 255)
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), camera_color , 1)
 
        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        camera_font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(masked_image,caption,(x1, y1),camera_font, 1, camera_color)
 
        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)
 
    return masked_image.astype(np.uint8)

def putImages(q):
    while(True):
        for i in range(IMAGES_BATCH):
                # 動画ストリームからフレームを取得
                ret, frame = cap.read()
                # カメラ画像をリサイズ
                image_cv2 = cv2.resize(frame,(width,height))
                if i == 0:
                    images = np.array([image_cv2])
                else:
                    images = np.append(images,[image_cv2],axis=0)
        q.put(images)

def main():
    # FPS 測定
    tm = cv2.TickMeter()
    tm.start()
    count = 0
    max_count = IMAGES_BATCH * 2
    fps = 0
    
    while(True):
        
        # for i in range(IMAGES_BATCH):
        #     # 動画ストリームからフレームを取得
        #     ret, frame = cap.read()
        #     # カメラ画像をリサイズ
        #     image_cv2 = cv2.resize(frame,(width,height))
        #     if i == 0:
        #         images = np.array([image_cv2])
        #     else:
        #         images = np.append(images,[image_cv2],axis=0)

        Q = Queue()
        P = Process(target=putImages,args=(Q,))
        P.start()
        images = Q.get()
        print(len(images))
        P.join()

        tm.reset()
        tm.start()
        # t1 = tm.getTimeSec()
        results = model.detect(images)

        # FPS 測定
        # if count == max_count:
        #     tm.stop()
        #     fps = max_count / tm.getTimeSec()
        #     tm.reset()
        #     tm.start()
        #     count = 0
        #     print('fps: {:.2f}'.format(fps))
        tm.stop()
        # t2 = tm.getTimeSec()
        # print(t2-t1)
        fps = len(results) / tm.getTimeSec()
        # tm.reset()
        # tm.start()
        count = 0
        print('fps: {:.2f}'.format(fps))

        cv2.putText(frame, 'FPS: {:.2f}'.format(fps),(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),thickness=2)        

        for i in range(IMAGES_BATCH):
            r = results[i]
            camera = display_instances(images[i], r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
            cv2.imshow("camera window", camera) 
            count += 1
            # print(i)
            # escを押したら終了。
            if cv2.waitKey(1) == 27:
                break
    
    #終了
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == '__main__':
    main()