
import glob
import cv2
from object_detector import YOLO

def convert(size, box):
    dw = 1./size[0] 
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

object_detector = YOLO()

for img_path in glob.glob("img4auto/*.jpg"):
    filename = img_path.split('/')[-1][:-4]   
    path = "".join(img_path.split('/')[:-1])
    print(filename, path)
    img = cv2.imread(img_path)   
    img = cv2.resize(img,(416,416))
    f = open('{}/{}.txt'.format(path,filename),'w')
    detections = object_detector.detect_object(img, '12', '1')
    print('@@@@@@@@@', detections)
    for det in detections:
        print('@@@@',det)
        print(img.shape)
        classe = int(det[2])
        ymin, xmin, ymax , xmax = det[0]
        box = (ymin, ymax, xmin, xmax)
        h,w,c = img.shape
        bb = convert((w,h), box)
        x ,y , w, h = bb
        #x ,y , w, h = bb
        print('#####', box)
        print('## YOLO ###', bb)
        #cv2.rectangle(img, (int(ymin),int(xmin)), (int(ymin+(ymax-ymin)),int(xmin+(xmax-xmin))), (0, 255, 255), 2) 
        #cv2.imshow('img', img)
        #cv2.waitKey(0)             

        if classe == 0:
            classe = 1
            a = '{} {} {} {} {}\n'.format(classe, x ,y , w, h)
            f.write(str(a))
    f.close()
        
        

'''


a= Label_xml()
a.create_XML()
a.create_objects()
dic = a.pred("./best_77.jpg")
a.get_boxes(dic)'''