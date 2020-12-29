import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *

def pre_process(image, imgsz):
    img = [letterbox(x, new_shape=imgsz)[0] for x in image]

    # Stack
    img = np.stack(img, 0)

    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)

    return img

def load_model(weights='weights/best3.pt', imgsz=416, device=''):
    # Initialize
    device = torch_utils.select_device(device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # load to FP32
    if half:
        model.half()  # to FP16

    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    return model, names, colors, device, half

def detect(model, names, colors, device, half, img, img0, cam, max_person, output='inference/output', conf_thres=0.4, iou_thres=0.5, augment=False, classes=[], agnostic_nms=False):
    # Set Dataloader
    view_img = True

    #write_list = []
    black_list = ['cat', 'dog', 'person', 'horse','car']
    
    dataset = [(cam, img, img0)]

    detections = list()

    # Run inference
    f = open('times_cpu', 'a')
    t0 = time.time()
    for path, img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if max_person == '-1':
                        print("APPend", names[int(cls)])
                        if names[int(cls)] in black_list:
                            
                            detections.append([[int(i.item()) for i in xyxy], conf.item(), cls.item()])
                    else:
                        if cls.item() == 0:
                            detections.append([[int(i.item()) for i in xyxy], conf.item(), cls.item()])
                        if view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            f.write('{}\n'.format(t2-t1))
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     if cv2.waitKey(1) == ord('q'):  # q to quit
            #         raise StopIteration
    
    print('Done. (%.3fs)' % (time.time() - t0))

    return detections