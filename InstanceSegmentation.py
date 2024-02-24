# Instance model (Mask RCNN)

ins_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
def random_colour_masks(image):
    
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def SetId_on_masks(image,id):
    
    ids = np.zeros_like(image).astype(np.uint8)
    ids[image == 1] = id
    
    maskId = ids
    return maskId

def get_prediction(img, threshold):

    pred = ins_model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = 0
    for i in range(len(pred_score)):
        if pred_score[i] < 0.5:
            break
        else :
            pred_t += 1
    #pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    masks_score = pred[0]['masks'].squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    masks_id = pred[0]['labels'].numpy()
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t]
    masks_score = masks_score[:pred_t]
    pred_boxes = pred_boxes[:pred_t]
    pred_class = pred_class[:pred_t]
    pred_score = pred_score[:pred_t]
    masks_id = masks_id[:pred_t]
    return masks, pred_boxes, pred_class, masks_score, masks_id

import cv2
def instance_segmentation_api(img, threshold=0.5, rect_th=1, text_size=0.25, text_th=1):

    masks, boxes, pred_cls, masks_score, masks_id = get_prediction(img, threshold)
    mask = []
    mask = np.ndarray(mask)
    mask_score = np.zeros((224,224))
    mask_id = np.zeros((224,224))
    for i in range(len(masks)):        
        maskId = SetId_on_masks(masks[i],masks_id[i])
        if i > 0 and masks[i].ndim == 2:
            for j in range(224):
                for k in range(224):           
                    if masks_score[i][j][k] > 0.5 and masks_score[i][j][k] > mask_score[j][k]:
                        mask[j][k] = False
                        mask_score[j][k] = 0.0
                        mask_id[j][k] = 0

                    elif masks_score[i][j][k] > 0.5 and masks_score[i][j][k] < mask_score[j][k]:
                        masks[i][j][k] = False
                        masks_score[i][j][k] = 0.0
                        maskId[j][k] = 0
        if masks[i].ndim == 2:
            rgb_mask = random_colour_masks(masks[i])
            mask = cv2.add(mask,rgb_mask)
            mask_score = mask_score + masks_score[i]
            mask_id = mask_id + maskId

    return mask, mask_id
