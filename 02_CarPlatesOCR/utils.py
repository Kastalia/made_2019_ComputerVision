import logging
import torch
import cv2
import numpy as np


def get_logger(filename: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def dice_coeff(input, target):
    smooth = 1.

    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()
    union = input_flat.sum() + target_flat.sum()

    return (2. * intersection + smooth) / (union + smooth)


def dice_loss(input, target):
    return - torch.log(dice_coeff(input, target))


def get_boxes_from_mask(mask, margin, clip=False):
    """
    Detect connected components on mask, calculate their bounding boxes (with margin) and return them (normalized).
    If clip is True, cutoff the values to (0.0, 1.0).
    :return np.ndarray boxes shaped (N, 4)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for j in range(1, num_labels):  # j = 0 == background component
        x, y, w, h = stats[j][:4]
        x1 = int(x - margin * w)
        y1 = int(y - margin * h)
        x2 = int(x + w + margin * w)
        y2 = int(y + h + margin * h)
        box = np.asarray([x1, y1, x2, y2])
        boxes.append(box)
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes).astype(np.float)
    boxes[:, [0, 2]] /= mask.shape[1]
    boxes[:, [1, 3]] /= mask.shape[0]
    if clip:
        boxes = boxes.clip(0.0, 1.0)
    return boxes


def prepare_for_inference(image, fit_size):
    """
    Scale proportionally image into fit_size and pad with zeroes to fit_size
    :return: np.ndarray image_padded shaped (*fit_size, 3), float k (scaling coef), float dw (x pad), dh (y pad)
    """
    # pretty much the same code as detection.transforms.Resize
    h, w = image.shape[:2]
    k = fit_size[0] / max(w, h)
    image_fitted = cv2.resize(image, dsize=None, fx=k, fy=k)
    h_, w_ = image_fitted.shape[:2]
    dw = (fit_size[0] - w_) // 2
    dh = (fit_size[1] - h_) // 2
    image_padded = cv2.copyMakeBorder(image_fitted, top=dh, bottom=dh, left=dw, right=dw,
                                      borderType=cv2.BORDER_CONSTANT, value=0.0)
    if image_padded.shape[0] != fit_size[1] or image_padded.shape[1] != fit_size[0]:
        image_padded = cv2.resize(image_padded, dsize=fit_size)
    return image_padded, k, dw, dh


def get_boxes_from_mask(mask, margin, clip=False):
    """
    Detect connected components on mask, calculate their bounding boxes (with margin) and return them (normalized).
    If clip is True, cutoff the values to (0.0, 1.0).
    :return np.ndarray boxes shaped (N, 4)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for j in range(1, num_labels):  # j = 0 == background component
        x, y, w, h = stats[j][:4]
        x1 = int(x - margin * w)
        y1 = int(y - margin * h)
        x2 = int(x + w + margin * w)
        y2 = int(y + h + margin * h)
        box = np.asarray([x1, y1, x2, y2])
        boxes.append(box)
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes).astype(np.float)
    boxes[:, [0, 2]] /= mask.shape[1]
    boxes[:, [1, 3]] /= mask.shape[0]
    if clip:
        boxes = boxes.clip(0.0, 1.0)
    return boxes

# Бинарный поиск для приближения предсказанной маски 4-хугольником
def _simplify_contour(contour, n_corners=4):
    n_iter, max_iter = 0, 1000
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            print('simplify_contour didnt coverege')
            return None

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx

def _order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    
    pts = pts[pts[:,0].argsort()]
    
    if(pts[0][1]<pts[1][1]):
        rect[0] = pts[0]
        rect[3] = pts[1]
    else:
        rect[0] = pts[1]
        rect[3] = pts[0]
        
    if(pts[2][1]<pts[3][1]):
        rect[1] = pts[2]
        rect[2] = pts[3]
    else:
        rect[1] = pts[3]
        rect[2] = pts[2]

    #s = pts.sum(axis = 1)   
    #rect[0] = pts[np.argmin(s)]
    #rect[2] = pts[np.argmax(s)]
    
    #pts = np.delete(pts, [np.argmin(s),np.argmax(s)], 0)
    #if(pts[0][0]<pts[1][0]):
    #    rect[1] = pts[1]
    #    rect[3] = pts[0]
    #else:
    #    rect[1] = pts[0]
    #    rect[3] = pts[1]
        
    return rect

# Отображаем 4-хугольник в прямоугольник
# Thanks за реализацию: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def transformPlate(image, pts):    
    rect = _order_points(pts)
    
    #tl, tr, br, bl = pts
    tl, tr, br, bl = rect
    
    width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_1), int(width_2))
    
    height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_1), int(height_2))
    
    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped
