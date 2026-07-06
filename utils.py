import numpy as np
from PIL import Image

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    def _compute_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, ground_truths):
        """Accumulate a batch of predictions and ground truths."""
        if predictions.ndim == 2:
            predictions = predictions[np.newaxis]
            ground_truths = ground_truths[np.newaxis]
        for pred, gt in zip(predictions, ground_truths):
            self.hist += self._compute_hist(pred.flatten(), gt.flatten())

    def evaluate(self):
        """Returns (per_class_iou, mean_iou)."""
        intersection = np.diag(self.hist)
        union = self.hist.sum(axis=1) + self.hist.sum(axis=0) - intersection
        iou = np.where(union > 0, intersection / union, 0.0)
        miou = np.mean(iou[union > 0]) if np.any(union > 0) else 0.0
        return iou, miou

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def compute_miou(self, predictions, gts):
        # Initialize histogram for the batch
        hist = np.zeros((self.num_classes, self.num_classes))

        # Accumulate histogram for all samples in the batch
        for pred, gt in zip(predictions, gts):
            hist += self._compute_hist(pred.flatten(), gt.flatten())

        # Compute intersection and union
        intersection = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - intersection

        # Compute per-class IoU, handling divide-by-zero by setting those IoUs to zero
        iou = np.where(union > 0, intersection / union, 0)
        
        # Return per-class IoU and mean IoU (ignoring classes with union=0)
        return iou, np.mean(iou[union > 0])


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap

