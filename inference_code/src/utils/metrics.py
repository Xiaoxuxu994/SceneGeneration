import torchmetrics

class OccMetricEvaluator:
    def __init__(self, occ_threshold=0.5):
        self.occ_threshold = occ_threshold
        self.precision = torchmetrics.Precision(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.f1 = torchmetrics.F1Score(task='binary')
        self.iou = torchmetrics.JaccardIndex(task='binary', num_classes=2)

    def __call__(self, output_logit, label):
        output_logit = output_logit.detach().cpu()
        label = label.detach().cpu().int()

        pred = (output_logit >= self.occ_threshold).int()

        return (
            self.precision(pred, label),
            self.recall(pred, label),
            self.f1(pred, label),
            self.iou(pred, label)
        )