import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ClassificationMetric(object):
    def __init__(self, 
                metrics = [],
                softmax = True,
                force_binary = False):
            self.metrics = metrics
            self.metrics_dict = {
                "acc": accuracy_score,
                "pre": precision_score,
                "rec": recall_score,
                "auc": roc_auc_score,
                "f1": f1_score
            }
            self.keep_logits = {"auc"}
            assert all([_ in self.metrics_dict for _ in self.metrics])
            self.softmax = softmax
            self.force_binary = force_binary

    def preprocess(self, output):
        return output

    def __call__(self, outputs):
        y = torch.cat([_[0] for _ in outputs]).detach().cpu().numpy()
        yhat = torch.cat([_[1] for _ in outputs]).detach()
        if self.softmax:
            yhat = yhat.softmax(1).cpu().numpy()
        else:
            yhat = yhat.sigmoid().cpu().numpy()
        
        ret = {}
        for metric in self.metrics:
            if metric not in self.keep_logits:
                yh = yhat.argmax(1) if self.softmax else yhat.round()
            else:
                yh = yhat[:,1] if self.softmax and self.force_binary else yhat
            val = self.metrics_dict[metric](y, yh)
            ret["val_" + metric] = val

        return ret

