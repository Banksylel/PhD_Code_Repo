import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import numpy as np
import torch.nn.functional as F



def split_targets(t, num_classes, pos_class_val=255):
    out = []
    for i in range(num_classes):
        out.append(torch.where(t == i, pos_class_val, 0))

    return(out)

def process_results(target, pred):
    # convert target list into a tensor of shape (batch, width, height)
    target = torch.stack(target)

    # traverse the second dimension of pred and reshape to (batch, width, height) where the largest value of the 3 channels is the class
    pred = pred.argmax(dim=1)

    return(target, pred)



# TODO: change both to multi calss metrics
class DiceScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        # self.smooth = smooth

    def forward(self, logits, targets, device, num_classes):
        # gets logits probabilities
        probs = F.softmax(logits, dim = 1)
        # reformats the logits and targets to match shape and class contents.
        currentTarget, currentProbs  = process_results(targets, probs)

        # target and logits needs to be the same shape and contain the most likley classs for each pixel. calculates the dice score for each class
        metric1 = MulticlassF1Score(num_classes=num_classes).to(device)
        metric2 = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
        return ([metric1(currentProbs, currentTarget), metric2(currentProbs, currentTarget)])




class Jaccardindex(torch.nn.Module):
    def __init__(self, smooth=1):
        super(Jaccardindex, self).__init__()
        # self.smooth = smooth

    def forward(self, logits, targets, device, num_classes):
        # gets logits probabilities
        probs = F.softmax(logits, dim = 1)
        # reformats the logits and targets to match shape and class contents.
        currentTarget, currentProbs  = process_results(targets, probs)

        # target and logits needs to be the same shape and contain the most likley classs for each pixel. calculates the jaccard index for each class
        metric1 = MulticlassJaccardIndex(num_classes=num_classes).to(device)
        metric2 = MulticlassJaccardIndex(num_classes=num_classes, average=None).to(device)
        return ([metric1(currentProbs, currentTarget), metric2(currentProbs, currentTarget)])
    


class Accuracy(torch.nn.Module):
    def __init__(self, smooth=1):
        super(Accuracy, self).__init__()
        # self.smooth = smooth

    def forward(self, logits, targets, device, num_classes):
        # gets logits probabilities
        probs = F.softmax(logits, dim = 1)
        # reformats the logits and targets to match shape and class contents.
        currentTarget, currentProbs  = process_results(targets, probs)

        # target and logits needs to be the same shape and contain the most likley classs for each pixel. calculates the accuracy for each class
        metric1 = MulticlassAccuracy(num_classes=num_classes).to(device)
        metric2 = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
        return ([metric1(currentProbs, currentTarget), metric2(currentProbs, currentTarget)])



class Precision(torch.nn.Module):
    def __init__(self, smooth=1):
        super(Precision, self).__init__()
        # self.smooth = smooth

    def forward(self, logits, targets, device, num_classes):
        # gets logits probabilities
        probs = F.softmax(logits, dim = 1)
        # reformats the logits and targets to match shape and class contents.
        currentTarget, currentProbs  = process_results(targets, probs)

        # target and logits needs to be the same shape and contain the most likley classs for each pixel. calculates the precision for each class
        metric1 = MulticlassPrecision(num_classes=num_classes).to(device)
        metric2 = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
        return ([metric1(currentProbs, currentTarget), metric2(currentProbs, currentTarget)])


class Recall(torch.nn.Module):
    def __init__(self, smooth=1):
        super(Recall, self).__init__()
        # self.smooth = smooth

    def forward(self, logits, targets, device, num_classes):
        # gets logits probabilities
        probs = F.softmax(logits, dim = 1)
        # reformats the logits and targets to match shape and class contents.
        currentTarget, currentProbs  = process_results(targets, probs)

        # target and logits needs to be the same shape and contain the most likley classs for each pixel. calculates the Recall for each class
        metric1 = MulticlassRecall(num_classes=num_classes).to(device)
        metric2 = MulticlassRecall(num_classes=num_classes, average=None).to(device)
        return ([metric1(currentProbs, currentTarget), metric2(currentProbs, currentTarget)])