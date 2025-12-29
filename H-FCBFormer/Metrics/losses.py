import torch
import torch.nn as nn
import torch.nn.functional as F
from tree_util import getTreeList
import math
import segmentation_models_pytorch as smp
import os
import torchvision
from Metrics import performance_metrics

    

# Herarchical loss functions

# NEW: soft dice loss with hierarchal loss
class TreeSoftDiceLoss(nn.Module):
    def __init__(self, smooth=1, num_classes=3):
        super(TreeSoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.diceloss = smp.losses.DiceLoss(mode='multiclass', classes=num_classes, log_loss=False, from_logits=False, smooth=self.smooth) # , eps=1e-07


    def forward(self, logits, targets, root):

        # subtract max logit to prevent overflow and underflow 
        model_probabilities = F.log_softmax(logits, dim = 1)
        loss = 0.0 # 0 []
        loss2 = []

        # turns target tensor [(352, 352)] to tensor of shape (5, 352, 352)
        targets = torch.stack(targets)

        precomputed_hierarchy_list = getTreeList(root) # see tree_util.py


        # for batch in range(len(targets)):
        for level_loss_list in precomputed_hierarchy_list:

            probabilities_tosum = model_probabilities.clone()
            summed_probabilities = probabilities_tosum
            for branch in level_loss_list:

                # Extract the relevant probabilities according to a branch in our hierarchy.
                branch_probs = torch.cuda.FloatTensor()
                for channel in branch:
                    branch_probs = torch.cat((branch_probs,probabilities_tosum[:,channel,:,:].unsqueeze(1)),1)

                # Sum these probabilities into a single slice; this is hierarchical inference. Sums the probs of  
                summed_tree_branch_slice = branch_probs.sum(1,keepdim=True)

                # Insert inferred probability slice into each channel of summed_probabilities given by branch.
                # This duplicates probabilities for easy passing to standard loss functions such as nll_loss.
                for channel in branch:  
                    summed_probabilities[:,channel:(channel+1),:,:] = summed_tree_branch_slice

            # changes 0 probabilites to 0.0000001 and 1 probabilities to 0.9999999 and leaves all other probabilities unchanged. prevents deviding by 0 while also preventing overflow if you just sum by a small constant
            summed_probabilities = torch.clamp(summed_probabilities, 0.0000001, 0.9999999) # log sum exponent trick
            # summed_probabilities + 0.0001, allows for 0.0 probabilities to be passed to nll_loss
            level_loss = self.diceloss((summed_probabilities), targets)
            # level_loss = self.dice_loss((summed_probabilities + 0.0001), targets, num)
            loss2.append(level_loss)

            loss = loss + level_loss
        print('dice loss: ', loss.item())
        return(loss, loss2)



# NEW: BCE loss with hierarchal loss
class TreeCrossEntropyLoss(nn.Module):
    def __init__(self, smooth=1, floss_alpha=0.25, floss_gamma=2):
        super(TreeCrossEntropyLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, root):
        # subtract max logit to prevent overflow and underflow 
        logits = logits - logits.max()
        model_probabilities = F.softmax(logits, dim = 1)
        loss = 0.0 #0 []
        loss2 = []

        # turns target tensor [(352, 352)] to tensor of shape (5, 352, 352)
        targets = torch.stack(targets)

        precomputed_hierarchy_list = getTreeList(root) # see tree_util.py

        for level_loss_list in precomputed_hierarchy_list:

            probabilities_tosum = model_probabilities.clone()
            summed_probabilities = probabilities_tosum
            for branch in level_loss_list:

                # Extract the relevant probabilities according to a branch in our hierarchy.
                branch_probs = torch.cuda.FloatTensor()
                for channel in branch:
                    branch_probs = torch.cat((branch_probs,probabilities_tosum[:,channel,:,:].unsqueeze(1)),1)

                # Sum these probabilities into a single slice; this is hierarchical inference.
                summed_tree_branch_slice = branch_probs.sum(1,keepdim=True)

                # Insert inferred probability slice into each channel of summed_probabilities given by branch.
                # This duplicates probabilities for easy passing to standard loss functions such as nll_loss.
                for channel in branch:  
                    summed_probabilities[:,channel:(channel+1),:,:] = summed_tree_branch_slice

            # changes 0 probabilites to 0.0000001 and 1 probabilities to 0.9999999 and leaves all other probabilities unchanged
            summed_probabilities = torch.log(torch.clamp(summed_probabilities, 0.0000001, 0.9999999)) # log softmax
            
            level_loss = F.nll_loss(summed_probabilities, targets) #  , weight=class_weight
            loss2.append(level_loss)

            loss = loss + level_loss
        print('CE loss: ', loss.item())
        return(loss, loss2)









# non-hierarchal loss functions

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1, num_classes=3):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.diceloss = smp.losses.DiceLoss(mode='multiclass', classes=num_classes, log_loss=False, from_logits=True, smooth=self.smooth) # , eps=1e-07

    def forward(self, logits, targets):

        targets = torch.stack(targets)
        loss = self.diceloss((logits), targets)

        return loss



class CrossEntropyLoss(nn.Module):
    def __init__(self, smooth=1):
        super(CrossEntropyLoss, self).__init__()
        self.smooth = smooth
        self.Loss = nn.CrossEntropyLoss(label_smoothing=self.smooth)
        

    def forward(self, logits, targets):

        # turns target tensor [(352, 352)] to tensor of shape (5, 352, 352)
        targets = torch.stack(targets)
        loss = self.Loss(logits, targets)
        return loss

