import torch as tc
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

NSGAN = 'nsgan'
LSGAN = 'lsgan'
HINGE = 'hinge'


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type=NSGAN, target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', tc.tensor(target_real_label))
        self.register_buffer('fake_label', tc.tensor(target_fake_label))

        if type == NSGAN:
            self.criterion = nn.BCELoss()
        elif type == LSGAN:
            self.criterion = nn.MSELoss()
        elif type == HINGE:
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == HINGE:
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            loss = self.criterion(outputs, labels)
            return loss


class AdversarialMSELoss(nn.Module):
    def __init__(self):
        super(AdversarialMSELoss, self).__init__()
        self.loss = tc.nn.MSELoss()
        return
    def __call__(self, output, is_real, device):
        #TODO: calculate bce loss between input & output.
        #INPUT: output is a vector of size b (batch), is_real indicate output from G or D
        b = len(output)
        target = tc.ones(b) if is_real == True else tc.zeros(b)
        target = target.to(device)
        
        output = output.to(device)
        return self.loss(output, target)