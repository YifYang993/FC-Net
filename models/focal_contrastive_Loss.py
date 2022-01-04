import torch
import torch.nn as nn
import torch.nn.functional as F


class Focal_ContrastiveLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, margin=2, scale=16):
        super(Focal_ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
        self.gamma = gamma
        self.EPS = 1e-12
        self.scale = scale

    def forward(self, output1, output2, label):

        euclidean_distance = F.pairwise_distance(output1,
                                                 output2,
                                                 keepdim=True)
        euclidean_distance = euclidean_distance.squeeze(1)

        ########group similar sample and dissimilar sample###########
        sim_distance = euclidean_distance[label == 1]
        dis_sim_distance = euclidean_distance[label == 0]

        ########hard mining###########
        if len(sim_distance) > 0:  #Check if similar samples exist.
            mine_dis_sim = dis_sim_distance[dis_sim_distance -
                                            0.4 < max(sim_distance)]
        else:
            mine_dis_sim = dis_sim_distance
        ##############################

        mine_sim = sim_distance  #We only perform hard mining on different pairs because there are too few similar samples

        pt_sim = self.sigmoid(mine_sim)  #Focal factor for similar pair
        pt_dissim = self.sigmoid(
            mine_dis_sim)  #Focal factor for dissimilar pair

        mine_dis_sim_ = torch.pow(
            torch.clamp(self.margin - mine_dis_sim, min=0.0), 2)
        mine_sim_ = torch.pow(mine_sim, 2)

        dis_sim_final = (self.alpha) * self.scale * mine_dis_sim_ * torch.pow(
            (1 - pt_dissim), self.gamma
        )  #Multiply by balance factor and scale full value for similar pairs
        sim_final = (1 - self.alpha) * self.scale * mine_sim_ * torch.pow(
            (pt_sim + self.EPS), self.gamma
        )  #Multiply by balance factor and scale full value for dissimilar pairs

        focal_contrastive_loss = (torch.sum(dis_sim_final) + torch.sum(sim_final)
                            ) / (int(mine_sim.size(0) + mine_dis_sim.size(0)))

        return focal_contrastive_loss
