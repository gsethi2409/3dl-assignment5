import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as bp

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # pass
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(3, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 128, 1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),

            torch.nn.Conv1d(128, 1024, 1),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),

            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),

            torch.nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        points = points.permute(0, 2, 1)
        feats = self.encoder(points)
        feats = torch.max(feats, 2, keepdim=True)[0]
        feats = feats.view(-1, 1024)
        feats = self.decoder(feats)
        return feats


# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, args, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # pass

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(3, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
        )

        self.globalfeatures_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),

            torch.nn.Conv1d(128, 1024, 1),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(1088, 512),
            torch.nn.Conv1d(1088, 512, 1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),

            # torch.nn.Linear(512, 256),
            torch.nn.Conv1d(512, 256, 1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),

            # torch.nn.Linear(256, 128),
            torch.nn.Conv1d(256, 128, 1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),

            # torch.nn.Linear(128, num_seg_classes)
            torch.nn.Conv1d(128, num_seg_classes, 1)
        )

        self.maxpool = torch.nn.MaxPool1d(args.num_points)

        self.num_seg_classes = num_seg_classes

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        num_points = points.shape[1]
        points = points.permute(0, 2, 1)
        encodedfeats = self.encoder(points)
        globalfeats = self.globalfeatures_encoder(encodedfeats)
        pooled_globalfeats = self.maxpool(globalfeats)
    
        pooled_globalfeats = pooled_globalfeats.expand(points.shape[0], 1024, num_points)
        
        pointfeats = torch.cat((encodedfeats, pooled_globalfeats), dim = 1)
        out = self.decoder(pointfeats)
        out = out.permute(0, 2, 1)
        return out

def knn(x, k):
    # TODO
    elemwise_sq = torch.sum(x**2, dim=1, keepdim=True)
    elemwise_sqT = elemwise_sq.transpose(2, 1)
    xT = x.transpose(2, 1)
    inner_prod = torch.matmul(xT, x)

    res = -elemwise_sq - (2 * inner_prod) - elemwise_sqT
    topk_preds = res.topk(k=k, dim=-1)[1]
    
    return topk_preds


def get_graph_feature(x, args, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=args.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    # TODO: check idx size and see if any reshaping is needed
    idx = idx.view(-1)

    # TODO: check x size and see if any reshaping is needed
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  

    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    # TODO: convert x = B x N x 1 x D to shape x = B x N x k x D (hint: repeating the elements in that dimension)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # feature = torch.cat((feature-x, x), dim=3)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = 20
        opdims = 1024
        dropout = 0.5

        # TODO: 4 Batch Norm 2D + 1 Batch Norm 1D
        # TODO: 5 conv2D layers + BN + ReLU/Leaky ReLU
        self.layer1 = nn.Sequential(
                            nn.Conv2d(6, 64, kernel_size=1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.LeakyReLU(negative_slope=0.2)
                            )

        self.layer2 = nn.Sequential(
                            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.LeakyReLU(negative_slope=0.2)
                            )
        
        self.layer3 = nn.Sequential(
                            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.LeakyReLU(negative_slope=0.2)
                            )

        self.layer4 = nn.Sequential(
                            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.LeakyReLU(negative_slope=0.2)
                            )

        self.layer5 = nn.Sequential(
                            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                            nn.BatchNorm1d(1024),
                            nn.LeakyReLU(negative_slope=0.2)
                            )


        # TODO: 2 Linear layers + BN + Dropout
        # TODO: 1 final Linear layer
        self.final_layer = nn.Sequential(
            nn.Linear(1024*2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Dropout(p=dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Dropout(p=dropout),

            nn.Linear(256, output_channels)

        )

    def forward(self, x):
        x = x.permute(0,2,1)
        batch_size = x.size(0)
        x = get_graph_feature(x, self.args, k=self.k)
        # TODO: conv
        x = self.layer1(x)
        # TODO: max -> x1
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, self.args, k=self.k)
        # TODO: conv
        x = self.layer2(x)
        # TODO: max -> x2
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, self.args, k=self.k)
        # TODO: conv
        x = self.layer3(x)
        # TODO: max -> x3
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, self.args, k=self.k)
        # TODO: conv
        x = self.layer4(x)
        # TODO: max -> x4
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = # TODO: concat all x1 to x4
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # TODO: conv
        x = self.layer5(x)

        # TODO: pooling
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # TODO: ReLU / Leaky ReLU
        x = self.final_layer(x)
        return x