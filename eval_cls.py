import numpy as np
import argparse

import torch
from models import cls_model, DGCNN
from utils import create_dir, viz_cls

from data_loader import get_eval_data_loader

from pdb import set_trace as bp


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--rot_points_x', type=int, default=90, help='Rotate points by the degrees mentioned about X-axis')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model1 = cls_model().to(args.device)
    model2 = DGCNN(args).to(args.device)
    
    # Load Model Checkpoint
    model1_path = './checkpoints/q1cls/{}.pt'.format(args.load_checkpoint)
    model2_path = './checkpoints/cls/model_epoch_40.pt'

    with open(model1_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model1.load_state_dict(state_dict)
    model1.eval()
    print ("successfully loaded checkpoint from {}".format(model1_path))

    with open(model2_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model2.load_state_dict(state_dict)
    model2.eval()
    print ("successfully loaded checkpoint from {}".format(model2_path))


    # # Sample Points per Object -- Moved to Dataloader
    # ind = np.random.choice(10000, args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy(np.load(args.test_label))

    correct_point = 0
    num_point = 0
    # Test Dataloader
    test_dataloader = get_eval_data_loader(args=args)
    j=0
    # Rot in x by 90
    R90 = torch.Tensor([[1,0,0], [0,0,-1], [0,1,0]]).to(args.device)
    # Rot in x by 180
    R180 = torch.Tensor([[1,0,0], [0,-1,0], [0,0,-1]]).to(args.device)

    # Rot in x by 45 and y by 45
    R4545 = torch.Tensor([
                            [  0.7071068,  0.0000000,  0.7071068 ],
                            [ 0.5000000,  0.7071068, -0.5000000 ],
                            [-0.5000000,  0.7071068,  0.5000000 ]]).to(args.device)

    for batch in test_dataloader:
        point_clouds, labels = batch
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)
        
        # point_clouds = point_clouds @ R4545


        with torch.no_grad():
            pred_labels1 = model1.forward(point_clouds)
            pred_labels1 = pred_labels1.max(dim=1)[1]
        
            pred_labels2 = model2.forward(point_clouds)
            pred_labels2 = pred_labels2.max(dim=1)[1]


        for i in range(point_clouds.shape[0]):
            # if(j==619):
            if(pred_labels1[i] != pred_labels2[i]):
        #     # if(pred_labels[i]==labels[i] and j%50==0):
                viz_cls(args, point_clouds[i], pred_labels1[i], args.output_dir+'/q1cls_' + str(j) + '.gif', args.device)
                viz_cls(args, point_clouds[i], pred_labels2[i], args.output_dir+'/cls_' + str(j) + '.gif', args.device)
            j += 1


    #     correct_point += pred_labels.eq(labels.data).cpu().sum().item()
    #     num_point += labels.view([-1,1]).size()[0]

    
    # # Compute Accuracy
    # test_accuracy = correct_point / num_point
    # # test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    # print ("test accuracy: {}".format(test_accuracy))

