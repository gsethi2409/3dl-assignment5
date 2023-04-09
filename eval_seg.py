import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader, get_eval_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_epoch')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
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

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(args).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # # Sample Points per Object
    # ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    correct_point = 0
    num_point = 0
    j = 0
    test_dataloader = get_eval_data_loader(args=args)

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

            point_clouds = point_clouds @ R4545

            with torch.no_grad():   
                pred_labels = model.forward(point_clouds)
                pred_labels = pred_labels.max(dim=-1)[1]

            for i in range(point_clouds.shape[0]):
                num_corr_pts = pred_labels[i].eq(labels[i].data).cpu().sum().item()
                # if(num_corr_pts > 9900):
                # if(j == 609 or j == 80 or j == 490 or j == 505 or j == 397):
                # if(j == 471):
                # if(num_corr_pts > 9000 and j%50==0):
                if(j == 400 or j == 150 or j == 135):
                # if(j == 9 or j == 471 or j == 432):
                    print(str(j) + ': ' + str(num_corr_pts/args.num_points))
                    # viz_seg(args, point_clouds[i], pred_labels[i], args.output_dir+'/seg_' + str(j) + '.gif', args.device)
                    # viz_seg(args, point_clouds[i], labels[i], args.output_dir+'/seg_gt_' + str(j) + '.gif', args.device)
                j += 1

            correct_point += pred_labels.eq(labels.data).cpu().sum().item()
            num_point += labels.view([-1,1]).size()[0]


    test_accuracy = correct_point / num_point
    # test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # # Visualize Segmentation Result (Pred VS Ground Truth)
    # viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    # viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
