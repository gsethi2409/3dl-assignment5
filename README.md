# Assignment 5

Gunjan Sethi: gunjans@andrew.cmu.edu

## Q1 Classification Model

`python train.py --task cls`

## Q2 Segmentation Model

`python train.py --task seg`

## Q3 Robustness Analysis

### Part 1 - Varying Number of Points

`python eval_cls.py --num_points n`

`python eval_seg.py --num_points n`

where n = 100, 1000, 5000, 10000

### Part 2 - Rotating Pointclouds

`python eval_cls.py`

`python eval_seg.py`

## Q4 Expressive Architectures

`python train.py --task cls`