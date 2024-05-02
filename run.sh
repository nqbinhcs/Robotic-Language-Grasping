
# conda activate grasp-anything

python train_network.py --dataset grasp-anything --dataset-path train_data/grasp-anything/seen --network trans_ragt --use-instruction --use-depth 0 --batch-size 32 --vis 

# python train_network.py --dataset grasp-anything --dataset-path train_data/grasp-anything/seen --use-depth 0 --batch-size 4 --vis


# python train_network.py --dataset grasp-anything --dataset-path train_data/grasp-anything/seen --network ragt --channel-size 18 --use-depth 0 --batch-size 4 


# python evaluate.py --network weights/model_grasp_anything --dataset grasp-anything --dataset-path train_data/grasp-anything/seen --iou-eval --use-depth 0 --vis --n-grasps 1