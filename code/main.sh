

#python main.py --cfg cfg/train_FiveK.yml --gpu 0

#python main.py --cfg cfg/train_FiveK_l1=5.yml --gpu 2

# 感觉VGG也很重要，因为L1的指导实在太过粗糙
python main.py --cfg cfg/train_FiveK_VGG=1.yml --gpu 0

