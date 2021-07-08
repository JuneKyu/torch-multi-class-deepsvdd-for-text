# news20
python main.py news20 news20_LinearNet ../log/news20_comp ../data \
  --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 \
  --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 \
  --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-3 \
  --normal_class 0;
#
# python main.py news20 news20_LinearNet ../log/news20_rec ../data \
#   --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 \
#   --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 \
#   --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-3 \
#   --normal_class 1;
#
# python main.py news20 news20_LinearNet ../log/news20_sci ../data \
#   --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 \
#   --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 \
#   --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-3 \
#   --normal_class 2;
#
# python main.py news20 news20_LinearNet ../log/news20_misc ../data \
#   --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 \
#   --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 \
#   --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-3 \
#   --normal_class 3;
#
# python main.py news20 news20_LinearNet ../log/news20_pol ../data \
#   --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 \
#   --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 \
#   --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-3 \
#   --normal_class 4;
#
# python main.py news20 news20_LinearNet ../log/news20_rel ../data \
#   --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 \
#   --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 \
#   --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-3 \
#   --normal_class 5;


# mnist
# python main.py mnist mnist_LeNet ../log/mnist_test ../data \
#   --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 \
#   --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 \
#   --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 \
#   --normal_class 3;
