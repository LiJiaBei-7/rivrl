#rootpath=$4

#space=latent
overwrite=0

collection=msrvtt10k
visual_feature=resnext101-resnet152
#
#collection=tgif
#visual_feature=resnet152.pth
##
#collection=vatex
#visual_feature=i3d_kinetics

num_cnn=1
num_layer=1
num_head=1
visual_kernel_sizes=3
visual_kernel_stride=1
visual_kernel_num=2048
qkv_out_dim=1024
qkv_input_dim=2048
visual_rnn_size=512
postfix=runs_0

visual_mapping_layers_global=0-1024
visual_mapping_layers_local=$visual_mapping_layers_global
text_mapping_layers=$visual_mapping_layers_local

model_name=ablation_cnn_qkv_global_res


for space in latent hybrid
do
  # training
  gpu=$1
  CUDA_VISIBLE_DEVICES=$gpu python trainer.py --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                              --collection $collection --visual_feature $visual_feature --space $space\
                                              --num_cnn $num_cnn --num_head $num_head --num_layer $num_layer\
                                              --visual_kernel_sizes $visual_kernel_sizes --model_name $model_name \
                                              --visual_kernel_stride $visual_kernel_stride --visual_rnn_size $visual_rnn_size \
                                              --qkv_input_dim $qkv_input_dim --qkv_out_dim $qkv_out_dim \
                                              --visual_mapping_layers_global $visual_mapping_layers_global\
                                              --text_mapping_layers $text_mapping_layers\
                                              --visual_mapping_layers_local $visual_mapping_layers_local\
                                              --visual_kernel_num $visual_kernel_num\
                                              --postfix $postfix
done
