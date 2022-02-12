rootpath=$1
space=hybrid
overwrite=0

collection=$2
visual_feature=resnext101-resnet152

num_cnn=1
visual_kernel_sizes=3
visual_kernel_stride=2
visual_kernel_num=2048
qkv_out_dim=2048
qkv_input_dim=2048
visual_rnn_size=512
postfix=runs_0
pooling=max
use_bert=$3

visual_mapping_layers_preview=0-1024
visual_mapping_layers_intensive=$visual_mapping_layers_preview
text_mapping_layers=$visual_mapping_layers_intensive

model_name=baseline



# training
gpu=$4
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collection $collection --visual_feature $visual_feature --space $space\
                                            --num_cnn $num_cnn --rootpath $rootpath\
                                            --visual_kernel_sizes $visual_kernel_sizes --model_name $model_name \
                                            --visual_kernel_stride $visual_kernel_stride --visual_rnn_size $visual_rnn_size \
                                            --qkv_input_dim $qkv_input_dim --qkv_out_dim $qkv_out_dim \
                                            --visual_mapping_layers_preview $visual_mapping_layers_preview\
                                            --text_mapping_layers $text_mapping_layers\
                                            --visual_mapping_layers_intensive $visual_mapping_layers_intensive\
                                            --visual_kernel_num $visual_kernel_num\
                                            --postfix $postfix --pooling $pooling\
                                            --use_bert $use_bert

