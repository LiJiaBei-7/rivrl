rootpath=/home/wyb/workplace/disk3/wyb/VisualSearch_hybrid
collectionStrt=single
testCollection=tgif_li
logger_name=/home/wyb/workplace/disk3/wyb/VisualSearch_hybrid/tgif_li/cv_tpami_2021/tgif_li/model_name_is_best_model/use_bert_is_No/num_layer_is_1/num_head_is_1/qkv_input_dim_is_2048/qkv_out_dim_is_2048/num_cnn_is_1/preview_intensive_encoding_hybrid_concate_full_dp_0.2_measure_cosine_jaccard/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_resnext-101_resnet152-13k_visual_rnn_size_512_visual_norm_True_kernel_sizes_3_num_2048_kernel_stride_2/mapping_text_0-1024_video_preview_0-1024_intensive_0-1024_tag_vocab_size_512/loss_func_mrl_margin_0.2_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_0
overwrite=0

gpu=3

CUDA_VISIBLE_DEVICES=$gpu python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name
