import logging
import numpy as np
import evaluation
import util.metrics as metrics


def norm_score(t2v_all_errors):
    t2v_all_score = -t2v_all_errors
    t2v_all_score = t2v_all_score - np.min(t2v_all_score)
    t2v_all_score = t2v_all_score / np.max(t2v_all_score)
    return -t2v_all_score



def cal_perf(t2v_all_errors, v2t_gt, t2v_gt, tb_logger=None, model=None):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_medr, t2v_meanr) = metrics.eval_q2m(t2v_all_errors, t2v_gt)
    t2v_map_score = metrics.t2v_map(t2v_all_errors, t2v_gt)

    # caption retrieval
    (v2t_r1, v2t_r5, v2t_r10, v2t_medr, v2t_meanr) = metrics.eval_q2m(t2v_all_errors.T, v2t_gt)
    v2t_map_score = metrics.v2t_map(t2v_all_errors, v2t_gt)

    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_medr, 1), round(t2v_meanr, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    logging.info(" * Video to text:")
    logging.info(" * r_1_5_10, medr, meanr: {}".format([round(v2t_r1, 1), round(v2t_r5, 1), round(v2t_r10, 1), round(v2t_medr, 1), round(v2t_meanr, 1)]))
    logging.info(" * recall sum: {}".format(round(v2t_r1+v2t_r5+v2t_r10, 1)))
    logging.info(" * mAP: {}".format(round(v2t_map_score, 4)))
    logging.info(" * "+'-'*10)

    if tb_logger is not None:        
        # record metrics in tensorboard
        tb_logger.log_value('v2t_r1', v2t_r1, step=model.Eiters)
        tb_logger.log_value('v2t_r5', v2t_r5, step=model.Eiters)
        tb_logger.log_value('v2t_r10', v2t_r10, step=model.Eiters)
        tb_logger.log_value('v2t_medr', v2t_medr, step=model.Eiters)
        tb_logger.log_value('v2t_meanr', v2t_meanr, step=model.Eiters)

        tb_logger.log_value('t2v_r1', t2v_r1, step=model.Eiters)
        tb_logger.log_value('t2v_r5', t2v_r5, step=model.Eiters)
        tb_logger.log_value('t2v_r10', t2v_r10, step=model.Eiters)
        tb_logger.log_value('t2v_medr', t2v_medr, step=model.Eiters)
        tb_logger.log_value('t2v_meanr', t2v_meanr, step=model.Eiters)

        tb_logger.log_value('v2t_map', v2t_map_score, step=model.Eiters)
        tb_logger.log_value('t2v_map', t2v_map_score, step=model.Eiters)

    return (v2t_r1, v2t_r5, v2t_r10, v2t_medr, v2t_meanr, v2t_map_score), (t2v_r1, t2v_r5, t2v_r10, t2v_medr, t2v_meanr, t2v_map_score)



def validate(opt, tb_logger, vid_data_loader, text_data_loader, model, measure='cosine'):
    # compute the encoding for all the validation video and captions
    model.val_start()
    video_embs_preview, video_embs_intensive, video_ids = evaluation.encode_vid(model.embed_vis, vid_data_loader)
    cap_embs, caption_ids = evaluation.encode_text(model.embed_txt, text_data_loader)

    t2v_all_errors = evaluation.cal_error(video_embs_preview, cap_embs[0], measure)
    t2v_all_errors += evaluation.cal_error(video_embs_intensive, cap_embs[1], measure)
    v2t_gt, t2v_gt = metrics.get_gt(video_ids, caption_ids)

    (v2t_r1, v2t_r5, v2t_r10, v2t_medr, v2t_meanr, v2t_map_score), (t2v_r1, t2v_r5, t2v_r10, t2v_medr, t2v_meanr, t2v_map_score) = cal_perf(t2v_all_errors, v2t_gt, t2v_gt, tb_logger=tb_logger, model=model)
    
    currscore = 0
    if opt.val_metric == "recall":
        if opt.direction == 'i2t' or opt.direction == 'all':
            currscore += (v2t_r1 + v2t_r5 + v2t_r10)
        if opt.direction == 't2i' or opt.direction == 'all':
            currscore += (t2v_r1 + t2v_r5 + t2v_r10)
    elif opt.val_metric == "map":
        if opt.direction == 'i2t' or opt.direction == 'all':
            currscore += v2t_map_score
        if opt.direction == 't2i' or opt.direction == 'all':
            currscore += t2v_map_score

    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore



def validate_hybrid(opt, tb_logger, vid_data_loader, text_data_loader, model, measure='cosine', measure_2='cosine'):
    # compute the encoding for all the validation video and captions
    model.val_start()

    video_embs_preview, video_tag_probs_preview, video_embs_intensive, video_tag_probs_intensive, video_ids = evaluation.encode_hybrid(model.embed_vis,
                                                                                             vid_data_loader)
    cap_embs_preview, cap_tag_probs_preview, cap_embs_intensive, cap_tag_probs_intensive, caption_ids = evaluation.encode_hybrid(model.embed_txt,
                                                                                          text_data_loader)
    v2t_gt, t2v_gt = metrics.get_gt(video_ids, caption_ids)

    t2v_all_errors_preview = evaluation.cal_error(video_embs_preview, cap_embs_preview, measure)
    t2v_all_errors_intensive = evaluation.cal_error(video_embs_intensive, cap_embs_intensive, measure)

    t2v_all_errors_tag_preview = evaluation.cal_error_batch(video_tag_probs_preview, cap_tag_probs_preview, measure_2)
    t2v_all_errors_tag_intensive = evaluation.cal_error_batch(video_tag_probs_intensive, cap_tag_probs_intensive, measure_2)
    t2v_all_errors_preview = norm_score(t2v_all_errors_preview)
    t2v_all_errors_intensive = norm_score(t2v_all_errors_intensive)
    t2v_all_errors_tag_preview = norm_score(t2v_all_errors_tag_preview)
    t2v_all_errors_tag_intensive = norm_score(t2v_all_errors_tag_intensive)
    t2v_all_errors = 0.6 * (t2v_all_errors_preview + t2v_all_errors_intensive) + 0.4 * (t2v_all_errors_tag_preview + t2v_all_errors_tag_intensive)

    (v2t_r1, v2t_r5, v2t_r10, v2t_medr, v2t_meanr, v2t_map_score), (
    t2v_r1, t2v_r5, t2v_r10, t2v_medr, t2v_meanr, t2v_map_score) = cal_perf(t2v_all_errors, v2t_gt, t2v_gt,
                                                                            tb_logger=tb_logger, model=model)
    currscore = 0
    if opt.val_metric == "recall":
        if opt.direction == 'i2t' or opt.direction == 'all':
            currscore += (v2t_r1 + v2t_r5 + v2t_r10)
        if opt.direction == 't2i' or opt.direction == 'all':
            currscore += (t2v_r1 + t2v_r5 + t2v_r10)
    elif opt.val_metric == "map":
        if opt.direction == 'i2t' or opt.direction == 'all':
            currscore += v2t_map_score
        if opt.direction == 't2i' or opt.direction == 'all':
            currscore += t2v_map_score

    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore