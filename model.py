import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from loss import TripletLoss
from basic.bigfile import BigFile
from collections import OrderedDict

import math

# from gru_pooling_res import qkv_layer as qkv_res
from PaA import qkv_layer as preview_aware_attention


def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we))
    return np.array(we)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                              fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

def get_mask(kernel_size, stride, lengths, mask):
    lengths = torch.tensor(lengths)
    # num = (lengths / stride).int()
    num = ((lengths - kernel_size) / stride).int() + 1
    conv_len = math.floor((mask.size(1) - kernel_size) / stride) + 1
    mask_change = torch.zeros(mask.size(0), conv_len)
    for k in range(mask.size(0)):
        # if num[k] > conv_len:
        #     num[k] = conv_len
        mask_change[k, :num[k]] = 1.0
    return mask_change.cuda()


class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """

    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch normalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features



class Video_preview_intensive_encoding(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Video_preview_intensive_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.gru_pool = opt.gru_pool
        self.space = opt.space

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        self.num_cnn = opt.num_cnn
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (opt.visual_kernel_sizes[i], 2048), stride=opt.visual_kernel_stride[i])
            for i in range(self.num_cnn)
        ])

        self.kernel_sizes = opt.visual_kernel_sizes
        self.stride = opt.visual_kernel_stride

        self.fc_org = nn.Linear(opt.visual_feat_dim, 2048)

        self.paa = preview_aware_attention(opt, self.rnn_output_size, opt.qkv_input_dim, opt.qkv_out_dim)

        if opt.space == 'latent':
            self.vid_mapping_preview = Latent_mapping(opt.visual_mapping_layers_preview,
                                                     opt.dropout, opt.tag_vocab_size).cuda()
            self.vid_mapping_intensive = Latent_mapping(opt.visual_mapping_layers_intensive,
                                                    opt.dropout, opt.tag_vocab_size).cuda()
        else:
            self.vid_mapping_preview = Hybrid_mapping(opt.visual_mapping_layers_preview,
                                                     opt.dropout, opt.tag_vocab_size).cuda()
            self.vid_mapping_intensive = Hybrid_mapping(opt.visual_mapping_layers_intensive,
                                                    opt.dropout, opt.tag_vocab_size).cuda()


    def forward(self, videos):
        """Extract video feature vectors."""
        # self.rnn.flatten_parameters()
        videos, videos_origin, lengths, mask = videos
        del videos_origin

        # previewing_branch
        gru_init_out, _ = self.rnn(videos)
        if self.gru_pool == 'mean':
            mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
            for i, batch in enumerate(gru_init_out):
                mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
            gru_out = mean_gru
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, mask.unsqueeze(-1)), 1)[0]
        preview_out = self.dropout(gru_out)

        # intensive-reading_branch
        # Map to lower dimensions
        con_input = F.relu(self.fc_org(videos))

        mask_con = mask.unsqueeze(2).expand(-1, -1, con_input.size(2))  # (N,C,F1)
        con_input_mask = con_input * mask_con
        con_input = con_input_mask.unsqueeze(1)
        con_out_list = []
        for i in range(self.num_cnn):
            con_out_i = F.relu(self.convs1[i](con_input)).squeeze(3).permute(0, 2, 1)
            con_out_list.append(con_out_i)
        del mask_con, con_input

        # previewing-aware attention
        intensive_out_list = []
        aware_out_frame = self.paa(mask, con_input_mask, preview_out.unsqueeze(1))
        intensive_out_list.append(aware_out_frame)
        for i in range(self.num_cnn):
            aware_mask = get_mask(self.kernel_sizes[i], self.stride[i], lengths, mask)
            aware_out_i = self.paa(aware_mask, con_out_list[i], preview_out.unsqueeze(1))
            intensive_out_list.append(aware_out_i)

        intensive_out = torch.cat(intensive_out_list, 1)
        del mask, con_out_list, intensive_out_list

        # mapping--
        if self.space == 'latent':
            preview_out = self.vid_mapping_preview(preview_out)
            intensive_out = self.vid_mapping_intensive(intensive_out)
        else:
            preview_out, preview_concept = self.vid_mapping_preview(preview_out)
            intensive_out, intensive_concept = self.vid_mapping_intensive(intensive_out)

        if self.space == 'latent':
            return preview_out, intensive_out
        else:
            return (preview_out, preview_concept), (intensive_out, intensive_concept)

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
                # print(new_state[name], ':', param)

        super(Video_preview_intensive_encoding, self).load_state_dict(new_state)



class Text_multilevel_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.gru_pool = opt.gru_pool
        self.loss_fun = opt.loss_fun

        # visual bidirectional rnn encoder
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size),
                      padding=(int((window_size - 1) / 2), 0))
            for window_size in opt.text_kernel_sizes
        ])

        if opt.space == 'latent':
            self.text_mapping_preview = Latent_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size).cuda()
            self.text_mapping_intensive = Latent_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size).cuda()
        else:
            self.text_mapping_preview = Hybrid_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size).cuda()
            self.text_mapping_intensive = Hybrid_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size).cuda()

        self.init_weights()

        self.space = opt.space

        self.use_bert = opt.use_bert

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, text, *args):
        # Embed word ids to vectors
        self.rnn.flatten_parameters()

        cap_wids, cap_bows, cap_bert, lengths, cap_mask = text

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids = self.embed(cap_wids)
        packed = pack_padded_sequence(cap_wids, lengths, batch_first=True)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]

        if self.gru_pool == 'mean':
            gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).cuda()
            for i, batch in enumerate(padded[0]):
                gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, cap_mask.unsqueeze(-1)), 1)[0]
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        if self.use_bert == 1:
            features = torch.cat((gru_out, con_out, org_out, cap_bert), 1)
        else:
            features = torch.cat((gru_out, con_out, org_out), 1)

        if self.space == 'latent':
            features_list = []
            features_preview = self.text_mapping_preview(features)
            features_intensive = self.text_mapping_intensive(features)
            features_list.append(features_preview)
            features_list.append(features_intensive)
        else:
            features_preview, features_caption_preview = self.text_mapping_preview(features)
            features_intensive, features_caption_intensive = self.text_mapping_intensive(features)

        if self.space == 'latent':
            return features_list
        else:
            return (features_preview, features_caption_preview), (features_intensive, features_caption_intensive)



class Hybrid_mapping(nn.Module):

    def __init__(self, mapping_layers, dropout, tag_vocab_size, l2norm=True):
        super(Hybrid_mapping, self).__init__()

        self.l2norm = l2norm
        self.mapping = MFC(mapping_layers, dropout, have_bn=True, have_last_bn=True)

        self.tag_fc = nn.Linear(mapping_layers[0], tag_vocab_size)
        self.tag_fc_batch_norm = nn.BatchNorm1d(tag_vocab_size)

    def forward(self, features):
        # mapping to concept space
        tag_prob = self.tag_fc(features)
        tag_prob = self.tag_fc_batch_norm(tag_prob)
        concept_features = torch.sigmoid(tag_prob)

        # mapping to latent space
        latent_features = self.mapping(features)
        if self.l2norm:
            latent_features = l2norm(latent_features)

        return (latent_features, concept_features)


class Latent_mapping(nn.Module):

    def __init__(self, mapping_layers, dropout, l2norm=True):
        super(Latent_mapping, self).__init__()

        self.l2norm = l2norm
        self.mapping = MFC(mapping_layers, dropout, have_bn=True, have_last_bn=True)

    def forward(self, features):
        # mapping to latent space
        latent_features = self.mapping(features)
        if self.l2norm:
            latent_features = l2norm(latent_features)

        return latent_features


class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoding.state_dict(), self.text_encoding.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.vid_encoding.train()
        self.text_encoding.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.vid_encoding.eval()
        self.text_encoding.eval()

    def init_info(self):
        # init gpu
        if torch.cuda.is_available():
            self.vid_encoding.cuda()
            self.text_encoding.cuda()
            cudnn.benchmark = True

        # init params
        params = list(self.vid_encoding.parameters())
        params += list(self.text_encoding.parameters())
        self.params = params

        # print structure
        print(self.vid_encoding)
        print(self.text_encoding)


class Preview_Intensive_Encoding(BaseModel):
    """
    dual encoding network
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip

        self.vid_encoding = Video_preview_intensive_encoding(opt)
        self.text_encoding = Text_multilevel_encoding(opt)

        self.init_info()

        # Loss and Optimizer
        if opt.loss_fun == 'mrl':
            self.criterion = TripletLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation,
                                         cost_style=opt.cost_style,
                                         direction=opt.direction)
        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params, lr=opt.learning_rate)

        self.Eiters = 0

        self.use_bert = opt.use_bert

    def forward_emb(self, videos, targets, volatile=False, *args):
        """Compute the video and caption embeddings
        """
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = videos
        if torch.cuda.is_available():
            frames = frames.cuda()

        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        videos_data = (frames, mean_origin, video_lengths, vidoes_mask)

        # text data
        captions, cap_bows, cap_bert, lengths, cap_masks = targets
        if captions is not None:
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        if cap_bert is not None:
            if torch.cuda.is_available():
                cap_bert = cap_bert.cuda()
        text_data = (captions, cap_bows, cap_bert, lengths, cap_masks)

        vid_emb_preview, vid_emb_intensive = self.vid_encoding(videos_data)
        cap_embs = self.text_encoding(text_data)
        return vid_emb_preview, vid_emb_intensive, cap_embs

    def embed_vis(self, vis_data, volatile=True):
        """Compute the video embeddings
        """
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = vis_data
        if torch.cuda.is_available():
            frames = frames.cuda()

        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames, mean_origin, video_lengths, vidoes_mask)

        vid_emb_preview, vid_emb_intensive = self.vid_encoding(vis_data)

        return vid_emb_preview, vid_emb_intensive

    def embed_txt(self, txt_data):
        """Compute the caption embeddings
        """
        # text data
        captions, cap_bows, cap_bert, lengths, cap_masks = txt_data
        if captions is not None:
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()

        if cap_bert is not None:
            if torch.cuda.is_available():
                cap_bert = cap_bert.cuda()
        txt_data = (captions, cap_bows, cap_bert, lengths, cap_masks)
        cap_embs = self.text_encoding(txt_data)
        return cap_embs

    def forward_loss(self, cap_embs, vid_emb_preview, vid_emb_intensive, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        loss = self.criterion(cap_embs[0], vid_emb_preview)
        loss += self.criterion(cap_embs[1], vid_emb_intensive)

        self.logger.update('Le', loss.item(), vid_emb_preview.size(0))

        return loss

    def train_emb(self, videos, captions, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb_preview, vid_emb_intensive, cap_embs = self.forward_emb(videos, captions, False)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(cap_embs, vid_emb_preview, vid_emb_intensive)
        loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        return vid_emb_preview.size(0), loss_value


class Preview_Intensive_Encoding_Hybrid(Preview_Intensive_Encoding):
    """
    dual encoding network
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.tag_vocab_size = opt.tag_vocab_size
        self.loss_fun = opt.loss_fun
        self.measure_2 = opt.measure_2
        self.space = opt.space

        self.vid_encoding = Video_preview_intensive_encoding(opt)
        self.text_encoding = Text_multilevel_encoding(opt)


        self.init_info()

        # Loss and Optimizer
        self.triplet_latent_criterion = TripletLoss(margin=opt.margin,
                                                    measure=opt.measure,
                                                    max_violation=opt.max_violation,
                                                    cost_style=opt.cost_style,
                                                    direction=opt.direction)

        self.triplet_concept_criterion = TripletLoss(margin=opt.margin_2,
                                                     measure=opt.measure_2,
                                                     max_violation=opt.max_violation,
                                                     cost_style=opt.cost_style,
                                                     direction=opt.direction)

        self.tag_criterion = nn.BCELoss()

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params, lr=opt.learning_rate)

        self.Eiters = 0

    def forward_loss(self, cap_emb_preview, cap_emb_intensive, vid_emb_preview, vid_emb_intensive,
                     cap_tag_prob_preview, cap_tag_prob_intensive, vid_tag_prob_preview, vid_tag_prob_intensive,
                     target_tag, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """

        # classification on both video and text
        if cap_emb_preview is not None:
            batch_size = cap_emb_preview.shape[0]
        else:
            batch_size = vid_tag_prob_preview.shape[0]

        loss_1 = self.triplet_latent_criterion(cap_emb_preview, vid_emb_preview)
        loss_1 += self.triplet_latent_criterion(cap_emb_intensive, vid_emb_intensive)

        loss_2 = self.triplet_concept_criterion(cap_tag_prob_preview, vid_tag_prob_preview)
        loss_2 += self.triplet_concept_criterion(cap_tag_prob_intensive, vid_tag_prob_intensive)

        loss_3 = self.tag_criterion(vid_tag_prob_preview, target_tag)
        loss_3 += self.tag_criterion(vid_tag_prob_intensive, target_tag)
        loss_4 = self.tag_criterion(cap_tag_prob_preview, target_tag)
        loss_4 += self.tag_criterion(cap_tag_prob_intensive, target_tag)

        loss = loss_1 + loss_2 + batch_size * (loss_3 + loss_4)
        if vid_emb_preview is not None:
            self.logger.update('Le', loss.item(), vid_emb_preview.size(0))
        else:
            self.logger.update('Le', loss.item(), vid_tag_prob_preview.size(0))

        return loss

    def train_emb(self, videos, captions, target_tag, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb_preview_s, vid_emb_intensive_s, cap_embs = self.forward_emb(videos, captions, False)

        vid_emb_preview, vid_tag_prob_preview = vid_emb_preview_s
        vid_emb_intensive, vid_tag_prob_intensive = vid_emb_intensive_s

        cap_emb_preview_s, cap_emb_intensive_s = cap_embs
        cap_emb_preview, cap_tag_prob_preview = cap_emb_preview_s
        cap_emb_intensive, cap_tag_prob_intensive = cap_emb_intensive_s


        target_tag = Variable(target_tag, volatile=False)
        if torch.cuda.is_available():
            target_tag = target_tag.cuda()

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(cap_emb_preview, cap_emb_intensive, vid_emb_preview, vid_emb_intensive, cap_tag_prob_preview,
                                 cap_tag_prob_intensive, vid_tag_prob_preview, vid_tag_prob_intensive,  target_tag)
        loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        if vid_emb_preview is not None:
            batch_size = vid_emb_preview.size(0)
        else:
            batch_size = vid_tag_prob_preview.size(0)
        return batch_size, loss_value

    def get_pre_tag(self, vid_emb_wo_norm):
        pred_prob = vid_emb_wo_norm[:, :self.tag_vocab_size]
        pred_prob = torch.sigmoid(pred_prob)
        return pred_prob


NAME_TO_MODELS = {'preview_intensive_encoding_latent': Preview_Intensive_Encoding, 'preview_intensive_encoding_hybrid': Preview_Intensive_Encoding_Hybrid}


def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.' % name
    return NAME_TO_MODELS[name]
