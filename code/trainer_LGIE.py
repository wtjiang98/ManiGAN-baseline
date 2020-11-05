from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET, DCM_Net
from datasets import prepare_data, prepare_data_LGIE
from model import RNN_ENCODER, CNN_ENCODER
from VGGFeatureLoss import VGGNet

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss

import os
import time
import numpy as np
import sys
import cv2

class condGANTrainer(object):
  def __init__(self, output_dir, data_loader, n_words, ixtoword):
    if cfg.TRAIN.FLAG:
      self.model_dir = os.path.join(output_dir, 'Model')
      self.image_dir = os.path.join(output_dir, 'Image')
      mkdir_p(self.model_dir)
      mkdir_p(self.image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    self.batch_size = cfg.TRAIN.BATCH_SIZE
    self.max_epoch = cfg.TRAIN.MAX_EPOCH
    self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

    self.n_words = n_words
    self.ixtoword = ixtoword
    self.data_loader = data_loader
    self.num_batches = len(self.data_loader)

  def build_models(self):
    ################### Text and Image encoders ########################################
    # if cfg.TRAIN.NET_E == '':
    #   print('Error: no pretrained text-image encoders')
    #   return

    VGG = VGGNet()

    for p in VGG.parameters():
      p.requires_grad = False

    print("Load the VGG model")
    VGG.eval()

    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)

    # when NET_E = '', train the image_encoder and text_encoder jointly
    if cfg.TRAIN.NET_E != '':
      state_dict = torch.load(cfg.TRAIN.NET_E,map_location=lambda storage, loc: storage)     .state_dict()
      text_encoder.load_state_dict(state_dict)
      for p in text_encoder.parameters():
        p.requires_grad = False
      print('Load text encoder from:', cfg.TRAIN.NET_E)
      text_encoder.eval()

      img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
      state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)    .state_dict()
      image_encoder.load_state_dict(state_dict)
      for p in image_encoder.parameters():
        p.requires_grad = False
      print('Load image encoder from:', img_encoder_path)
      image_encoder.eval()

    ####################### Generator and Discriminators ##############
    netsD = []
    if cfg.GAN.B_DCGAN:
      if cfg.TREE.BRANCH_NUM ==1:
        from model import D_NET64 as D_NET
      elif cfg.TREE.BRANCH_NUM == 2:
        from model import D_NET128 as D_NET
      else:  # cfg.TREE.BRANCH_NUM == 3:
        from model import D_NET256 as D_NET
      netG = G_DCGAN()
      if cfg.TRAIN.W_GAN:
        netsD = [D_NET(b_jcu=False)]
    else:
      from model import D_NET64, D_NET128, D_NET256
      netG = G_NET()
      netG.apply(weights_init)
      if cfg.TRAIN.W_GAN:
        if cfg.TREE.BRANCH_NUM > 0:
          netsD.append(D_NET64())
        if cfg.TREE.BRANCH_NUM > 1:
          netsD.append(D_NET128())
        if cfg.TREE.BRANCH_NUM > 2:
          netsD.append(D_NET256())
        for i in range(len(netsD)):
          netsD[i].apply(weights_init)

    print('# of netsD', len(netsD))
    #
    epoch = 0
    if cfg.TRAIN.NET_G != '':
      state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
      netG.load_state_dict(state_dict)
      print('Load G from: ', cfg.TRAIN.NET_G)
      istart = cfg.TRAIN.NET_G.rfind('_') + 1
      iend = cfg.TRAIN.NET_G.rfind('.')
      epoch = cfg.TRAIN.NET_G[istart:iend]
      epoch = int(epoch) + 1
      if cfg.TRAIN.B_NET_D:
        Gname = cfg.TRAIN.NET_G
        for i in range(len(netsD)):
          s_tmp = Gname[:Gname.rfind('/')]
          Dname = '%s/netD%d.pth' % (s_tmp, i)
          print('Load D from: ', Dname)
          state_dict = \
            torch.load(Dname, map_location=lambda storage, loc: storage)
          netsD[i].load_state_dict(state_dict)
    # ########################################################### #
    if cfg.CUDA:
      text_encoder = text_encoder.cuda()
      image_encoder = image_encoder.cuda()
      netG.cuda()
      VGG = VGG.cuda()
      for i in range(len(netsD)):
        netsD[i].cuda()
    return [text_encoder, image_encoder, netG, netsD, epoch, VGG]

  def define_optimizers(self, netG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    if cfg.TRAIN.W_GAN:
      for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))

    return optimizerG, optimizersD

  def prepare_labels(self):
    batch_size = self.batch_size
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if cfg.CUDA:
      real_labels = real_labels.cuda()
      fake_labels = fake_labels.cuda()
      match_labels = match_labels.cuda()

    return real_labels, fake_labels, match_labels

  def save_model(self, netG, avg_param_G, netsD, epoch, text_encoder, image_encoder):
    backup_para = copy_G_params(netG)
    load_params(netG, avg_param_G)
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
    load_params(netG, backup_para)
    #
    for i in range(len(netsD)):
      netD = netsD[i]
      torch.save(netD.state_dict(), '%s/netD%d.pth' % (self.model_dir, i))
    print('Save G/Ds models.')

    if cfg.ANNO_PATH:
      torch.save(text_encoder, '%s/text_encoder_%d.pth' % (self.model_dir, epoch))
      torch.save(image_encoder, '%s/image_encoder_%d.pth' % (self.model_dir, epoch))
      print('Save text/image encoder')

  def set_requires_grad_value(self, models_list, brequires):
    for i in range(len(models_list)):
      for p in models_list[i].parameters():
        p.requires_grad = brequires

  def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                       image_encoder, captions, cap_lens,
                       gen_iterations, cnn_code, region_features,
                       real_imgs, name='current'):
    # Save images
    fake_imgs, attention_maps, _, _, _, _ = netG(noise, sent_emb, words_embs, mask,
                                                 cnn_code, region_features)
    for i in range(len(attention_maps)):
      if len(fake_imgs) > 1:
        img = fake_imgs[i + 1].detach().cpu()
        lr_img = fake_imgs[i].detach().cpu()
      else:
        img = fake_imgs[0].detach().cpu()
        lr_img = None
      attn_maps = attention_maps[i]
      att_sze = attn_maps.size(2)
      img_set, _ = build_super_images(img, captions, self.ixtoword,
                           attn_maps, att_sze, lr_imgs=lr_img)
      if img_set is not None:
        im = Image.fromarray(img_set)
        fullpath = '%s/G_%s_%d_%d.png' % (self.image_dir, name, gen_iterations, i)
        im.save(fullpath)

    i = -1
    img = fake_imgs[i].detach()
    region_features, _ = image_encoder(img)
    att_sze = region_features.size(2)
    _, _, att_maps = words_loss(region_features.detach(),
                                words_embs.detach(),
                                None, cap_lens,
                                None, self.batch_size)
    img_set, _ = build_super_images(fake_imgs[i].detach().cpu(),
                         captions, self.ixtoword, att_maps, att_sze)
    if img_set is not None:
      im = Image.fromarray(img_set)
      fullpath = '%s/D_%s_%d.png' \
                 % (self.image_dir, name, gen_iterations)
      im.save(fullpath)

    '''
    # save the real images 
    for k in range(8):
        im = real_imgs[-1][k].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        fullpath = '%s/R_%s_%d_%d.png'\
                % (self.image_dir, name, gen_iterations, k)
        im.save(fullpath)
    '''

  def train(self):
    text_encoder, image_encoder, netG, netsD, start_epoch, VGG = self.build_models()
    avg_param_G = copy_G_params(netG)
    optimizerG, optimizersD = self.define_optimizers(netG, netsD)
    real_labels, fake_labels, match_labels = self.prepare_labels()

    batch_size = self.batch_size
    nz = cfg.GAN.Z_DIM
    noise = Variable(torch.FloatTensor(batch_size, nz))
    fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
    if cfg.CUDA:
      noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    gen_iterations = 0
    for epoch in range(start_epoch, self.max_epoch):
      start_t = time.time()

      data_iter = iter(self.data_loader)
      step = 0
      while step < self.num_batches:

        ######################################################
        # (1) Prepare training data and Compute text embeddings
        ######################################################
        data = data_iter.next()
        input_imgs_list, output_imgs_list, captions, cap_lens = prepare_data_LGIE(data)

        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef

        # matched text embeddings
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        # 不detach! 需要训练！
        # words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

        # if not cfg.ANNO_PATH:
        #   # mismatched text embeddings
        #   w_words_embs, w_sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
        #   w_words_embs, w_sent_emb = w_words_embs.detach(), w_sent_emb.detach()

        # image features: regional and global
        region_features, cnn_code = image_encoder(input_imgs_list[cfg.TREE.BRANCH_NUM-1])

        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
          mask = mask[:, :num_words]

        #######################################################
        # (2) Modify real images
        ######################################################
        noise.data.normal_(0, 1)
        fake_imgs, _, mu, logvar, _, _ = netG(noise, sent_emb, words_embs, mask,
                                              cnn_code, region_features)

        #######################################################
        # (3) Update D network
        ######################################################
        errD_total = 0
        if cfg.TRAIN.W_GAN:
          D_logs = ''
          for i in range(len(netsD)):
            netsD[i].zero_grad()
            errD = discriminator_loss(netsD[i], input_imgs_list[i], fake_imgs[i], sent_emb, real_labels, fake_labels)

            # backward and update parameters
            errD.backward(retain_graph=True)
            optimizersD[i].step()
            errD_total += errD
            D_logs += 'errD%d: %.2f ' % (i, errD)

        #######################################################
        # (4) Update G network: maximize log(D(G(z)))
        ######################################################
        # compute total loss for training G
        step += 1
        gen_iterations += 1

        netG.zero_grad()
        errG_total, G_logs = generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                              words_embs, sent_emb, None, None,
                                              None, VGG, output_imgs_list)
        kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.W_KL
        errG_total += kl_loss
        G_logs += 'kl_loss: %.2f ' % kl_loss
        # backward and update parameters
        errG_total.backward()
        optimizerG.step()
        for p, avg_p in zip(netG.parameters(), avg_param_G):
          avg_p.mul_(0.999).add_(0.001, p.data)

        if gen_iterations % 100 == 0:
          if cfg.TRAIN.W_GAN:
            print(D_logs + '\n' + G_logs)
        # save images
        if gen_iterations % 500 == 0:
          backup_para = copy_G_params(netG)
          load_params(netG, avg_param_G)
          # self.save_img_results(netG, fixed_noise, sent_emb,
          #                       words_embs, mask, image_encoder,
          #                       captions, cap_lens, epoch, cnn_code,
          #                       region_features, output_imgs_list, name='average')

          # JWT_VIS
          nvis = 5
          input_img, output_img, fake_img = input_imgs_list[-1], output_imgs_list[-1], fake_imgs[-1]
          input_img, output_img, fake_img = self.tensor_to_numpy(input_img), self.tensor_to_numpy(output_img), self.tensor_to_numpy(fake_img)
          # (b x h x w x c)
          gap = 50
          text_bg = np.zeros((gap, 256 * 3, 3))
          res = np.zeros((1, 256 * 3, 3))
          for vis_idx in range(nvis):
            cur_input_img, cur_output_img, cur_fake_img = input_img[vis_idx], output_img[vis_idx], fake_img[vis_idx]
            row = np.concatenate([cur_input_img, cur_output_img, cur_fake_img], 1)  # (h, w * 3, 3)
            row = np.concatenate([row, text_bg], 0)  # (h+gap, w * 3, 3)

            cur_cap = captions[vis_idx].data.cpu().numpy()
            sentence = []
            for cap_idx in range(len(cur_cap)):
              if cur_cap[cap_idx] == 0:
                break
              word = self.ixtoword[cur_cap[cap_idx]].encode('ascii', 'ignore').decode('ascii')
              sentence.append(word)
            cv2.putText(row, ' '.join(sentence), (40, 256 + 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)
            res = np.concatenate([res, row], 0)

          # finish and write image
          cv2.imwrite(os.path.join(self.image_dir, f'G_jwtvis_{gen_iterations}.png'), res)
          load_params(netG, backup_para)

      end_t = time.time()

      print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
            % (epoch, self.max_epoch, self.num_batches,
               errD_total, errG_total,
               end_t - start_t))

      if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
        self.save_model(netG, avg_param_G, netsD, epoch, text_encoder, image_encoder)

    self.save_model(netG, avg_param_G, netsD, self.max_epoch, text_encoder, image_encoder)


  def tensor_to_numpy(self, input_img):
    # [-1, 1] --> [0, 1]
    input_img.add_(1).div_(2).mul_(255)
    input_img = input_img.data.cpu().numpy()
    # b x c x h x w --> b x h x w x c
    input_img = np.transpose(input_img, (0, 2, 3, 1))
    return input_img

  def save_singleimages(self, images, filenames, save_dir,
                        split_dir, sentenceID=0):
    for i in range(images.size(0)):
      s_tmp = '%s/single_samples/%s/%s' % \
              (save_dir, split_dir, filenames[i])
      folder = s_tmp[:s_tmp.rfind('/')]
      if not os.path.isdir(folder):
        print('Make a new folder: ', folder)
        mkdir_p(folder)

      fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
      img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
      # range from [0, 1] to [0, 255]
      ndarr = img.permute(1, 2, 0).data.cpu().numpy()
      im = Image.fromarray(ndarr)
      im.save(fullpath)

  def sampling(self, split_dir):
    if cfg.TRAIN.NET_G == '' or cfg.TRAIN.NET_C == '':
      print('Error: the path for main module or DCM is not found!')
    else:
      if split_dir == 'test':
        split_dir = 'valid'

      if cfg.GAN.B_DCGAN:
        netG = G_DCGAN()
      else:
        netG = G_NET()
      netG.apply(weights_init)
      netG.cuda()
      netG.eval()
      # The text encoder
      text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
      state_dict = \
        torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
      text_encoder.load_state_dict(state_dict)
      print('Load text encoder from:', cfg.TRAIN.NET_E)
      text_encoder = text_encoder.cuda()
      text_encoder.eval()
      # The image encoder
      image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
      img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
      state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
      image_encoder.load_state_dict(state_dict)
      print('Load image encoder from:', img_encoder_path)
      image_encoder = image_encoder.cuda()
      image_encoder.eval()

      # The VGG network
      VGG = VGGNet()
      print("Load the VGG model")
      VGG.cuda()
      VGG.eval()

      batch_size = self.batch_size
      nz = cfg.GAN.Z_DIM
      noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
      noise = noise.cuda()

      # The DCM
      netDCM = DCM_Net()
      if cfg.TRAIN.NET_C != '':
        state_dict = \
          torch.load(cfg.TRAIN.NET_C, map_location=lambda storage, loc: storage)
        netDCM.load_state_dict(state_dict)
        print('Load DCM from: ', cfg.TRAIN.NET_C)
      netDCM.cuda()
      netDCM.eval()

      model_dir = cfg.TRAIN.NET_G
      state_dict = \
        torch.load(model_dir, map_location=lambda storage, loc: storage)
      netG.load_state_dict(state_dict)
      print('Load G from: ', model_dir)

      # the path to save modified images
      s_tmp = model_dir[:model_dir.rfind('.pth')]
      save_dir = '%s/%s' % (s_tmp, split_dir)
      mkdir_p(save_dir)

      cnt = 0
      idx = 0
      for _ in range(5):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(self.data_loader, 0):
          cnt += batch_size
          if step % 100 == 0:
            print('step: ', step)

          imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
          wrong_caps_len, wrong_cls_id = prepare_data(data)

          #######################################################
          # (1) Extract text and image embeddings
          ######################################################

          hidden = text_encoder.init_hidden(batch_size)

          words_embs, sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
          words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

          mask = (wrong_caps == 0)
          num_words = words_embs.size(2)
          if mask.size(1) > num_words:
            mask = mask[:, :num_words]

          region_features, cnn_code = \
            image_encoder(imgs[cfg.TREE.BRANCH_NUM - 1])

          #######################################################
          # (2) Modify real images
          ######################################################

          noise.data.normal_(0, 1)
          fake_imgs, attention_maps, mu, logvar, h_code, c_code = netG(noise,
                                                                       sent_emb, words_embs, mask, cnn_code, region_features)

          real_img = imgs[cfg.TREE.BRANCH_NUM - 1]
          real_features = VGG(real_img)[0]

          fake_img = netDCM(h_code, real_features, sent_emb, words_embs, \
                            mask, c_code)
          for j in range(batch_size):
            s_tmp = '%s/single' % (save_dir)
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
              print('Make a new folder: ', folder)
              mkdir_p(folder)
            k = -1
            im = fake_img[j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            fullpath = '%s_s%d.png' % (s_tmp, idx)
            idx = idx+1
            im.save(fullpath)

  def gen_example(self, data_dic):
    if cfg.TRAIN.NET_G == '' or cfg.TRAIN.NET_C == '':
      print('Error: the path for main module or DCM is not found!')
    else:
      # The text encoder
      text_encoder = \
        RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
      state_dict = \
        torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
      text_encoder.load_state_dict(state_dict)
      print('Load text encoder from:', cfg.TRAIN.NET_E)
      text_encoder = text_encoder.cuda()
      text_encoder.eval()

      # The image encoder
      image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
      img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
      state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
      image_encoder.load_state_dict(state_dict)
      print('Load image encoder from:', img_encoder_path)
      image_encoder = image_encoder.cuda()
      image_encoder.eval()

      # The VGG network
      VGG = VGGNet()
      print("Load the VGG model")
      VGG.cuda()
      VGG.eval()

      # The main module
      if cfg.GAN.B_DCGAN:
        netG = G_DCGAN()
      else:
        netG = G_NET()
      s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
      model_dir = cfg.TRAIN.NET_G
      state_dict = \
        torch.load(model_dir, map_location=lambda storage, loc: storage)
      netG.load_state_dict(state_dict)
      print('Load G from: ', model_dir)
      netG.cuda()
      netG.eval()

      # The DCM
      netDCM = DCM_Net()
      if cfg.TRAIN.NET_C != '':
        state_dict = \
          torch.load(cfg.TRAIN.NET_C, map_location=lambda storage, loc: storage)
        netDCM.load_state_dict(state_dict)
        print('Load DCM from: ', cfg.TRAIN.NET_C)
      netDCM.cuda()
      netDCM.eval()

      for key in data_dic:
        save_dir = '%s/%s' % (s_tmp, key)
        mkdir_p(save_dir)
        captions, cap_lens, sorted_indices, imgs = data_dic[key]

        batch_size = captions.shape[0]
        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions), volatile=True)
        cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
        for i in range(1):
          noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
          noise = noise.cuda()

          #######################################################
          # (1) Extract text and image embeddings
          ######################################################
          hidden = text_encoder.init_hidden(batch_size)

          # The text embeddings
          words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)

          # The image embeddings
          region_features, cnn_code = \
            image_encoder(imgs[cfg.TREE.BRANCH_NUM - 1].unsqueeze(0))
          mask = (captions == 0)

          #######################################################
          # (2) Modify real images
          ######################################################
          noise.data.normal_(0, 1)
          fake_imgs, attention_maps, mu, logvar, h_code, c_code = netG(noise,
                                                                       sent_emb, words_embs, mask, cnn_code, region_features)

          real_img = imgs[cfg.TREE.BRANCH_NUM - 1].unsqueeze(0)
          real_features = VGG(real_img)[0]

          fake_img = netDCM(h_code, real_features, sent_emb, words_embs, \
                            mask, c_code)

          cap_lens_np = cap_lens.cpu().data.numpy()
          for j in range(batch_size):
            save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
            for k in range(len(fake_imgs)):
              im = fake_imgs[k][j].data.cpu().numpy()
              im = (im + 1.0) * 127.5
              im = im.astype(np.uint8)
              im = np.transpose(im, (1, 2, 0))
              im = Image.fromarray(im)
              fullpath = '%s_g%d.png' % (save_name, k)
              im.save(fullpath)

            for k in range(len(attention_maps)):
              if len(fake_imgs) > 1:
                im = fake_imgs[k + 1].detach().cpu()
              else:
                im = fake_imgs[0].detach().cpu()
              attn_maps = attention_maps[k]
              att_sze = attn_maps.size(2)
              img_set, sentences = \
                build_super_images2(im[j].unsqueeze(0),
                                    captions[j].unsqueeze(0),
                                    [cap_lens_np[j]], self.ixtoword,
                                    [attn_maps[j]], att_sze)
              if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s_a%d.png' % (save_name, k)
                im.save(fullpath)


            save_name = '%s/%d_sf_%d' % (save_dir, 1, sorted_indices[j])
            im = fake_img[j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            fullpath = '%s_SF.png' % (save_name)
            im.save(fullpath)

          save_name = '%s/%d_s_%d' % (save_dir, 1, 9)
          im = imgs[2].data.cpu().numpy()
          im = (im + 1.0) * 127.5
          im = im.astype(np.uint8)
          im = np.transpose(im, (1, 2, 0))
          im = Image.fromarray(im)
          fullpath = '%s_SR.png' % (save_name)
          im.save(fullpath)
