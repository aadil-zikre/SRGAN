# aliased imports
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils

# package imports no alias
import torch
import os
import pytorch_ssim
import math

# Class/Fuction Imports
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Local Imports
from generator import ganGenerator
from discriminator import ganDiscriminator
from generator_loss import GeneratorLoss
from data_utils import Div2kValDataset, Div2kTrainDataset, display_transform

# filter warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    NUM_EPOCHS = 10

    train_set = Div2kTrainDataset('data/DIV2K_train_HR', 'data/DIV2K_train_LR_difficult')
    val_set = Div2kValDataset('data/DIV2K_valid_HR', 'data/DIV2K_valid_LR_difficult' )
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)


    netG = ganGenerator()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = ganDiscriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     netG.to(device)
    #     print("netG sent to cuda")
    #     netD.to(device)
    #     print("netD sent to cuda")
    #     generator_criterion.to(device)
    #     print("generator criterion sent to cuda")

    if torch.cuda.is_available():
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        netG.to(device)
        print("netG sent to cuda")
        netD.to(device)
        print("netD sent to cuda")
        generator_criterion.to(device)
        print("generator criterion sent to cuda")

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()

        for target, source in train_bar:
            g_update_first = True
            batch_size = target.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.to(device)
            z = Variable(source)
            if torch.cuda.is_available():
                z = z.to(device)
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            # fake_img = netG(z)
            # fake_out = netD(fake_img).mean()
            ##
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            # fake_img = netG(z)
            # fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = 'training_results/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            validation_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_hr, val_lr in val_bar:
                batch_size = val_lr.size(0)
                validation_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.to(device)
                    hr = hr.to(device)
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                validation_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                validation_results['ssims'] += batch_ssim * batch_size
                validation_results['psnr'] = 10 * math.log10((hr.max()**2) / (validation_results['mse'] / validation_results['batch_sizes']))
                validation_results['ssim'] = validation_results['ssims'] / validation_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        validation_results['psnr'], validation_results['ssim']))

                val_images.extend(
                    [display_transform()(lr.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                        display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (4, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (4, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(validation_results['psnr'])
        results['ssim'].append(validation_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                        'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(4) + '_train_results.csv', index_label='Epoch')