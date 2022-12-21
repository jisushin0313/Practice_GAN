import os, random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from util import save_image_list, get_fid

def trainer(args):
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    #writer = SummaryWriter(args.log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('  *** Device: ', device)
    print('  *** Current cuda device:', args.gpu_id)

    datapath = os.path.join(args.data_path, 'celeba/')
    celeba_dataset = dset.ImageFolder(root=datapath,
                          transform=transforms.Compose([
                              transforms.Resize(64),
                              transforms.CenterCrop(64),
                              transforms.ToTensor(),
                              transforms.Normalize(
                                  (0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5)
                              )
                          ]))
    dataloader = torch.utils.data.DataLoader(celeba_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    '''
    if args.model == 'dcgan' or args.model == 'deep_dcgan':
        if args.model == 'dcgan':
            from gan_family.dcgan import Generator, Discriminator, weights_init
        elif args.model == 'deep_dcgan':
            from gan_family.deep_dcgan import Generator, Discriminator
    '''
    if args.model == 'dcgan':
        from gan_family.dcgan import Generator, Discriminator, weights_init
        params = {
            'nc': args.nc,
            'nz': args.nz,
            'ngf': args.ngf,
            'ndf': args.ndf
        }
        netG = Generator(params['nz'], params['nc'], params['ngf']).cuda()
        netG.apply(weights_init)
        netD = Discriminator(params['nc'], params['ndf']).cuda()
        netD.apply(weights_init)
    elif args.model == 'hdcgan':
        from gan_family.hdcgan import Generator, Discriminator
        params = {
            'nc': 3,
            'nz': 100,
            'ngf': 64,
            'ndf': 64
        }
        netG = Generator(params['nz'], params['nc'], params['ngf']).cuda()
        netD = Discriminator(params['nc'], params['ndf']).cuda()
    elif args.model == 'stylegan2':
        from gan_family.stylegan2 import Generator, Discriminator
        params = {'size': 64,
                  'latent': 512,
                  'n_mlp': 4,
                  'channel_multiplier': 2
        }
        netG = Generator(params['size'], params['latent'], params['n_mlp'], channel_multiplier=params['channel_multiplier'])
        netD = Discriminator(params['size'], params['latent'], params['n_mlp'], channel_multiplier=params['channel_multiplier'])
    elif args.model =='pggan':
        from pggan import Generator, Discriminator

    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)

    criterion = nn.BCELoss()

    n_epoch = args.epochs
    fixed_noise = torch.randn(1000, params['nz'], 1, 1).cuda()
    # if args.model != 'stylegan2':
    #    fixed_noise = torch.randn(1000, params['latent']).cuda()
    best_fid = 1e9
    best_epoch = 0
    for epoch in range(n_epoch):
        for i, (data, _) in enumerate(dataloader):
            # train D with real
            optimizerD.zero_grad()

            data = data.cuda()  # real image
            batch_size = data.size(0)
            label = torch.ones((batch_size,)).cuda()  # real label (1)

            output = netD(data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            # train D with fake
            if args.model != 'stylegan2':
                noise = torch.randn(batch_size, params['nz'], 1, 1).cuda()
            else:
                noise = torch.randn(batch_size, params['latent']).cuda()


            fake = netG(noise)  # fake image
            label = torch.zeros((batch_size,)).cuda()  # fake label (0)

            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            #errD.backward()
            optimizerD.step()

            # train G
            optimizerG.zero_grad()
            label = torch.ones((batch_size,)).cuda()  # should be fake D, pretend to be real (1)

            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()


        # save the output
        with torch.no_grad():
            fake = netG(fixed_noise)

        fake_image_path = save_image_list(args.model, fake, False, False, 'rand_{}_b_{}_nz_{}'.format(args.seed, args.batch_size, args.nz))
        #training_progress_images_list = save_gif(training_progress_images_list, fake)

        fid_score = get_fid('./../real', fake_image_path, 100)

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f FID: %.4f' % (epoch, n_epoch, errD.item(), errG.item(), fid_score))

        if fid_score < best_fid:
            #print("save the best model...")
            best_fid = fid_score
            best_epoch = epoch
            _ = save_image_list(args.model, fake, False, True, 'rand_{}_b_{}_nz_{}'.format(args.seed, args.batch_size, args.nz))
            torch.save(netG.state_dict(), './checkpoint/{}/rand_{}_b_{}_netG_best.pth'.format(args.model, args.seed, args.batch_size))
            torch.save(netD.state_dict(), './checkpoint/{}/rand_{}_b_{}_netD_best.pth'.format(args.model, args.seed, args.batch_size))

    print("Model {} Random {} Batch {} Best Epoch {} Best FID {}\n\n\n".format(args.model, args.seed, args.batch_size, best_epoch, best_fid))






