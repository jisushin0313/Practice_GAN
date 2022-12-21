import os, random

import torch
import torch.nn.parallel
import torch.utils.data

from util import save_image_list, get_fid

def tester(args):
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    #writer = SummaryWriter(args.log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('  *** Device: ', device)
    print('  *** Current cuda device:', args.gpu_id)

    modelG_path = os.path.join(args.output_dir,
                              args.model,
                              'rand_{}_b_{}_netG_best.pth'.format(args.seed, args.batch_size))
    if args.model == 'dcgan':
        from gan_family.dcgan import Generator
        params = {
            'nc': args.nc,
            'nz': args.nz,
            'ngf': args.ngf,
            'ndf': args.ndf
        }
        netG = Generator(params['nz'], params['nc'], params['ngf']).cuda()
        netG.load_state_dict(torch.load(modelG_path))
    elif args.model == 'deep_dcgan':
        from gan_family.deep_dcgan import Generator
        from gan_family.dcgan import Generator
        params = {
            'nc': args.nc,
            'nz': args.nz,
            'ngf': args.ngf,
            'ndf': args.ndf
        }
        netG = Generator(params['nz'], params['nc'], params['ngf']).cuda()
        netG.load_state_dict(torch.load(modelG_path))
    elif args.model == 'hdcgan':
        from gan_family.hdcgan import Generator
        params = {
            'nc': args.nc,
            'nz': args.nz,
            'ngf': args.ngf,
            'ndf': args.ndf
        }
        netG = Generator(params['nz'], params['nc'], params['ngf']).cuda()
        netG.load_state_dict(torch.load(modelG_path))

    fixed_noise = torch.randn(1000, params['nz'], 1, 1).cuda()
    fake_dataset = netG(fixed_noise)
    fake_image_path = save_image_list(model_name = args.model,
                                      dataset = fake_dataset,
                                      real = False,
                                      best = False,
                                      path = 'rand_{}_b_{}_nz_{}'.format(args.seed, args.batch_size, args.nz))

    fid_score = get_fid('./../real', fake_image_path, 100)
    print('FID: {:.4f}'.format(fid_score))









