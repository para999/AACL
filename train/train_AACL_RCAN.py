from options.option_AACL_train import args
from data.srdataset import SRDataset
from torch.utils.data import DataLoader
from utils import degradation, utility
from models.AACL_RCAN.AACL_RCAN import AACL_RCAN
import numpy as np
import torch
import random
import os
import time


def main():
    if args.seed is not None:
        random.seed(args.seed)  # 随机生成一个种子
    torch.manual_seed(args.seed)  # 设置随机数种子

    # creat training & testing dataset and load
    dataset_train = SRDataset(args, name=args.data_train, train=True)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.n_threads, pin_memory=True, drop_last=True)

    dataset_test = SRDataset(args, name=args.data_test, train=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=args.n_threads,
                                 pin_memory=True, drop_last=False)

    # creat model
    print("Creating model...")
    model = AACL_RCAN().cuda()
    # loss & optim
    criterion_moco = torch.nn.CrossEntropyLoss().cuda()
    criterion_sr = torch.nn.L1Loss().cuda()
    optimizer = utility.make_optimizer(args, model)

    # path settings
    model_name = str(args.model_name)
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = args.project_path
    model_path = os.path.join(current_dir, 'experiment', model_name, 'model')
    log_path = os.path.join(current_dir, 'experiment', model_name, 'logs')
    log_file_path = os.path.join(log_path, 'log.txt')

    # resume
    if args.resume:
        if os.path.isfile(args.resume_path):
            checkpoint = torch.load(args.resume_path)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Resume from checkpoint {} , next epoch: {}".format(args.resume_path, args.start_epoch))
            with open(log_file_path, "a") as file:
                file.write("Resume from checkpoint {}, next epoch: {}\n".format(args.resume_path, args.start_epoch))
        else:
            args.start_epoch = 1
            print("checkpoint is not exist. Start training from epoch: {}".format(args.start_epoch))
    else:
        args.start_epoch = 1
        print("checkpoint is not exist. Start training from epoch: {}".format(args.start_epoch))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            with open(log_file_path, "w") as file:
                file.write("Start Training\n")
                file.write("Training Set is {}\n".format(args.data_train))
                file.write("Testing Set is {}\n".format(args.data_test))
                file.write("checkpoint is not exist. Start training from epoch: {}\n".format(args.start_epoch))

    # train eval and test
    for epoch in range(args.start_epoch, args.epochs_sr + 1):
        # leaning rate settings
        adjust_learning_rate_sr(optimizer, epoch, args)

        # train and eval
        train(args, dataloader_train, model, criterion_sr, criterion_moco, optimizer, epoch, log_file_path)

        # save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(current_dir, 'experiment', model_name, 'model', 'model_{:04d}.pth.tar'.format(epoch))
        torch.save(checkpoint, save_path)

        # test
        if epoch > args.epochs_dp:
            print("Epoch:{} Do test".format(epoch))
            test(args, dataloader_test, model, log_file_path)


def train(args, train_loader, model, criterion_sr, criterion_dp, optimizer, epoch, log_file_path):
    # print training information and save
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_dp = AverageMeter('con_Loss', ':.4e')
    losses_sr = AverageMeter('sr_Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    psnrs = AverageMeter('PSNR', ':.4e')

    progress_dp = ProgressMeter(
        len(train_loader) - 1,
        [batch_time, data_time, losses],
        log_file_path,
        prefix="Epoch: [{}]".format(epoch))

    progress_sr = ProgressMeter(
        len(train_loader) - 1,
        [batch_time, data_time, losses, losses_dp, losses_sr, psnrs],
        log_file_path,
        prefix="Epoch: [{}]".format(epoch))
    model.train()  # switch to the train mode

    end = time.time()
    # degraded image
    degrade = degradation.SimpleDegradation(args)

    for batch, (hr, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        hr = hr.cuda(non_blocking=True)

        # lr image
        lr = degrade(hr)
        if epoch <= args.epochs_dp:
            feature, output, target = model.M(im_q=lr[:, 0, ...], im_k=lr[:, 1, ...])
            loss = criterion_dp(output, target)
            losses.update(loss.item())

        else:
            # compute output
            sr, output, target = model(lr)

            hr = hr[:, 0, ...]
            loss_dp = criterion_dp(output, target)
            loss_sr = criterion_sr(sr, hr)
            loss = loss_dp + loss_sr

            sr_eval = utility.quantize(sr, args.rgb_range)
            hr_eval = utility.quantize(hr, args.rgb_range)

            psnr = utility.calc_psnr(sr_eval, hr_eval, scale=args.scale, rgb_range=args.rgb_range)

            losses_sr.update(loss_sr.item())
            losses_dp.update(loss_dp.item())
            losses.update(loss.item())
            psnrs.update(psnr)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print
        if epoch <= args.epochs_dp:
            if batch % args.print_every == 0:
                progress_dp.display(batch)
        else:
            if batch % args.print_every == 0:
                progress_sr.display(batch)


def test(args, test_loader, model, log_file_path):
    # test
    model.eval()

    with torch.no_grad():
        test_psnr = 0
        test_ssim = 0

        for batch, (hr, lr, _) in enumerate(test_loader):
            hr = hr.cuda()  # 1 c w h
            lr = lr.cuda()  # 1 c w/scale h/scale

            # hr = crop_border_test(hr, args.scale)
            # lr = crop_border_test(lr, 1)

            sr = model(lr)

            sr = utility.quantize(sr, args.rgb_range)
            hr = utility.quantize(hr, args.rgb_range)

            test_psnr += utility.calc_psnr(sr, hr, args.scale, args.rgb_range, benchmark=True)
            test_ssim += utility.calc_ssim(sr, hr, args.scale, benchmark=True)

        print("PSNR:{} SSIM:{}".format(test_psnr / len(test_loader), test_ssim / len(test_loader)))
        with open(log_file_path, "a") as file:
            file.write("PSNR:{} SSIM:{}\n".format(test_psnr / len(test_loader), test_ssim / len(test_loader)))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, log_file_path, prefix="", ):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.log_file_path = log_file_path
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        with open(self.log_file_path, "a") as file:
            file.write('\t'.join(entries) + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate_sr(optimizer, epoch, args):
    lr = args.lr_sr * (args.gamma_sr ** ((epoch) // args.lr_decay_sr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def crop_border_test(img, scale):
    b, c, h, w = img.size()

    img = img[:, :, :int(h // scale * scale), :int(w // scale * scale)]

    return img


def crop_border_win(img, scale=4, window_size=16):
    _, _, h, w = img.size()
    img = img[:, :, :int(h // scale // window_size * scale * window_size),
          :int(w // scale // window_size * scale * window_size)]
    return img


def random_patch(random_hr):
    b, n, c, h, w = random_hr.shape
    for i in range(b):
        var = np.random.randint(10, 800)
        mean_1 = np.random.randint(0, 255)
        mean_2 = np.random.randint(0, 255)

        patch_1 = gen_tensor(c, h, w, var, mean_1)
        patch_2 = gen_tensor(c, h, w, var, mean_2)

        image = torch.cat((patch_1.unsqueeze(0), patch_2.unsqueeze(0)), dim=0)
        random_hr[i:i + 1, :] = image.unsqueeze(0)

    return random_hr


def gen_tensor(c, h, w, var, mean):
    x = torch.randn(c, h, w)

    current_variance = torch.var(x)

    scale_factor = torch.sqrt(var / current_variance)
    x_scaled = x * scale_factor

    x_scaled = torch.clamp(x_scaled + mean, 0, 255)
    return x_scaled


if __name__ == '__main__':
    main()
