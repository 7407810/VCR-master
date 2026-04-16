import os
import torch
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from data.data import *
from net.CIDNet import CIDNet


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

eval_parser = argparse.ArgumentParser(description='Evaluation for VCR/CIDNet')

# dataset selection
eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
eval_parser.add_argument('--lol', action='store_true', help='evaluate on LOL-v1')
eval_parser.add_argument('--lol_v2_real', action='store_true', help='evaluate on LOL-v2-Real')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='evaluate on LOL-v2-Synthetic')
eval_parser.add_argument('--SICE_grad', action='store_true', help='evaluate on SICE_Grad')
eval_parser.add_argument('--SICE_mix', action='store_true', help='evaluate on SICE_Mix')
eval_parser.add_argument('--fivek', action='store_true', help='evaluate on FiveK')

eval_parser.add_argument('--best_GT_mean', action='store_true', help='use best GT-mean setting for LOL-v2-Real')
eval_parser.add_argument('--best_PSNR', action='store_true', help='use best PSNR setting for LOL-v2-Real')
eval_parser.add_argument('--best_SSIM', action='store_true', help='use best SSIM setting for LOL-v2-Real')

# custom / unpaired
eval_parser.add_argument('--custome', action='store_true', help='evaluate custom dataset')
eval_parser.add_argument('--custome_path', type=str, default='./YOLO', help='path to custom input images')
eval_parser.add_argument('--unpaired', action='store_true', help='evaluate on unpaired dataset')
eval_parser.add_argument('--DICM', action='store_true', help='evaluate on DICM')
eval_parser.add_argument('--LIME', action='store_true', help='evaluate on LIME')
eval_parser.add_argument('--MEF', action='store_true', help='evaluate on MEF')
eval_parser.add_argument('--NPE', action='store_true', help='evaluate on NPE')
eval_parser.add_argument('--VV', action='store_true', help='evaluate on VV')

# inference control
eval_parser.add_argument('--alpha', type=float, default=1.0, help='alpha used in HVI inverse transform')
eval_parser.add_argument('--gamma', type=float, default=1.0, help='gamma applied to input before inference')
eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth', help='default checkpoint for unpaired setting')

# new: direct checkpoint / output override
eval_parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint')
eval_parser.add_argument('--output_dir', type=str, default='', help='path to save outputs')

ep = eval_parser.parse_args()


def load_checkpoint(model, model_path):
    ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    # strict=False for compatibility with updated structures
    msg = model.load_state_dict(ckpt, strict=False)
    print(f'Pre-trained model is loaded from: {model_path}')
    print(f'Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}')


def eval(model, testing_data_loader, model_path, output_folder,
         norm_size=True, LOL=False, v2=False, unpaired=False, alpha=1.0, gamma=1.0):
    torch.set_grad_enabled(False)

    load_checkpoint(model, model_path)
    model.eval()
    print('Evaluation...')

    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha

    os.makedirs(output_folder, exist_ok=True)

    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]

            input = input.cuda()
            output = model(input ** gamma)

        output = torch.clamp(output, 0, 1)

        if not norm_size:
            if torch.is_tensor(h):
                h = int(h[0])
            if torch.is_tensor(w):
                w = int(w[0])
            output = output[:, :, :h, :w]

        output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
        output_img.save(os.path.join(output_folder, name[0]))

        torch.cuda.empty_cache()

    print('===> End evaluation')

    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    elif unpaired:
        model.trans.gated2 = False

    torch.set_grad_enabled(True)


if __name__ == '__main__':
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")

    os.makedirs('./output', exist_ok=True)

    norm_size = True
    num_workers = 1
    alpha = 1.0
    weight_path = ''
    output_folder = ''

    if ep.lol:
        eval_data = DataLoader(
            dataset=get_eval_set("./datasets/LOLdataset/eval15/low"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False
        )
        output_folder = './output/LOLv1/'
        if ep.perc:
            weight_path = './weights/LOLv1/w_perc.pth'
        else:
            weight_path = './weights/LOLv1/wo_perc.pth'

    elif ep.lol_v2_real:
        eval_data = DataLoader(
            dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False
        )
        output_folder = './output/LOLv2_real/'

        if ep.best_GT_mean:
            weight_path = './weights/LOLv2_real/w_perc.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = './weights/LOLv2_real/best_PSNR.pth'
            alpha = 0.80
        elif ep.best_SSIM:
            weight_path = './weights/LOLv2_real/best_SSIM.pth'
            alpha = 0.82
        else:
            # fallback
            weight_path = './weights/LOLv2_real/w_perc.pth'
            alpha = 0.84

    elif ep.lol_v2_syn:
        eval_data = DataLoader(
            dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False
        )
        output_folder = './output/LOLv2_syn/'
        if ep.perc:
            weight_path = './weights/LOLv2_syn/w_perc.pth'
        else:
            weight_path = './weights/LOLv2_syn/wo_perc.pth'

    elif ep.SICE_grad:
        eval_data = DataLoader(
            dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False
        )
        output_folder = './output/SICE_grad/'
        weight_path = './weights/SICE.pth'
        norm_size = False

    elif ep.SICE_mix:
        eval_data = DataLoader(
            dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False
        )
        output_folder = './output/SICE_mix/'
        weight_path = './weights/SICE.pth'
        norm_size = False

    elif ep.fivek:
        eval_data = DataLoader(
            dataset=get_SICE_eval_set("./datasets/FiveK/test/input"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False
        )
        output_folder = './output/fivek/'
        weight_path = './weights/fivek.pth'
        norm_size = False

    elif ep.unpaired:
        if ep.DICM:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/DICM"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False
            )
            output_folder = './output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/LIME"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False
            )
            output_folder = './output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/MEF"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False
            )
            output_folder = './output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/NPE"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False
            )
            output_folder = './output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/VV"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False
            )
            output_folder = './output/VV/'
        elif ep.custome:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set(ep.custome_path),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False
            )
            output_folder = './output/custome/'
        else:
            raise ValueError("For --unpaired, please also specify one of --DICM/--LIME/--MEF/--NPE/--VV/--custome")

        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights

    else:
        raise ValueError("Please specify one evaluation setting.")

    # override checkpoint/output folder if user provides them
    if ep.checkpoint != '':
        weight_path = ep.checkpoint
    if ep.output_dir != '':
        output_folder = ep.output_dir

    if weight_path == '':
        raise ValueError("Checkpoint path is empty. Please specify --checkpoint or enable a preset dataset option.")

    print(f'Output folder: {output_folder}')
    print(f'Checkpoint  : {weight_path}')
    print(f'Gamma       : {ep.gamma}')
    print(f'Alpha       : {alpha}')

    eval_net = CIDNet().cuda()
    eval(
        eval_net,
        eval_data,
        weight_path,
        output_folder,
        norm_size=norm_size,
        LOL=ep.lol,
        v2=ep.lol_v2_real,
        unpaired=ep.unpaired,
        alpha=alpha,
        gamma=ep.gamma
    )