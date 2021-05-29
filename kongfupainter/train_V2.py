'''
**@Author** :shilongshen
**@data** :2020-9-23
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import random
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn import init
import torch.optim as optim
import itertools
from torch.optim import lr_scheduler
from visdom import Visdom
import argparse
import shutil

# set random seed
manualSeed = 100
print("Random seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# set some parameters
parser = argparse.ArgumentParser(description="set parameters for model")
parser.add_argument("--batch_size", type=int, default=3, help="set batchsize for training model")
parser.add_argument("--epochs", type=int, default=150, help="set epochs for training model")
parser.add_argument("--input_channel", type=int, default=3, help="set training image's channel size")
parser.add_argument("--ngf", type=int, default=64, help="set basic number")
parser.add_argument("--init_type", type=str, default='xavier', help="set init_type for training model")
parser.add_argument("--num_resnetblocks", type=int, default=6, help="set resnetblocks' number")
parser.add_argument("--is_test", action="store_true", default=False, help="whether to test ")
parser.add_argument("--is_train", action="store_false", default=True, help="whether to train")
parser.add_argument("--continue_train", action="store_true", default=False, help="whether to continue training")
parser.add_argument("--lr_G", type=float, default=0.0001, help="learning rate for Generator")
parser.add_argument("--lr_D", type=float, default=0.0002, help="learning rate for Discriminator")
parser.add_argument("--style_dir", default="/home/wag/ssl/kongfupainter/gan-getting-started" + "/monet_jpg",
                    help="Monet train dataset")
parser.add_argument("--content_dir", default="/home/wag/ssl/kongfupainter/gan-getting-started" + "/photo_jpg",
                    help="Photo traing dataset")
parser.add_argument("--save_model_root", default='/home/wag/ssl/kongfupainter/saved_model',
                    help="the path to save trained model")
parser.add_argument("--save_result_root", default="/home/wag/ssl/kongfupainter/result/",
                    help="the path to save test images")

args = parser.parse_args()


def str2bool(str):
    return True if str.lower() == 'true' else False


# style_dir = "/home/wag/ssl/kongfupainter/gan-getting-started" + "/monet_jpg"
# # style_dataset = FlatFolderDataset(style_dir)  # 加载训练数据集
# # print(len(style_dataset))
# content_dir = "/home/wag/ssl/kongfupainter/gan-getting-started" + "/photo_jpg"
# # content_dataset = FlatFolderDataset(content_dir)  # 加载训练数据集
# # print(len(content_dataset))
# save_model_root = '/home/wag/ssl/kongfupainter/saved_model'
#
# batch_size = 2
#
# input_channel = 3
#
# ngf = 64
#
# num_resnetblocks = 6
#
# epochs = 200
#
# init_type = 'xavier'
#
# continue_train = True
# is_test = False
# is_train = True


###################################
# set device
###################################
# TODO ： 思考模型并行训练的其他方式以及如何设置具体使用哪一个GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"  # 设置程序可见的实际GPU id
devices_ids = [0, 1]
device = torch.device("cuda:0,1" if (torch.cuda.is_available()) else "cpu")


###################################
# Define Datasetclass
###################################
class TrainDataset(Dataset):
    def __init__(self, style_root, content_root, transform=None):
        super(TrainDataset, self).__init__()
        self.style_root = style_root
        self.style_paths = list(Path(self.style_root).glob('*'))  # 将path路径下的文件路径以列表的形式存放,list存放的是文件的路径
        self.transform = transform

        self.content_root = content_root
        self.content_paths = list(Path(self.content_root).glob('*'))  # 将path路径下的文件路径以列表的形式存放,list存放的是文件的路径

    def __getitem__(self, index):
        style_path = self.style_paths[index]  # 通过索引的方式获取图片路径
        style_img = Image.open(str(style_path)).convert('RGB')  # 以RGB方式获取图片,
        style_img = self.transform(style_img)  # 对图像进行必要的处理,PIL转换为tensor

        content_path = self.content_paths[index]  # 通过索引的方式获取图片路径
        content_img = Image.open(str(content_path)).convert('RGB')  # 获取图片
        content_img = self.transform(content_img)  # 对图像进行必要的处理

        return {'style_img': style_img, 'content_img': content_img}

    def __len__(self):
        return len(self.style_paths)

    def name(self):
        return 'TrainDataset'


class TestDataset(Dataset):
    def __init__(self, content_root, transform=None):
        super(TestDataset, self).__init__()

        self.transform = transform

        self.content_root = content_root
        self.content_paths = list(Path(self.content_root).glob('*'))  # 将path路径下的文件路径以列表的形式存放,list存放的是文件的路径

    def __getitem__(self, index):
        content_path = self.content_paths[index]  # 通过索引的方式获取图片路径
        content_img = Image.open(str(content_path)).convert('RGB')  # 获取图片
        content_img = self.transform(content_img)  # 对图像进行必要的处理

        return content_img

    def __len__(self):
        return len(self.content_paths)

    def name(self):
        return 'TestDataset'


###################################
# Define transform
###################################
def train_transform():
    transform_list = [
        # transforms.Resize(size=(256, 256)),
        # transforms.RandomCrop(256),
        transforms.ToTensor(),
        # TODO: 思考是否可以去掉Normalize
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


def test_transform():
    transform_list = [
        # transforms.Resize(size=(512, 512)),
        # transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


###################################
# set dataset,dataloader
###################################
train_transform = train_transform()
data_set = TrainDataset(args.style_dir, args.content_dir, train_transform)
# print(len(data_set))
data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

test_transform = test_transform()
test_data_set = TestDataset(args.content_dir, test_transform)
# print(len(data_set))
test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=True)


# plot some trainging images
# real_images = next(iter(data_loader))
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("training images")
# plt.imshow(np.transpose(vutils.make_grid(real_images['style_img'].to(device)[:64], padding=2, normalize=True).cpu(),
#                         (1, 2, 0)))
# plt.show()
# plt.imshow(np.transpose(vutils.make_grid(real_images['content_img'].to(device)[:64], padding=2, normalize=True).cpu(),
#                         (1, 2, 0)))
# plt.show()


###################################
# init weight
###################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.normal_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:  # and hasattr(m, 'weight'):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:  # and hasattr(m, 'weight'):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:  # and hasattr(m, 'weight'):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:  # and hasattr(m, 'weight'):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:  # and hasattr(m, 'weight'):
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:  # and hasattr(m, 'weight'):
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


###################################
# Basic Model
###################################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(args.input_channel, args.ngf, 7, 1, 3, padding_mode='reflect')),
            nn.InstanceNorm2d(args.ngf),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(args.ngf, args.ngf * 2, 4, 2, 1, padding_mode="reflect")),
            nn.InstanceNorm2d(args.ngf * 2),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(args.ngf * 2, args.ngf * 4, 4, 2, 1, padding_mode='reflect')),
            nn.InstanceNorm2d(args.ngf * 4),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.main(x)


class Resnet(nn.Module):
    def __init__(self, inputfeature):
        super(Resnet, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inputfeature, inputfeature, 3, 1, 1, padding_mode='reflect')),
            nn.InstanceNorm2d(inputfeature),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(inputfeature, inputfeature, 3, 1, 1, padding_mode='reflect')),
            nn.InstanceNorm2d(inputfeature)
        )

    def forward(self, x):
        return self.conv(x) + x


class ResnetBlocks(nn.Module):
    def __init__(self, num_blocks, dim):
        super(ResnetBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [Resnet(dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(4 * args.ngf, 2 * args.ngf, 3, 1, 1, padding_mode='reflect')),
            nn.InstanceNorm2d(2 * args.ngf),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(2 * args.ngf, args.ngf, 3, 1, 1, padding_mode='reflect')),
            nn.InstanceNorm2d(args.ngf),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(args.ngf, args.input_channel, 3, 1, 1, padding_mode='reflect')),
            nn.InstanceNorm2d(args.input_channel),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(args.input_channel, args.input_channel, 3, 1, 1, padding_mode='reflect')),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


###################################
# Define Generator
###################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.resnetblocks = ResnetBlocks(args.num_resnetblocks, 4 * args.ngf)
        self.decoder = Decoder()

    def forward(self, x):
        # out1 = self.encoder(x)
        # print(out1.shape)
        # out2 = self.resnetblocks(out1)
        # print(out2.shape)
        # out3 = self.decoder(out2)
        # print(out3.shape)
        return self.decoder(self.resnetblocks(self.encoder(x)))


# G = Generator().to(device)
# G.apply(init_weight)
# print(G)


###################################
# Define Discriminator
###################################
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(args.input_channel, args.ngf, 7, 1, 3, padding_mode='reflect')),
            # nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(args.ngf, args.ngf * 2, 4, 2, 1, padding_mode="reflect")),
            nn.InstanceNorm2d(args.ngf * 2),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(args.ngf * 2, args.ngf * 4, 4, 2, 1, padding_mode='reflect')),
            nn.InstanceNorm2d(args.ngf * 4),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.main(x)


# D = Discriminor().to(device)
# D.apply(init_weight)
# print(D)


###################################
# Define CycleGAN
###################################
class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.M2P = Generator()
        self.P2M = Generator()
        self.D_P = Discriminator()
        self.D_M = Discriminator()

        self.M2P = nn.DataParallel(self.M2P, device_ids=devices_ids)
        self.P2M = nn.DataParallel(self.P2M, device_ids=devices_ids)
        self.D_P = nn.DataParallel(self.D_P, device_ids=devices_ids)
        self.D_M = nn.DataParallel(self.D_M, device_ids=devices_ids)

        self.M2P.cuda()
        self.P2M.cuda()
        self.D_P.cuda()
        self.D_M.cuda()

        self.optimM2P = optim.Adam(self.M2P.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.optimP2M = optim.Adam(self.P2M.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.optim_D_P = optim.Adam(self.D_P.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.optim_D_M = optim.Adam(self.D_M.parameters(), lr=0.002, betas=(0.5, 0.999))

        self.optimG = optim.Adam(itertools.chain(self.M2P.parameters(), self.P2M.parameters()), lr=args.lr_G,
                                 betas=(0.5, 0.999))
        self.optimD = optim.Adam(itertools.chain(self.D_P.parameters(), self.D_M.parameters()), lr=args.lr_D,
                                 betas=(0.5, 0.999))

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def weight_init(self):
        print('init model...')
        # print('init type:%s' % init_type)
        init_weights(self.M2P, init_type=args.init_type)
        init_weights(self.P2M, init_type=args.init_type)
        init_weights(self.D_P, init_type=args.init_type)
        init_weights(self.D_M, init_type=args.init_type)
        print('init model done.')

    # def tensor2im(image_tensor, imtype=np.uint8):
    #     image_numpy = image_tensor.cpu().numpy()
    #     if image_numpy.shape[0] == 1:
    #         image_numpy = np.tile(image_numpy, (3, 1, 1))
    #     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    #     return image_numpy.astype(imtype)
    #
    # def show_transform(self):
    #     transform_list = [
    #         transforms.Resize(size=(256, 256)),
    #         # transforms.RandomCrop(256),
    #         # transforms.ToTensor(),
    #         #
    #         transforms.Normalize(mean=[2, 2, 2],
    #                              std=[2, 2, 2])
    #     ]
    #     return transforms.Compose(transform_list)

    def updating_learning_rate(self, optim, name):
        lr_updating = lr_scheduler.StepLR(optimizer=optim, step_size=1, gamma=0.9)
        lr_updating.step()
        print('%s_lr=%.7f' % (name, optim.param_groups[0]['lr']))

    def save_model(self, model, name, epoch):
        print('saving %s epoch %s_model...' % (epoch, name))
        # save_model_root = '/home/wag/ssl/kongfupainter/saved_model'
        file_name = str(epoch) + 'epoch_' + name + '_model.pth'
        dir = os.path.join(args.save_model_root, file_name)
        torch.save(model.state_dict(), dir)
        print('save %s epoch %s_model end' % (epoch, name))

    def Train(self):
        model.train()
        self.weight_init()
        D_loss = []
        G_loss = []
        fake_photo_list = []
        fake_monet_list = []
        cycle_photo_list = []
        cycle_monet_list = []
        Monet_list = []
        Photo_list = []

        viz = Visdom(env='CycleGAN')
        # viz.line([[0.0, 0.0]], [0.0], win='loss', opts=dict(title='G_loss and D_loss', legend=['G_loss', 'D_loss']))
        global_step = 0

        for epoch in range(args.epochs):

            for i, data in enumerate(data_loader):
                Monet = data['style_img'].to(device)
                Photo = data['content_img'].to(device)

                ########################################################
                # updata Discriminator
                ########################################################

                # self.optim_D_M2P.zero_grad()
                # self.optim_D_P2M.zero_grad()
                self.optimD.zero_grad()

                fake_photo = self.M2P(Monet)  # x->G(x)

                fake_monet = self.P2M(Photo)  # y->F(y)

                real_photo_D = self.D_P(Photo)  # y->D_P(y)
                real_monet_D = self.D_M(Monet)  # x->D_M(x)
                fake_photo_D = self.D_P(fake_photo.detach())  # G(x)->D_P(G(x))
                fake_monet_D = self.D_M(fake_monet.detach())  # F(y)->D_P(F(y))

                real_label = torch.ones(real_photo_D.size()).to(device)
                fake_label = torch.zeros(fake_photo_D.size()).to(device)

                real_photo_Dloss = self.l2(real_photo_D, real_label)
                fake_photo_Dloss = self.l2(fake_photo_D, fake_label)
                real_monet_Dloss = self.l2(real_monet_D, real_label)
                fake_monet_Dloss = self.l2(fake_monet_D, fake_label)

                total_Dloss = real_photo_Dloss + fake_photo_Dloss + real_monet_Dloss + fake_monet_Dloss
                total_Dloss.backward()
                self.optimD.step()

                # updata Discriminator_P
                # real_photo_Dloss = self.l2(real_photo_D, real_label)
                # real_photo_Dloss.backward()
                # fake_photo_Dloss = self.l2(fake_photo_D, fake_label)
                # fake_photo_Dloss.backward()
                # photo_Dloss = real_photo_Dloss + fake_photo_Dloss
                # self.optim_D_P.step()

                # updata Discriminator_M
                # real_monet_Dloss = self.l2(real_monet_D, real_label)
                # real_monet_Dloss.backward()
                # fake_monet_Dloss = self.l2(fake_monet_D, fake_label)
                # fake_monet_Dloss.backward()
                # momet_Dloss = real_monet_Dloss + fake_monet_Dloss
                # self.optim_D_M.step()

                ########################################################
                # updata Generator
                ########################################################

                # self.optimM2P.zero_grad()
                # self.optimP2M.zero_grad()
                self.optimG.zero_grad()

                fake_photo = self.M2P(Monet)  # x->G(x)
                fake_monet = self.P2M(Photo)  # y->F(y)

                cycle_monet = self.P2M(fake_photo)  # x->G(x)->F(G(x))
                cycle_photo = self.M2P(fake_monet)  # y->F(y)->G(F(y))

                # GAN loss for G
                fake_photo_GD = self.D_P(fake_photo)
                fake_monet_GD = self.D_M(fake_monet)

                real_label = torch.ones(fake_photo_GD.size()).to(device)

                real_monet_Gloss = self.l2(fake_monet_GD, real_label)
                real_photo_Gloss = self.l2(fake_photo_GD, real_label)

                # cycle loss
                cycle_loss_monet = self.l1(cycle_monet, Monet)
                cycle_loss_photo = self.l1(cycle_photo, Photo)

                # identity loss
                identity_momet = self.P2M(Monet)  # x->F(x)
                identity_photo = self.M2P(Photo)  # y->G(y)

                identity_momet_loss = self.l1(identity_momet, Monet)
                identity_photo_loss = self.l1(identity_photo, Photo)

                total_Gloss = (real_monet_Gloss + real_photo_Gloss) + 10 * (cycle_loss_monet + cycle_loss_photo) + (
                        identity_momet_loss + identity_photo_loss)
                total_Gloss.backward()
                self.optimG.step()

                if i % 10 == 0:
                    print("[%d/%d][%d/%d]\tLoss_D:%.4f\tLoss_G:%.4f" % (
                        epoch, args.epochs, i, len(data_loader), total_Dloss.item(), total_Gloss.item()))

                    ########################################################
                    # plot loss using Visdom
                    ########################################################
                    viz.line([[total_Dloss.item(), total_Gloss.item()]], [global_step], win='loss',
                             update='append' if global_step > 0 else None,
                             opts=dict(title='D_loss and G_loss', legend=['D_loss', 'G_loss']))
                    global_step += 1

                    with torch.no_grad():
                        fake_photo = self.M2P(Monet)  # x->G(x)
                        fake_monet = self.P2M(Photo)  # y->F(y)

                        cycle_monet = self.P2M(fake_photo)  # x->G(x)->F(G(x))
                        cycle_photo = self.M2P(fake_monet)  # y->F(y)->G(F(y))
                    fake_photo_list.append(vutils.make_grid(fake_photo.cpu(), nrow=2, padding=1, normalize=True))
                    fake_monet_list.append(vutils.make_grid(fake_monet.cpu(), nrow=2, padding=1, normalize=True))
                    cycle_photo_list.append(vutils.make_grid(cycle_photo.cpu(), nrow=2, padding=1, normalize=True))
                    cycle_monet_list.append(vutils.make_grid(cycle_monet.cpu(), nrow=2, padding=1, normalize=True))
                    Monet_list.append(vutils.make_grid(Monet.cpu(), nrow=2, padding=1, normalize=True))
                    Photo_list.append(vutils.make_grid(Photo.cpu(), nrow=2, padding=1, normalize=True))

                    ########################################################
                    # plot images using Visdom
                    ########################################################
                    #
                    # fake_photo = fake_photo.cpu().clone()
                    # fake_photo = fake_photo.squeeze(0)
                    # fake_photo = transforms.ToPILImage()(fake_photo)

                    # photo_show=Photo.cpu().numpy()
                    # if i==0:
                    #     print(photo_show.shape)
                    #     print('before',photo_show)#-1~1
                    Photo_show = (Photo + 1) / 2.0  # 0~1
                    Monet_show = (Monet + 1) / 2.0  # 0~1
                    fake_photo_show = (fake_photo + 1) / 2.0  # 0~1
                    fake_monet_show = (fake_monet + 1) / 2.0  # 0~1
                    cycle_photo_show = (cycle_photo + 1) / 2.0  # 0~1
                    cycle_monet_show = (cycle_monet + 1) / 2.0  # 0~1
                    # if i==0:
                    #     print('after',photo_show)
                    # photo_show=self.tensor2im(Photo)
                    viz.images(Photo_show, win='Photo', opts=dict(title='Photo'))
                    viz.images(Monet_show, win='Monet', opts=dict(title='Monet'))
                    viz.images(fake_photo_show, win='fake_photo', opts=dict(title='fake_photo'))
                    viz.images(fake_monet_show, win='fake_monet', opts=dict(title='fake_monet(target_image)'))
                    viz.images(cycle_photo_show, win='cycle_photo', opts=dict(title='cycle_photo'))
                    viz.images(cycle_monet_show, win='cycle_monet', opts=dict(title='cycle_monet'))
                    ########################################################
                    # plot generated image 10 iters
                    ########################################################
                    # plt.figure(figsize=(10, 15))
                    #
                    # plt.subplot(3, 2, 1)
                    # plt.axis = ("off")
                    # plt.title("Photo")
                    # plt.imshow(np.transpose(Photo_list[-1], (1, 2, 0)))
                    #
                    # plt.subplot(3, 2, 2)
                    # plt.axis = ("off")
                    # plt.title("Monet")
                    # plt.imshow(np.transpose(Monet_list[-1].numpy(), (1, 2, 0)))
                    #
                    # plt.subplot(3, 2, 3)
                    # plt.axis = ("off")
                    # plt.title("fake_photo")
                    # plt.imshow(np.transpose(fake_photo_list[-1], (1, 2, 0)))
                    #
                    # plt.subplot(3, 2, 4)
                    # plt.axis = ("off")
                    # plt.title("fake_monet")
                    # plt.imshow(np.transpose(fake_monet_list[-1], (1, 2, 0)))
                    #
                    # plt.subplot(3, 2, 5)
                    # plt.axis = ("off")
                    # plt.title("cycle_photo")
                    # plt.imshow(np.transpose(cycle_photo_list[-1], (1, 2, 0)))
                    #
                    # plt.subplot(3, 2, 6)
                    # plt.axis = ("off")
                    # plt.title("cycle_monet")
                    # plt.imshow(np.transpose(cycle_monet_list[-1], (1, 2, 0)))
                    # # plt.legend()
                    # plt.show()

                D_loss.append(total_Dloss.item())
                G_loss.append(total_Gloss.item())

            ########################################################
            # updating learning rate each epoch
            ########################################################
            self.updating_learning_rate(self.optimD, 'Discriminator')
            self.updating_learning_rate(self.optimG, 'Generator')

            ########################################################
            # saving model each epoch
            ########################################################

            self.save_model(self.M2P, 'M2P', epoch)
            self.save_model(self.P2M, 'P2M', epoch)
            self.save_model(self.D_P, 'D_P', epoch)
            self.save_model(self.D_M, 'D_M', epoch)

            ########################################################
            # plot generated image each 10 epoch
            ########################################################

            if epoch % 10 == 0:
                plt.figure(figsize=(10, 15))
                # plt.title('epoch:%d' % epoch)
                plt.subplot(3, 2, 1)
                plt.axis = ("off")
                plt.title("Photo")
                plt.imshow(np.transpose(Photo_list[-1], (1, 2, 0)))

                plt.subplot(3, 2, 2)
                plt.axis = ("off")
                plt.title("Monet")
                plt.imshow(np.transpose(Monet_list[-1].numpy(), (1, 2, 0)))

                plt.subplot(3, 2, 3)
                plt.axis = ("off")
                plt.title("fake_photo")
                plt.imshow(np.transpose(fake_photo_list[-1], (1, 2, 0)))

                plt.subplot(3, 2, 4)
                plt.axis = ("off")
                plt.title("fake_monet")
                plt.imshow(np.transpose(fake_monet_list[-1], (1, 2, 0)))

                plt.subplot(3, 2, 5)
                plt.axis = ("off")
                plt.title("cycle_photo")
                plt.imshow(np.transpose(cycle_photo_list[-1], (1, 2, 0)))

                plt.subplot(3, 2, 6)
                plt.axis = ("off")
                plt.title("cycle_monet")
                plt.imshow(np.transpose(cycle_monet_list[-1], (1, 2, 0)))
                # plt.legend()
                plt.show()

        ########################################################
        # saving  latest model when finish training
        ########################################################
        self.save_model(self.M2P, 'M2P', 'latest')
        self.save_model(self.P2M, 'P2M', 'latest')
        self.save_model(self.D_P, 'D_P', 'latest')
        self.save_model(self.D_M, 'D_M', 'latest')

        ########################################################
        # plot loss when finish training
        ########################################################
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator loss during training")
        plt.plot(G_loss, label="G_loss")
        plt.plot(D_loss, label="D_loss")
        plt.xlabel("iters")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def Test(self):
        # pred_monet_list = []
        model.eval()
        # trans = transforms.ToPILImage()
        print('saving test images...\n')
        t = tqdm(test_data_loader, leave=False, total=test_data_loader.__len__())
        for i, photo in enumerate(t):
            with torch.no_grad():
                pred_monet = self.P2M(photo.to(device)).cpu().detach()
                pred_monet = (pred_monet + 1) / 2.0

            pred_monet = pred_monet.cpu().clone()
            pred_monet = pred_monet.squeeze(0)
            pred_monet = transforms.ToPILImage()(pred_monet)
            pred_monet.save(args.save_result_root + str(i + 1) + ".jpg")

        print('saving test images end.')


if __name__ == '__main__':
    model = CycleGAN()
    print(args.continue_train)
    print(args.is_train)
    print(args.is_test)
    if args.continue_train:
        file_nameM2P = 'latest' + 'epoch_' + 'M2P' + '_model.pth'
        dirM2P = os.path.join(args.save_model_root, file_nameM2P)

        file_nameP2M = 'latest' + 'epoch_' + 'P2M' + '_model.pth'
        dirP2M = os.path.join(args.save_model_root, file_nameP2M)

        file_nameD_M = 'latest' + 'epoch_' + 'D_M' + '_model.pth'
        dirD_M = os.path.join(args.save_model_root, file_nameD_M)

        file_nameD_P = 'latest' + 'epoch_' + 'D_P' + '_model.pth'
        dirD_P = os.path.join(args.save_model_root, file_nameD_P)

        model.M2P.load_state_dict(torch.load(dirM2P))
        model.P2M.load_state_dict(torch.load(dirP2M))
        model.D_M.load_state_dict(torch.load(dirD_M))
        model.D_P.load_state_dict(torch.load(dirD_P))

    if args.is_train:
        model.Train()

    if args.is_test:
        file_nameP2M = 'latest' + 'epoch_' + 'P2M' + '_model.pth'
        dirP2M = os.path.join(args.save_model_root, file_nameP2M)
        model.P2M.load_state_dict(torch.load(dirP2M))
        model.Test()
        # 将结果压缩
        shutil.make_archive("/home/wag/ssl/kongfupainter/kaggle/images", 'zip', "/home/wag/ssl/kongfupainter/result")
