import torch
import torch.nn as nn
import argparse
from resnet_model import resnet_
from dataloader import get_loader
from torch.autograd import Variable
from torch import optim
import torchvision.transforms as T
# from logger import Logger
from Tt import *
from gray_model import *
from model_ import *
# import tensorflow as tf

# def add_summary_value(writer, key, value, step):
#     summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
#     writer.add_summary(summary, step)


def train_model(model, train_img_dir, train_csv_file, valid_img_dir, valid_csv_file,\
                batch_size,num_workers, epoch, lr, transform=None, center=True):
    train_loader = get_loader(image_dir=train_img_dir, csv_dir=train_csv_file,\
                         batch_size=batch_size, num_workers=num_workers, transform=transform, center=center)
    valid_loader = get_loader(image_dir=valid_img_dir, csv_dir=valid_csv_file,\
                              batch_size=batch_size, num_workers=num_workers, transform=transform, center=center)

    # tf_summary_write = tf and tf.summary.FileWriter('../log/')

    model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # log = Logger('../log/')

    for e in range(epoch):
        model.train(True)
        print ('#'*20)
        print ('epoch%d'%e)
        for i, (img, target) in enumerate(train_loader):
            img_v = Variable(img, requires_grad=True).cuda()
            target = Variable(target, requires_grad=False).cuda()

            optimizer.zero_grad()

            pred = model(img_v)
            loss = criterion(pred, target)
            loss.backward()

            optimizer.step()

            if i%9 == 0:
                print ("training loss is %f"%loss.data[0])
                # log.scalar_summary('train_loss', loss.data[0], e*len(train_loader)+i)
                #add_summary_value(tf_summary_write, 'train_loss', loss.data[0], e*len(train_loader)+i)
                #tf_summary_write.flush()

        model.train(False)
        val_loss = 0.0
        for img, target in valid_loader:
            img_nv = Variable(img, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda()

            pred = model(img_nv)
            loss = criterion(pred, target)
            val_loss+= loss.data[0]
        print ('-'*20)
        val_loss = val_loss / len(valid_loader)
        # log.scalar_summary('val_loss', val_loss, e)
        # add_summary_value(tf_summary_write, 'val_loss', val_loss, e)
        print ('val_loss: %f'%val_loss)
        print ('-'*20)

        torch.save(model.state_dict(), '../model/model%d.pkl'%e)


def main(parser):
    # transform = T.ToTensor()
    transform = T.Compose([
        cv_resize(224, 224),
        T.ToTensor()
    ])
    model = resnet_()
    # model = gray_model()
    # model = myModle()
    train_model(model,
                train_img_dir=parser.img_dir,
                train_csv_file=parser.csv_file,
                valid_img_dir=parser.val_img_dir,
                valid_csv_file=parser.val_csv_file,
                batch_size=parser.batch_size,
                num_workers=parser.num_workers,
                epoch=parser.epoch,
                lr=parser.lr,
                transform=transform,
                center=parser.center)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--img_dir', type=str, default='../data/data/IMG/')
    args.add_argument('--csv_file', type=str, default='../data/data/driving_log.csv')
    args.add_argument("--val_img_dir", type=str, default='../data/IMG_val/')
    args.add_argument('--val_csv_file', type=str, default='../data/driving_log_val.csv')
    args.add_argument('--batch_size', type=int, default=256)
    args.add_argument('--epoch', type=int, default=50)
    args.add_argument('--lr', type=float, default=0.0001)
    args.add_argument('--num_workers', type=int, default=4)
    args.add_argument('--center', type=int, default=1)
    parser = args.parse_args()
    main(parser)