from logger import Logger
import torch
from torch.autograd import Variable
import argparse
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from T import *
from data_loader import *
from prepare_data import *
from resample import *
from mode_DenseNetl import *
import torchvision.transforms as transform


def train_model(data, model, criterion, optimizer, scheduler, num_epochs=30, batch_size=128, use_cuda=True, transforms = None):
    if use_cuda:
        model.cuda()

    # Initialize a logger
    logger = Logger('./log')

    X_train, y_train, X_valid, y_valid, _, _ = data
    train_dataloader = get_loader(X_train, y_train, transforms, batch_size=batch_size, shuffle=True)
    valid_dataloader = get_loader(X_valid, y_valid, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        if optimizer.param_groups[0]['lr'] > 0.00001:
            scheduler.step()
        print('Epoch %d' % (epoch))
        for i, (input, target) in enumerate(train_dataloader):
            # use cuda
            model.train(True)
            if use_cuda:
                input = Variable(input).cuda()
                target = Variable(target.squeeze()).cuda()
            else:
                input = Variable(input)
                target = Variable(target.squeeze())

            # zero_grad
            optimizer.zero_grad()

            # forward and cal loss
            output = model(input)
            loss = criterion(output, target)

            # evaluation for traing
            _, pred = torch.max(output.data, 1)
            acc = torch.sum(pred == target.data) / float(pred.size(0))

            # step
            loss.backward()
            optimizer.step()

            if i % 9 == 0:
                print('The %d batch acc is %f and the loss is %f' % (i, acc, loss.data[0]))
                step = epoch * len(train_dataloader) + i
                # =================TensorBoard logging==============#
                info = {
                    'loss': loss.data[0],
                    'acc': acc
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

        # validation for each epoch
        loss = 0
        all_equal = 0.0
        all_count = X_valid.shape[0]
        for i, (input, target) in enumerate(valid_dataloader):
            model.train(False)
            if use_cuda:
                input = Variable(input, volatile=True).cuda()
                target = Variable(target.squeeze(), volatile=True).cuda()
            else:
                input = Variable(input, volatile=True)
                target = Variable(target.squeeze(), volatile=True)

            output = model(input)

            loss += criterion(output, target)

            _, pred = torch.max(output.data, 1)
            all_equal += torch.sum(pred == target.data)
        print('-' * 20)
        print('validation')
        print("The %d epoch loss is %f and the acc is %f" % (epoch, loss.data[0], all_equal / all_count))
        print('-' * 20)
        info = {
            'val_acc': all_equal / all_count,
            'val_loss': loss.data[0]
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

    # save the model
    print ("saving the model")
    torch.save(model.state_dict(), 'model/model.pkl')
    print ('Finished')

def main(args):
    model = transfer_DenseNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)  # when you tune the learning_rate, it help a lot
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)

    # data augmenttion and preprocess
    transforms = transform.Compose([
        Zoom(0.2),
        transform.RandomCrop(32, padding=4),
        Rotation(35),
        Shift(0.2),
        transform.ToTensor(),
        Gray()
    ])

    data = prepare_data(args.train_file, args.valid_file, args.test_file)
    if args.use_resample:
        data[0], data[1] = resample_equal_prob(data[0], data[1])
        print ("Finishing resample")

    epochs = args.epochs
    batch_size = args.batch_size
    use_cuda = args.use_cuda
    train_model(data, model, criterion, optimizer, scheduler, num_epochs=epochs, batch_size=batch_size, use_cuda=use_cuda,
                transforms=transforms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/train.p')
    parser.add_argument('--valid_file', type=str, default='data/valid.p')
    parser.add_argument('--test_file', type=str, default='data/test.p')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--lr_decay_step', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_resample', type=bool, default=False)
    args = parser.parse_args()
    main(args)
