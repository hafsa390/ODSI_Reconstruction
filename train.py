from __future__ import print_function
from __future__ import division
from oral_dental_reconstruction.perceptual_loss_impl.models import LossNetwork
from oral_dental_reconstruction.perceptual_loss_impl.criterion import PerceptualLoss
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from spectral import calc_stats, noise_from_diffs, imshow, mnf
import os
import time
from dataset import DatasetFromHdf5
from resblock import resblock,conv_bn_relu_res_block
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, convert_1channel_to_3channel
from loss import rrmse_loss
from resblock import resblock,conv_bn_relu_res_block
from utils import save_matv73,reconstruction,load_mat,mrae,rmse
import os
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# change the paths
model_path = '10_layer_model_perceptual_loss_output_3_channel'
train_h5_path = "D://PhD_Projects//HSI_Project//vein-visualization-master//oral_dental_reconstruction//train_t32bands5565f021_front_1.h5"
valid_h5_path = "D://PhD_Projects//HSI_Project//vein-visualization-master//oral_dental_reconstruction//valid_t32bandsfe7b7582_face_2.h5"
usePretrain_model = False
initial_learning_rate = 0.0001

def main():
    cudnn.benchmark = True
    # Dataset
    train_data = DatasetFromHdf5(train_h5_path)
    print(len(train_data))
    val_data = DatasetFromHdf5(valid_h5_path)
    print(len(val_data))

    # Data Loader (Input Pipeline)
    train_data_loader = DataLoader(dataset=train_data,
                                   num_workers=0,
                                   batch_size=64,
                                   shuffle=True,
                                   pin_memory=True)

    val_loader = DataLoader(dataset=val_data,
                            num_workers=0,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True)

    # Model
    model = resblock(conv_bn_relu_res_block, 10, 3, 3)
    print(model)

    if usePretrain_model:
        pre_trained_model = 'HS_veins_34to1band.pkl'
        loaded_pre_trained_model = torch.load(pre_trained_model)
        model_param = loaded_pre_trained_model['state_dict']
        model.load_state_dict(model_param)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = model.to(device=device, dtype=torch.float)

    # Parameters, Loss and Optimizer
    start_epoch = 0
    end_epoch = 100
    init_lr = initial_learning_rate
    iteration = 0
    record_test_loss = 1000

    # criterion = rrmse_loss
    # criterion = torch.nn.MSELoss()

    criterion1 = torch.nn.L1Loss()
    lossnet = LossNetwork()

    for param in lossnet.parameters():
        param.requires_grad = False

    criterion2 = PerceptualLoss(lossnet)
    criterion2 = criterion2.to(device=device, dtype=torch.double)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path, 'loss.csv'), 'w+')

    log_dir = os.path.join(model_path, 'train.log')
    logger = initialize_logger(log_dir)

    last_loss = 10000
    patience = 5
    trigger_times = 0

    training_losses = []
    validation_losses = []
    epochs = []
    # plt.figure()

    plt.title('Training and validation losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    for epoch in range(start_epoch + 1, end_epoch):
        start_time = time.time()
        print("\nEpoch: " + str(epoch))
        train_loss, iteration, lr = train(train_data_loader, model, criterion1, criterion2, optimizer, iteration, init_lr, end_epoch, epoch)
        test_loss = validate(val_loader, model, criterion1)

        training_losses.append(train_loss)
        validation_losses.append(test_loss)
        epochs.append(epoch)
        plt.plot(epochs, training_losses,'b',label ='Training losses')
        plt.plot(epochs, validation_losses, 'g', label='Validation losses')
        if epoch == 1:
            plt.legend()

        plt.savefig(os.path.join(model_path,'loss_curves.png'))
        # Save model
        # if test_loss < record_test_loss:
        #    record_test_loss = test_loss
        #    save_checkpoint(model_path, epoch, iteration, model, optimizer)
        save_checkpoint(model_path, epoch, iteration, model, optimizer)
        # print loss
        end_time = time.time()
        epoch_time = end_time - start_time
        print("\nEpoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.15f Test Loss: %.15f " % (
        epoch, iteration, epoch_time, lr, train_loss, test_loss))
        # save loss
        record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss)
        logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " % (
        epoch, iteration, epoch_time, lr, train_loss, test_loss))

        # check if loss is increasing
        if test_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                break
        else:
            print('trigger times: 0')
            trigger_times = 0

        last_loss = test_loss


# Training
def train(train_data_loader, model, criterion1, criterion2, optimizer, iteration, init_lr, end_epoch, epoch):
    losses = AverageMeter()
    lr1 = init_lr

    for i, (images, labels) in enumerate(train_data_loader):

        #target = convert_1channel_to_3channel(labels.detach().cpu().numpy())
        #target = torch.from_numpy(target)

        labels = labels.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)

        images = Variable(images)
        labels = Variable(labels)

        lr1 = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=1.5)
        iteration = iteration + 1
        # Forward + Backward + Optimize
        optimizer.zero_grad()

        output = model(images)
        # if epoch == 5:
        #output1 = labels.detach().cpu().numpy()[0]
        #output1 = output1.transpose(1, 2, 0)
        # output1 = Image.fromarray(output1, 'RGB')
        # output1.show()
        #imshow(output1[:,:,0])
        # plt.show()
        #plt.savefig("image.jpg")

        loss1 = criterion1(output, labels)
        # output = output.detach().cpu().numpy()
        # output = output.astype('float')
        # output = torch.from_numpy(output)

        output = output.cuda(non_blocking=True)
        #target = target.cuda(non_blocking=True)
        output = Variable(output)
        output = output.double()
        labels = labels.double()
        #target = Variable(target)

        loss2 = criterion2(output, labels)
        loss = loss1 + loss2

        if iteration > 0 and iteration % 200 == 0:
            print(iteration, end=",")
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        #  record loss
        # print(loss.item())
        losses.update(loss.item())

    return losses.avg, iteration, lr1


# Validate
def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        #  record loss
        losses.update(loss.item())

    return losses.avg


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr * (1 - iteraion / max_iter) ** power

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
