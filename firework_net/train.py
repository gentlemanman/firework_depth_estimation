import os
import time
from datetime import datetime
import torch
from tensorboardX import SummaryWriter
from net.unet import UNet
from net.archs import NestedUNet, Arg
from loader import getFireWorkDataset
from metrics import Result, AverageMeter, print_train_metrics, print_test_metrics, save_log
from utils import loss_mse, update_lr, save_checkpoint, save_image, loss_smooth_l1, write_log

shape = '50' # 加载的当前数据集种类
resume = False
model_path = './run_log/checkpoint-16.pth.tar'
num_epoch = 30
batch_size = 2
run_log = './run_log'
learning_rate = 1.0e-3
momentum = 0.9

def main():
    train_loader, test_loader = getFireWorkDataset(batch_size=batch_size, shape=shape)
    print('数据加载成功')

    # 初始化时，把结果设置成最坏
    best_result = Result()
    best_result.set_to_worst()

    if resume:
        # best result应当从保存的模型中读出来
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        # del model_dict
        print("加载保存好的模型成功")
        # clear memory
        del checkpoint
        # del model_dict
        torch.cuda.empty_cache()
    else:
        # model = UNet(n_channels=1)
        arg = Arg()
        model = NestedUNet(arg)
        start_epoch = 0
        print('创建模型成功')

    if torch.cuda.is_available():
        model = model.cuda()

    # 损失函数和优化器
    # loss_fn = loss_mse()
    loss_fn = loss_smooth_l1()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # log
    global run_log_subdir
    run_log_subdir = os.path.join(run_log, datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(run_log_subdir)
    best_txt = os.path.join(run_log_subdir, 'best.txt')
    train_loader_txt = os.path.join(run_log_subdir, 'train_log.txt')
    log_path = os.path.join(run_log_subdir, 'logs')
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    # train
    for epoch in range(start_epoch, start_epoch + num_epoch):
        # 训练
        result = train(train_loader, model, loss_fn, optimizer, epoch, logger)
        write_log(train_loader_txt, result, shape, epoch)
        # 验证
        result = validate(test_loader, model, epoch, logger)
        is_best = result.absrel < best_result.absrel
        if is_best:
            best_result = result
            write_log(best_txt, result, shape, epoch)

        # 每个epoch保存检查点
        save_checkpoint({'epoch': epoch, 'model': model, 'optimizer': optimizer, 'best_result': best_result, 'shape':shape},
                        is_best, epoch, run_log_subdir)
        print('模型保存成功')



# 训练一个epoch
def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()
    average_meter = AverageMeter()
    end = time.time()
    batch_num = len(train_loader)
    cur_step = batch_num * batch_size * epoch

    lr = update_lr(optimizer, learning_rate, epoch)

    for i, (input, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
        data_time = time.time() - end
        cur_step += input.data.shape[0]
        end = time.time()

        # 剔除无效数据, 当target全都无效时需要跳过
        # if is_work_data(target.data) is False:
            # continue

        # 第0个epoch有一个预热的过程
        if epoch == 0:
            lr = update_lr(optimizer, learning_rate, epoch, float((i + 1) / batch_num))

        with torch.autograd.detect_anomaly():
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        gpu_time = time.time() - end

        # 保存一些中间图片
        if (i + 1) % 100 == 0:
            img_path = os.path.join(run_log_subdir, 'images')
            if os.path.exists(img_path) != True:
                os.mkdir(img_path)
            save_image(input, target, output, img_path, i + 1)

        # 度量
        result = Result()
        result.evaluate(output.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))

        if (i + 1) % 10 == 0:
            # 打印当前epoch
            print_train_metrics(epoch, i + 1, batch_num, lr, data_time, gpu_time, loss, result, average_meter)
            # 保存日志
            save_log(logger, lr, cur_step, loss, result)
    avg = average_meter.average()
    return avg


def validate(val_loader, model, epoch, logger):
    average_meter = AverageMeter()
    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()

        # 剔除无效数据, 当target全都无效时需要跳过
        if is_work_data(target.data) is False:
            continue

        data_time = time.time() - end
        with torch.no_grad():
            output = model(input)
        gpu_time = time.time() - end
        # 度量
        result = Result()
        result.evaluate(output.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))

        if (i + 1) % 10 == 0:
            print_test_metrics(i + 1, len(val_loader), gpu_time, result, average_meter)
    avg = average_meter.average()
    return avg


def is_work_data(data):
    # 有效像素，超过3%才进行
    threshold =  data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3] * 0.02
    valid_mask = (data > 1e-3)
    data = data[valid_mask]
    if data.shape[0] <= threshold:
        return False
    return True

if __name__ == '__main__':
    main()






