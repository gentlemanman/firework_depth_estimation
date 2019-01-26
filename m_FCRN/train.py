from loader import *
from m_utils import *
import os
import time
from fcrn import FCRN
from tensorboardX import SummaryWriter
from datetime import datetime
import socket
from weights import load_weights
from m_utils import load_split, loss_mse, loss_huber, loss_mse_alone, update_lr
from metrics import AverageMeter, Result


output_dir = './run'
resume = False
model_path = './run/checkpoint-0.pth.tar'
#model_path = './run/model_best.pth.tar'

use_tensorflow = False # 加载官方参数，从tensorflow转过来
weights_file = "./model/NYU_ResNet-UpProj.npy"

dtype = torch.cuda.FloatTensor
num_epochs = 30
batch_size = 2  #测试集和验证集的batch_size设置成1方便保存
learning_rate = 1.0e-3
momentum = 0.9
weight_decay = 0.0005


def main():
    # 1.Load data
    print("Loading data...")
    # train_loader, test_loader = getNYUDataset()
    train_loader, test_loader = getFireWorkDataset(batch_size=batch_size)
    # 先把结果设置成最坏
    best_result = Result()
    best_result.set_to_worst()

    # 2.Load model
    if resume:
        # TODO
        # best result应当从保存的模型中读出来
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        # model_dict = checkpoint['model']
        # model.load_state_dict(model_dict)
        model = checkpoint['model']
        print("loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        del checkpoint  # 删除载入的模型
        #del model_dict
        print("加载已经保存好的模型")

    else:
        print("创建模型")
        model = FCRN(batch_size)
        if use_tensorflow:
            model.load_state_dict(load_weights(model, weights_file, dtype))
        start_epoch = 0

    model = model.cuda()
    # 3.Loss
    # 官方MSE
    # loss_fn = torch.nn.MSELoss()
    # 自定义MSE
    # loss_fn = loss_mse()
    # 论文的loss,the reverse Huber
    # loss_fn = loss_huber()
    loss_fn = loss_mse_alone()
    # 4.Optim
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    best_txt = os.path.join(output_dir, 'best.txt')
    log_path = os.path.join(output_dir, 'logs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    # 5.Train
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # TODO 调整学习率
        train(train_loader, model, loss_fn, optimizer, epoch, logger)

        result, img_merge = validate(test_loader, model, epoch, logger)
        is_best = result.absrel < best_result.absrel
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}\nrmse={:.3f}\nrel={:.3f}\nlog10={:.3f}\nd1={:.3f}\nd2={:.3f}\ndd31={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                               result.delta3,
                               result.gpu_time))
            if img_merge is not None:
                img_filename = output_dir + '/comparison_best.png'
                save_image(img_merge, img_filename)

        # 每个epoch保存检查点
        save_checkpoint({'epoch': epoch, 'model': model, 'optimizer': optimizer, 'best_result': best_result},
                        is_best, epoch, output_dir)
        print("模型保存成功\n")


# 在数据集上训练一个epoch
def train(train_loader, model, criterion, optimizer, epoch, logger):
    average_meter = AverageMeter()
    model.train()
    end = time.time()
    batch_num = len(train_loader)
    current_step = batch_num * batch_size * epoch

    len_train_loader = len(train_loader) # 训练集分了多少个bath-size
    for i, (input, target) in enumerate(train_loader):
        # 当前学习率,因为有预热过程
        lr = update_lr(optimizer, learning_rate, epoch, float((i + 1) / len_train_loader)) # 最后一个参数为学习率的增加比率

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
        data_time = time.time() - end
        current_step += input.data.shape[0]

        end = time.time()
        with torch.autograd.detect_anomaly():
            output = model(input)
            # print('output:', output.size())
            # os.system('pause')
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gpu_time = time.time() - end

        # 度量
        result = Result()
        result.evaluate(output.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % 10 == 0:
            print('Train Epoch: {0} [{1}/{2}]\t'
                  'learning_rate={lr:.8f} '
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={loss:.3f} '
                  'RMSE={result.rmse:.3f}({average.rmse:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                epoch, i + 1, batch_num, lr=lr, data_time=data_time, loss=loss.item(),
                gpu_time=gpu_time, result=result, average=average_meter.average()))

            logger.add_scalar('Learning_rate', lr, current_step)
            logger.add_scalar('Train/Loss', loss.item(), current_step)
            logger.add_scalar('Train/RMSE', result.rmse, current_step)
            logger.add_scalar('Train/Rel', result.absrel, current_step)
            logger.add_scalar('Train/Log10', result.lg10, current_step)
            logger.add_scalar('Train/Delta1', result.delta1, current_step)
            logger.add_scalar('Train/Delta2', result.delta2, current_step)
            logger.add_scalar('Train/Delta3', result.delta3, current_step)
        avg = average_meter.average()

# 每个epoch结束进行验证
def validate(val_loader, model, epoch, logger):
    average_meter = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
        data_time = time.time() - end

        with torch.no_grad():
            output = model(input)
        gpu_time = time.time() - end

        # 度量
        result = Result()
        result.evaluate(output.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))

        img_merge = None

        # 保存一些验证结果
        skip = 10
        rgb = input
        step = i // skip
        if i % skip == 0:
            img_path = output_dir + '/result'
            save_image(input, target, output, img_path, step)

        if (i + 1) % 10== 0:
            print('Validate: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'REL={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'Rel={average.absrel:.3f}\n'
          'Log10={average.lg10:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    logger.add_scalar('Test/rmse', avg.rmse, epoch)
    logger.add_scalar('Test/Rel', avg.absrel, epoch)
    logger.add_scalar('Test/log10', avg.lg10, epoch)
    logger.add_scalar('Test/Delta1', avg.delta1, epoch)
    logger.add_scalar('Test/Delta2', avg.delta2, epoch)
    logger.add_scalar('Test/Delta3', avg.delta3, epoch)
    return avg, img_merge

if __name__ == '__main__':
    main()