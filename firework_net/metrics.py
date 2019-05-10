import torch
import math
import numpy as np

def log10(x):
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = target > 1e-5
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg


def print_train_metrics(epoch, idx, batch_num, lr, data_time, gpu_time, loss, result, average_meter):
    print('Train Epoch({0}):[{1}/{2}]'
          'Lr={lr:.6f} '
          'Loss={loss:.3f} '
          'RMSE={result.rmse:.3f}({average.rmse:.3f}) '
          'REL={result.absrel:.3f}({average.absrel:.3f}) '
          'Log10={result.lg10:.3f}({average.lg10:.3f}) '
          'D1={result.delta1:.3f}({average.delta1:.3f}) '
          'D2={result.delta2:.3f}({average.delta2:.3f}) '
          'D3={result.delta3:.3f}({average.delta3:.3f})'.format(
        epoch, idx, batch_num, lr=lr, data_time=data_time, loss=loss.item(),
        gpu_time=gpu_time, result=result, average=average_meter.average()))

def print_test_metrics(idx, batch_num, gpu_time, result, average_meter):
    print('Validate: [{0}/{1}]'
          'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
          'REL={result.absrel:.2f}({average.absrel:.2f}) '
          'Log10={result.lg10:.3f}({average.lg10:.3f}) '
          'D1={result.delta1:.3f}({average.delta1:.3f}) '
          'D2={result.delta2:.3f}({average.delta2:.3f}) '
          'D3={result.delta3:.3f}({average.delta3:.3f})'.format(
        idx, batch_num, result=result, average=average_meter.average()))

def save_log(logger, lr, cur_step, loss, result):

    logger.add_scalar('Learning_rate', lr, cur_step)
    logger.add_scalar('Train/Loss', loss.item(), cur_step)
    logger.add_scalar('Train/RMSE', result.rmse, cur_step)
    logger.add_scalar('Train/Rel', result.absrel, cur_step)
    logger.add_scalar('Train/Log10', result.lg10, cur_step)
    logger.add_scalar('Train/Delta1', result.delta1, cur_step)
    logger.add_scalar('Train/Delta2', result.delta2, cur_step)
    logger.add_scalar('Train/Delta3', result.delta3, cur_step)
