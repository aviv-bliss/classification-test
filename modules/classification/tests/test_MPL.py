import itertools
import os
import numpy as np

import dataloaders.img_transforms as transforms
from dataloaders.dataloader_builder import DataLoader
from metrics.metric_builder import Metric
from metrics.multiclass_accuracy import evaluate_nclass_metrics
from tests.test_builder import Test
from utils.auxiliary import AverageMeter, save_loss_to_resultstable, check_if_best_model_and_save, \
    load_model_and_weights, softmax
from utils.plot_ROC import plot_ROC

import torch
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))


class test_MPL(Test):
    def __init__(self, FLAGS):
        super(test_MPL, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.metric = FLAGS.metric
        self.model = FLAGS.model
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        if self.worker_num != None:
            self.results_table_path = os.path.join(self.train_dir, 'results_{}.csv'.format(self.worker_num))
        else:
            self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        self.mode = 'test'
        if hasattr(FLAGS, 'mode'):
            self.mode = FLAGS.mode
        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

        self.dataloader, num_samples = DataLoader.dataloader_builder(self.data_loader, self.FLAGS, self.mode)
        self.epoch_size = num_samples // self.batch_size


    def _test(self, model):
        losses = AverageMeter(precision=4)
        x, y, = [], []

        model.eval()

        for i, (input_image, output_image, score, _) in enumerate(self.dataloader):
            # transform to pytorch tensor
            totensor = transforms.Compose([transforms.ToTensor(), ])
            input_image_t = totensor(input_image)                                 # (batch_size, num_channels, height, width)
            output_image_t = totensor(output_image)                                 # (batch_size, num_channels, height, width)
            score_t = torch.from_numpy(score)
            score_t = score_t.type(torch.LongTensor)
            score_t = Variable(score_t)
            input_image_t = Variable(input_image_t)
            output_image_t = Variable(output_image_t)
            if use_cuda:
                input_image_t, output_image_t, score_t = input_image_t.cuda(), output_image_t.cuda(), score_t.cuda()

            # run test and get predictions
            output = model(input_image_t, output_image_t)


            loss = F.cross_entropy(output, score_t)
            losses.update(loss.data.item(), self.batch_size)
            x.append(output.data.cpu().numpy()[:])
            y.append(score[:])

            if i >= self.epoch_size - 1:
                break

        logits, acc, macro_precision, macro_recall, macro_f1 = evaluate_nclass_metrics(x, y, self.train_dir,
                                                                                             self.n_iter, self.n_epoch,
                                                                                             self.mode, self.debug)

        test_acc = Metric.metric_builder(self.metric, logits, y, self.FLAGS)

        print('\t\t\tTest (epoch {}, iter {}): Average loss: {:.4f}, {}: {:.2f}%\n'.format(self.n_epoch,
                                                    self.n_iter, losses.avg[0], self.metric, test_acc*100))
        return losses.avg[0], test_acc


    def build(self):
        self._check_args()

        # load models and weights
        models_loaded, models, model_names, self.n_iter, self.n_epoch = load_model_and_weights(self.load_ckpt,
                                    self.FLAGS, use_cuda, model=self.model, ckpts_dir=self.ckpts_dir, train=False,
                                    worker_num=self.worker_num)
        model = models[0]

        if models_loaded:
            # run test
            test_loss, test_acc = self._test(model)

            # save test losses and metrics to results table
            col_names = ['test_loss', 'test_acc']
            values = [test_loss, test_acc]
            save_loss_to_resultstable(values, col_names, self.results_table_path, self.n_iter, self.n_epoch, self.debug)

            # check if best model (saves best model not in debug mode)
            save_path = os.path.join(self.train_dir, 'ckpts')
            check_if_best_model_and_save(self.results_table_path, models, model_names, self.n_iter,
                                         self.n_epoch, save_path, self.debug, self.worker_num)
        else:
            print('no ckpt found for running test')
































