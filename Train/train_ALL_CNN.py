# coding=utf-8
# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : train_ALL_LSTM.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils

import random
from DataUtils.Common import seed_num
from DataUtils.Evaluate import evaluate_b
torch.manual_seed(seed_num)
random.seed(seed_num)



def train(train_iter, dev_iter, test_iter, model, args):
    if args.cuda:
        model.cuda()

    if args.Adam is True:
        print("Adam Training......")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
    elif args.SGD is True:
        print("SGD Training.......")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay,
                                    momentum=args.momentum_value)
    elif args.Adadelta is True:
        print("Adadelta Training.......")
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)

    model_count = 0
    best_accuracy = Best_Result()
    model.train()
    for epoch in range(1, args.epochs+1):
        steps = 0
        print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda is True:
                feature, target = feature.cuda(), target.cuda()

            target = autograd.Variable(target)  # question 1
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target, weight=torch.tensor([1,125]).float().cuda())
            loss.backward()
            if args.init_clip_max_norm is not None:
                utils.clip_grad_norm(model.parameters(), max_norm=args.init_clip_max_norm)
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                train_size = len(train_iter.dataset)
                y_pred = torch.max(logit, 1)[1].cpu().data.numpy().tolist()
                try:
                    result=evaluate_b(target.cpu().data.numpy().tolist(),y_pred)
                    sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  g-measure: {:.2f})'.format(steps,train_size,loss.item(),result[-1]))
                except Exception:
                    # print('\nValueError: Number of classes, 1, does not match size of target_names, 2')
                    print('\n')
            if steps % args.test_interval == 0:
                print("\nDev  Accuracy: ", "end=")
                eval(dev_iter, model, args, best_accuracy, epoch, test=False)
                print("\nTest Accuracy: ", "end=")
                eval(test_iter, model, args, best_accuracy, epoch, test=True)
            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model.state_dict(), save_path)
                if os.path.isfile(save_path) and args.rm_model is True:
                    os.remove(save_path)
                model_count += 1
    return model_count


def eval(data_iter, model, args, best_accuracy, epoch, test=False):
    model.eval()
    y_pred = []
    y_true = []
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align

        if args.cuda is True:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, weight=torch.tensor([1,125]).float().cuda(), size_average=True)

        avg_loss += loss.item()
        y_pred.extend(torch.max(logit, 1)[1].cpu().data.numpy().tolist())
        y_true.extend(target.cpu().numpy().tolist())
    result = evaluate_b(y_true, y_pred)
    g_measure=result[-1]
    size = len(data_iter.dataset)
    avg_loss = loss.item()/size
    model.train()
    print(' Evaluation - loss: {:.6f}  g-measure: {:.2f} '.format(avg_loss, g_measure))
    if test is False:
        if g_measure >= best_accuracy.best_dev_gmeasure:
            best_accuracy.best_dev_gmeasure = g_measure
            best_accuracy.best_epoch = epoch
            best_accuracy.best_test = True
    if test is True and best_accuracy.best_test is True:
        best_accuracy.gmeasure = g_measure

    if test is True:
        print("The Current Best Dev g-measure: {:.2f}, and Test g-measure is :{:.2f}, locate on {} epoch.\n".format(
            best_accuracy.best_dev_gmeasure, best_accuracy.gmeasure, best_accuracy.best_epoch))
    if test is True:
        best_accuracy.best_test = False

class Best_Result:
    def __init__(self):
        self.best_dev_gmeasure = -1
        self.best_gmeasure = -1
        self.best_epoch = 1
        self.best_test = False
        self.gmeasure = -1


