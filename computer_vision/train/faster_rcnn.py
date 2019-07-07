import os
import random
import time
from typing import List

import numpy as np
import pandas as pd
from keras.utils import generic_utils

from data_models.object_detection_models import ImageData
from object_detection.config import Config
from object_detection.faster_rcnn.neural_network import build_faster_rcnn
from object_detection.faster_rcnn.neural_network_components import rpn_to_roi, calc_iou
from object_detection.ground_truth_anchor_generator import get_anchor_gt, get_img_output_length
from keras import backend as K
import matplotlib.pyplot as plt


def train_faster_rcnn(config: Config, train_imgs: List[ImageData], class_mapping):
    class_count = config.class_count
    model_rpn, model_classifier, model_all = build_faster_rcnn(config, class_num=class_count)

    model_rpn, model_classifier, record_df = load_weights_for_models(config, model_rpn, model_classifier)

    data_gen_train = get_anchor_gt(train_imgs, config, get_img_output_length, mode='train')

    total_epochs = len(record_df)
    trained_epochs = len(record_df)

    img_num_per_epoch = 1000
    num_epochs = 40
    iter_num = 0

    total_epochs += num_epochs

    losses = np.zeros((img_num_per_epoch, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []

    if len(record_df) == 0:
        best_loss = np.Inf
    else:
        best_loss = np.min(record_df['curr_loss'])

    start_time = time.time()

    for epoch_num in range(num_epochs):
        progbar = generic_utils.Progbar(img_num_per_epoch)
        print('Epoch {}/{}'.format(trained_epochs + 1, total_epochs))
        trained_epochs += 1

        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == img_num_per_epoch and config.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    if mean_overlapping_bboxes == 0:
                        print(
                            'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
                X, Y, img_data, debug_img, debug_num_pos = next(data_gen_train)

                # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                loss_rpn = model_rpn.train_on_batch(X, Y)

                # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                P_rpn = model_rpn.predict_on_batch(X)

                # R: bboxes (shape=(300,4))
                # Convert rpn layer to roi bboxes
                R = rpn_to_roi(P_rpn[0], P_rpn[1], config, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,
                               max_boxes=300)

                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
                # Y1: one hot code for bboxes from above => x_roi (X)
                # Y2: corresponding labels and corresponding gt bboxes
                X2, Y1, Y2, IouS = calc_iou(R, img_data, config, class_mapping)

                # If X2 is None means there are no matching bboxes
                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                # Find out the positive anchors and negative anchors
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if config.num_rois > 1:
                    # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                    if len(pos_samples) < config.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, config.num_rois // 2,
                                                                replace=False).tolist()

                    # Randomly choose (num_rois - num_pos) neg samples
                    try:
                        selected_neg_samples = np.random.choice(neg_samples,
                                                                config.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples,
                                                                config.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    # Save all the pos and neg samples in sel_samples
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                # training_data: [X, X2[:, sel_samples, :]]
                # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                #  X                     => img_data resized image
                #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
                #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
                #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('final_cls', np.mean(losses[:iter_num, 2])),
                                ('final_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == img_num_per_epoch:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if config.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))
                        elapsed_time = (time.time() - start_time) / 60

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if config.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(config.model_path)

                    new_row = {'mean_overlapping_bboxes': round(mean_overlapping_bboxes, 3),
                               'class_acc': round(class_acc, 3),
                               'loss_rpn_cls': round(loss_rpn_cls, 3),
                               'loss_rpn_regr': round(loss_rpn_regr, 3),
                               'loss_class_cls': round(loss_class_cls, 3),
                               'loss_class_regr': round(loss_class_regr, 3),
                               'curr_loss': round(curr_loss, 3),
                               'elapsed_time': round(elapsed_time, 3),
                               'mAP': 0}

                    record_df = record_df.append(new_row, ignore_index=True)
                    record_df.to_csv(config.training_record_path, index=0)
                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

    print('Training complete, exiting.')
    draw_losses(config, trained_epochs)


def draw_losses(config: Config, epoch_total_count):
    record_df = pd.read_csv(config.training_record_path)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epoch_total_count), record_df['mean_overlapping_bboxes'], 'r')
    plt.title('mean_overlapping_bboxes')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, epoch_total_count), record_df['class_acc'], 'r')
    plt.title('class_acc')

    plt.show()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epoch_total_count), record_df['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, epoch_total_count), record_df['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epoch_total_count), record_df['loss_class_cls'], 'r')
    plt.title('loss_class_cls')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, epoch_total_count), record_df['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    plt.show()

    plt.plot(np.arange(0, epoch_total_count), record_df['curr_loss'], 'r')
    plt.title('total_loss')
    plt.show()


def load_weights_for_models(config, model_rpn, model_classifier):
    if not os.path.isfile(config.model_path):
        # If this is the begin of the training, load the pre-traind base network such as vgg-16
        try:
            print('This is the first time of your training')
            print('loading weights from {}'.format(config.base_net_weights))
            model_rpn.load_weights(config.base_net_weights, by_name=True)
            model_classifier.load_weights(config.base_net_weights, by_name=True)
        except:
            print('Could not load pretrained model weights. Weights can be found in the keras application folder \
                https://github.com/fchollet/keras/tree/master/keras/applications')
        record_df = pd.DataFrame(
            columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls',
                     'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
    else:
        # If this is a continued training, load the trained model from before
        print('Continue training based on previous trained model')
        print('Loading weights from {}'.format(config.model_path))
        model_rpn.load_weights(config.model_path, by_name=True)
        model_classifier.load_weights(config.model_path, by_name=True)

        # Load the records
        record_df = pd.read_csv(config.training_record_path)

        print('Already train {} batches'.format(len(record_df)))
    return model_rpn, model_classifier, record_df
