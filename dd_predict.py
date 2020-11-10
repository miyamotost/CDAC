import tensorflow as tf
import numpy as np
import os, warnings, sys, re
import time, pickle
import datetime
import dd_data_helpers
from cdac import ContextualDAC
from tensorflow.contrib import learn
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import cPickle, time, json
from embedding import Word2Vec
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.05, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("Training_Data", "./dataset/dd_datset_training.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("Test_Data", "./dataset/dd_datset_test.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("Pred_Data", "./dataset/dd_datset_validation.txt", "Data source for the positive data.")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# Training parameters
tf.flags.DEFINE_integer("embedding_char_dim", 16, "Dimensionality of entity embedding (default: 16)")
tf.flags.DEFINE_integer("embedding_entity_dim", 16, "Dimensionality of entity embedding (default: 16)")
tf.flags.DEFINE_integer("num_quantized_chars", 40, "num_quantized_chars")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, x_char, x_ib, x_pos, x_mtopic, x_features, x_spId, x_hub, y_text, handcraft, vocab, w2v, pos_vocab, train_dev_index, data_pred, class_label = dd_data_helpers.load_data_and_labels(FLAGS.Training_Data, FLAGS.Test_Data)

# Build vocabulary
max_document_length = max([len(x1.split(" ")) for x1 in x_text[0]]) #max_document_length = 25
char_length = 550 #char_length = max([len(x) for x in x_text[0]])
print("max_document_length:  ", max_document_length)
print("char_length:  ", char_length)
print("Vocabulary Size: {:d}".format(len(vocab)))
vocab_processor = np.zeros([len(x_text[0]), max_document_length])
x = dd_data_helpers.fit_transform(x_text[0], vocab_processor, vocab)

# for prediction
print("Predicting...")
x_text_pred, x_char_pred, x_ib_pred, x_pos_pred, x_mtopic_pred, x_features_pred, x_spId_pred, x_hub_pred, y_pred, handcraft_pred = dd_data_helpers.load_data_and_labels_for_pred(FLAGS.Pred_Data)

vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
x_pred = dd_data_helpers.fit_transform(x_text_pred[0], vocab_processor, vocab)
vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
x_utt1_pred = dd_data_helpers.fit_transform(x_text_pred[1], vocab_processor, vocab)
vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
x_utt2_pred = dd_data_helpers.fit_transform(x_text_pred[2], vocab_processor, vocab)
vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
x_utt3_pred = dd_data_helpers.fit_transform(x_text_pred[3], vocab_processor, vocab)

vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
x_pos0_pred = dd_data_helpers.fit_transform_pos(x_pos_pred[0], vocab_processor)
vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
x_pos1_pred = dd_data_helpers.fit_transform_pos(x_pos_pred[1], vocab_processor)
vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
x_pos2_pred = dd_data_helpers.fit_transform_pos(x_pos_pred[2], vocab_processor)
vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
x_pos3_pred = dd_data_helpers.fit_transform_pos(x_pos_pred[3], vocab_processor)

# Prediction
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = ContextualDAC(
            sequence_length=x.shape[1],
            num_classes=4,
            vocab_size=len(vocab),
            vocab_entity_size=len(pos_vocab),
            vocab_pos_size = 38,
            sequence_char_length= char_length,
            num_quantized_chars= FLAGS.num_quantized_chars,
            embedding_size=FLAGS.embedding_dim,
            embedding_entity_size=FLAGS.embedding_entity_dim,
            embedding_char_size = FLAGS.embedding_char_dim,
            embedding_pos_size = 16,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            speakerid_size=3
        )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Write vocabulary
        vocab_data = dict()
        vocab_data['data'] = vocab
        vocab_data['max_doc_len'] = max_document_length
        vocab_path =  os.path.join(out_dir, "vocab.json")
        handel = open(vocab_path, 'wb')
        json.dump(vocab_data,handel)
        handel.close()

        # Initialize all variables
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        ckpt = tf.train.get_checkpoint_state('./runs/1604966472/checkpoints')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("load " + last_model)
            saver.restore(sess, last_model)
        else:
            sess.run(tf.global_variables_initializer())
            exit()

        time_str = datetime.datetime.now().isoformat()

        feed_dict = {
            cnn.input_utt_pred: x_pred,
            cnn.input_utt1: x_utt1_pred,
            cnn.input_utt2: x_utt2_pred,
            cnn.input_utt3: x_utt3_pred,
            cnn.input_char: x_char_pred[0],
            cnn.input_char1: x_char_pred[1],
            cnn.input_char2: x_char_pred[2],
            cnn.input_ib1: x_ib_pred[0],
            cnn.input_ib2: x_ib_pred[1],
            cnn.input_pos0: x_pos0_pred,
            cnn.input_pos1: x_pos1_pred,
            cnn.input_pos2: x_pos2_pred,
            cnn.input_pos3: x_pos3_pred,
            cnn.input_spId0: x_spId_pred[0],
            cnn.input_spId1: x_spId_pred[1],
            cnn.input_spId2: x_spId_pred[2],
            cnn.input_spId3: x_spId_pred[3],
            cnn.input_hub0: x_hub_pred[0],
            cnn.input_hub1: x_hub_pred[1],
            cnn.input_hub2: x_hub_pred[2],
            cnn.input_mtp0: x_mtopic_pred[0],
            cnn.input_mtp1: x_mtopic_pred[1],
            cnn.input_mtp2: x_mtopic_pred[2],
            cnn.input_feat0: x_features_pred[0],
            cnn.input_feat1: x_features_pred[1],
            cnn.input_feat2: x_features_pred[2],
            cnn.input_feat3: x_features_pred[3],
            cnn.input_x_hand: handcraft_pred,
            cnn.input_y: y_pred,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }

        step, summaries, loss, predictions, true_labels, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.predictions, cnn.true_labels, cnn.accuracy], feed_dict)
        print("{}: loss {:g}, -acc {:g}".format(time_str, loss, accuracy))
