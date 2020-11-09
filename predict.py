import tensorflow as tf
import numpy as np
import os, warnings, sys, re
import time, pickle
import datetime
import data_helpers
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
tf.flags.DEFINE_string("Training_Data", "./dataset/swda_datset_training.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("Test_Data", "./dataset/swda_datset_test.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("Pred_Data", "./dataset/swda_datset_pred.txt", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("embedding_char_dim", 16, "Dimensionality of entity embedding (default: 16)")
tf.flags.DEFINE_integer("embedding_entity_dim", 16, "Dimensionality of entity embedding (default: 16)")
tf.flags.DEFINE_integer("num_quantized_chars", 40, "num_quantized_chars")
tf.flags.DEFINE_integer("evaluate_every", 50 , "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, x_char, x_ib, x_pos, x_mtopic, x_features, x_spId, x_hub, y_text, handcraft, vocab, w2v, pos_vocab, train_dev_index, data_pred, class_label = data_helpers.load_data_and_labels(FLAGS.Training_Data, FLAGS.Test_Data)

# Build vocabulary
max_document_length = max([len(x1.split(" ")) for x1 in x_text[0]])
print max_document_length
# max_document_length = 25
vocab_processor = np.zeros([len(x_text[0]), max_document_length])
x = data_helpers.fit_transform(x_text[0], vocab_processor, vocab)
vocab_processor = np.zeros([len(x_text[0]), max_document_length])
x_utt1 = data_helpers.fit_transform(x_text[1], vocab_processor, vocab)
vocab_processor = np.zeros([len(x_text[0]), max_document_length])
x_utt2 = data_helpers.fit_transform(x_text[2], vocab_processor, vocab)
vocab_processor = np.zeros([len(x_text[0]), max_document_length])
x_utt3 = data_helpers.fit_transform(x_text[3], vocab_processor, vocab)


x_shuf, x_utt1_shuf, x_utt2_shuf, x_utt3_shuf, x_char_shuf, x_char1_shuf, x_char2_shuf, x_ib1_shuf, x_ib2_shuf, x_pos0_shuf, x_pos1_shuf, x_pos2_shuf, x_pos3_shuf, x_spId0_shuf, x_spId1_shuf, x_spId2_shuf, x_spId3_shuf, x_hub0_shuf, x_hub1_shuf, x_hub2_shuf, x_mtp0_shuf, x_mtp1_shuf, x_mtp2_shuf, x_feat0_shuf, x_feat1_shuf, x_feat2_shuf, x_feat3_shuf, y_shuf, handcraft_shuf = \
x, x_utt1, x_utt2, x_utt3, x_char[0], x_char[1], x_char[2], x_ib[0], x_ib[1], x_pos[0], x_pos[1], x_pos[2], x_pos[3], x_spId[0], x_spId[1], x_spId[2], x_spId[3], x_hub[0], x_hub[1], x_hub[2], x_mtopic[0], x_mtopic[1], x_mtopic[2], x_features[0],  x_features[1], x_features[2], x_features[3], y_text, handcraft

vocab_processor = np.zeros([len(x_text[0]), max_document_length])
x_pos0_shuf = data_helpers.fit_transform_pos(x_pos0_shuf, vocab_processor)
vocab_processor = np.zeros([len(x_text[0]), max_document_length])
x_pos1_shuf = data_helpers.fit_transform_pos(x_pos1_shuf, vocab_processor)
vocab_processor = np.zeros([len(x_text[0]), max_document_length])
x_pos2_shuf = data_helpers.fit_transform_pos(x_pos2_shuf, vocab_processor)
vocab_processor = np.zeros([len(x_text[0]), max_document_length])
x_pos3_shuf = data_helpers.fit_transform_pos(x_pos3_shuf, vocab_processor)

#char_length = max([len(x) for x in x_text[0]])
char_length = 550
print "char_length:  ", char_length

offset = int(x_shuf.shape[0] * 0)
x_shuffled, x_utt1_shuffled, x_utt2_shuffled, x_utt3_shuffled, x_char_shuffled, x_char1_shuffled, x_char2_shuffled, x_ib1_shuffled, x_ib2_shuffled, x_pos0_shuffled, x_pos1_shuffled, x_pos2_shuffled, x_pos3_shuffled, x_spId0_shuffled, x_spId1_shuffled, x_spId2_shuffled, x_spId3_shuffled, x_hub0_shuffled, x_hub1_shuffled, x_hub2_shuffled, x_mtp0_shuffled, x_mtp1_shuffled, x_mtp2_shuffled, x_feat0_shuffled, x_feat1_shuffled, x_feat2_shuffled, x_feat3_shuffled, y_shuffled, handcraft_shuffled = \
    x_shuf[offset:], x_utt1_shuf[offset:], x_utt2_shuf[offset:], x_utt3_shuf[offset:], x_char_shuf[offset:], x_char1_shuf[offset:], x_char2_shuf[offset:], x_ib1_shuf[offset:], x_ib2_shuf[offset:], x_pos0_shuf[offset:], x_pos1_shuf[offset:], x_pos2_shuf[offset:], x_pos3_shuf[offset:], x_spId0_shuf[offset:], x_spId1_shuf[offset:], x_spId2_shuf[offset:], x_spId3_shuf[offset:], x_hub0_shuf[offset:], x_hub1_shuf[offset:], x_hub2_shuf[offset:], x_mtp0_shuf[offset:], x_mtp1_shuf[offset:], x_mtp2_shuf[offset:], x_feat0_shuf[offset:], x_feat1_shuf[offset:], x_feat2_shuf[offset:], x_feat3_shuf[offset:],y_shuf[offset:], handcraft_shuf[offset:]

# print "len(x): ",  x_shuffled.shape
print "train_dev_index: ", train_dev_index
dev_sample_index = -1 * ( x_shuffled.shape[0] - train_dev_index)

x_train, x_dev = x_shuffled[:train_dev_index], x_shuffled[dev_sample_index:]
x_utt1_train, x_utt1_dev = x_utt1_shuffled[:train_dev_index], x_utt1_shuffled[dev_sample_index:]
x_utt2_train, x_utt2_dev = x_utt2_shuffled[:train_dev_index], x_utt2_shuffled[dev_sample_index:]
x_utt3_train, x_utt3_dev = x_utt3_shuffled[:train_dev_index], x_utt3_shuffled[dev_sample_index:]

x_char_train, x_char_dev = x_char_shuffled[:train_dev_index], x_char_shuffled[dev_sample_index:]
x_char1_train, x_char1_dev = x_char1_shuffled[:train_dev_index], x_char1_shuffled[dev_sample_index:]
x_char2_train, x_char2_dev = x_char2_shuffled[:train_dev_index], x_char2_shuffled[dev_sample_index:]

x_spId0_train, x_spId0_dev = x_spId0_shuffled[:train_dev_index], x_spId0_shuffled[dev_sample_index:]
x_spId1_train, x_spId1_dev = x_spId1_shuffled[:train_dev_index], x_spId1_shuffled[dev_sample_index:]
x_spId2_train, x_spId2_dev = x_spId2_shuffled[:train_dev_index], x_spId2_shuffled[dev_sample_index:]
x_spId3_train, x_spId3_dev = x_spId3_shuffled[:train_dev_index], x_spId3_shuffled[dev_sample_index:]


x_hub0_train, x_hub0_dev = x_hub0_shuffled[:train_dev_index], x_hub0_shuffled[dev_sample_index:]
x_hub1_train, x_hub1_dev = x_hub1_shuffled[:train_dev_index], x_hub1_shuffled[dev_sample_index:]
x_hub2_train, x_hub2_dev = x_hub2_shuffled[:train_dev_index], x_hub2_shuffled[dev_sample_index:]

x_mtp0_train, x_mtp0_dev = x_mtp0_shuffled[:train_dev_index], x_mtp0_shuffled[dev_sample_index:]
x_mtp1_train, x_mtp1_dev = x_mtp1_shuffled[:train_dev_index], x_mtp1_shuffled[dev_sample_index:]
x_mtp2_train, x_mtp2_dev = x_mtp2_shuffled[:train_dev_index], x_mtp2_shuffled[dev_sample_index:]

x_feat0_train, x_feat0_dev = x_feat0_shuffled[:train_dev_index], x_feat0_shuffled[dev_sample_index:]
x_feat1_train, x_feat1_dev = x_feat1_shuffled[:train_dev_index], x_feat1_shuffled[dev_sample_index:]
x_feat2_train, x_feat2_dev = x_feat2_shuffled[:train_dev_index], x_feat2_shuffled[dev_sample_index:]
x_feat3_train, x_feat3_dev = x_feat3_shuffled[:train_dev_index], x_feat3_shuffled[dev_sample_index:]

x_ib1_train, x_ib1_dev = x_ib1_shuffled[:train_dev_index], x_ib1_shuffled[dev_sample_index:]
x_ib2_train, x_ib2_dev = x_ib2_shuffled[:dev_sample_index], x_ib2_shuffled[dev_sample_index:]

x_pos0_train, x_pos0_dev = x_pos0_shuffled[:train_dev_index], x_pos0_shuffled[dev_sample_index:]
x_pos1_train, x_pos1_dev = x_pos1_shuffled[:train_dev_index], x_pos1_shuffled[dev_sample_index:]
x_pos2_train, x_pos2_dev = x_pos2_shuffled[:train_dev_index], x_pos2_shuffled[dev_sample_index:]
x_pos3_train, x_pos3_dev = x_pos3_shuffled[:train_dev_index], x_pos3_shuffled[dev_sample_index:]

y_train, y_dev = y_shuffled[:train_dev_index], y_shuffled[dev_sample_index:]
handcraft_train, handcraft_dev = np.array(handcraft_shuffled[:train_dev_index]), np.array(handcraft_shuffled[dev_sample_index:])


print("Vocabulary Size: {:d}".format(len(vocab)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = ContextualDAC(
            sequence_length=x_train.shape[1],
            num_classes=42,
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
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

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
        ckpt = tf.train.get_checkpoint_state('./runs/1604419137/checkpoints')
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print "load " + last_model
            saver.restore(sess, last_model)
        else:
            sess.run(tf.global_variables_initializer())
            exit()

        # Generate batches
        # batches = data_helpers.batch_iter(list(zip(x_train, x_utt1_train, x_utt2_train, x_utt3_train, x_char_train, x_char1_train, x_char2_train, x_ib1_train, x_ib2_train, x_pos0_train, x_pos1_train, x_pos2_train, x_pos3_train, x_spId0_train, x_spId1_train, x_spId2_train, x_spId3_train, x_hub0_train, x_hub1_train, x_hub2_train, x_mtp0_train, x_mtp1_train, x_mtp2_train, x_feat0_train, x_feat1_train, x_feat2_train, x_feat3_train, handcraft_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        time_str = datetime.datetime.now().isoformat()

        feed_dict = {
            cnn.input_utt_pred: x_train[:5],
            cnn.input_utt1: x_utt1_train[:5],
            cnn.input_utt2: x_utt2_train[:5],
            cnn.input_utt3: x_utt3_train[:5],
            cnn.input_char: x_char_train[:5],
            cnn.input_char1: x_char1_train[:5],
            cnn.input_char2: x_char2_train[:5],
            cnn.input_ib1: x_ib1_train[:5],
            cnn.input_ib2: x_ib2_train[:5],
            cnn.input_pos0: x_pos0_train[:5],
            cnn.input_pos1: x_pos1_train[:5],
            cnn.input_pos2: x_pos2_train[:5],
            cnn.input_pos3: x_pos3_train[:5],
            cnn.input_spId0: x_spId0_train[:5],
            cnn.input_spId1: x_spId1_train[:5],
            cnn.input_spId2: x_spId2_train[:5],
            cnn.input_spId3: x_spId3_train[:5],
            cnn.input_hub0: x_hub0_train[:5],
            cnn.input_hub1: x_hub1_train[:5],
            cnn.input_hub2: x_hub2_train[:5],
            cnn.input_mtp0: x_mtp0_train[:5],
            cnn.input_mtp1: x_mtp1_train[:5],
            cnn.input_mtp2: x_mtp2_train[:5],
            cnn.input_feat0: x_feat0_train[:5],
            cnn.input_feat1: x_feat1_train[:5],
            cnn.input_feat2: x_feat2_train[:5],
            cnn.input_feat3: x_feat3_train[:5],
            cnn.input_x_hand: handcraft_train[:5],
            cnn.input_y: y_train[:5],
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }

        step, summaries, loss, predictions, true_labels, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.predictions, cnn.true_labels, cnn.accuracy], feed_dict)
        print("{}: loss {:g}, -acc {:g}".format(time_str, loss, accuracy))

        feed_dict = {
            cnn.input_utt_pred: x_dev[:15],
            cnn.input_utt1: x_utt1_dev[:15],
            cnn.input_utt2: x_utt2_dev[:15],
            cnn.input_utt3: x_utt3_dev[:15],
            cnn.input_char: x_char_dev[:15],
            cnn.input_char1: x_char1_dev[:15],
            cnn.input_char2: x_char2_dev[:15],
            cnn.input_ib1: x_ib1_dev[:15],
            cnn.input_ib2: x_ib2_dev[:15],
            cnn.input_pos0: x_pos0_dev[:15],
            cnn.input_pos1: x_pos1_dev[:15],
            cnn.input_pos2: x_pos2_dev[:15],
            cnn.input_pos3: x_pos3_dev[:15],
            cnn.input_spId0: x_spId0_dev[:15],
            cnn.input_spId1: x_spId1_dev[:15],
            cnn.input_spId2: x_spId2_dev[:15],
            cnn.input_spId3: x_spId3_dev[:15],
            cnn.input_hub0: x_hub0_dev[:15],
            cnn.input_hub1: x_hub1_dev[:15],
            cnn.input_hub2: x_hub2_dev[:15],
            cnn.input_mtp0: x_mtp0_dev[:15],
            cnn.input_mtp1: x_mtp1_dev[:15],
            cnn.input_mtp2: x_mtp2_dev[:15],
            cnn.input_feat0: x_feat0_dev[:15],
            cnn.input_feat1: x_feat1_dev[:15],
            cnn.input_feat2: x_feat2_dev[:15],
            cnn.input_feat3: x_feat3_dev[:15],
            cnn.input_x_hand: handcraft_dev[:15],
            cnn.input_y: y_dev[:15],
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }

        step, summaries, loss, predictions, true_labels, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.predictions, cnn.true_labels, cnn.accuracy], feed_dict)
        print("{}: loss {:g}, -acc {:g}".format(time_str, loss, accuracy))

        # for prediction
        print("Predicting...")
        x_text_pred, x_char_pred, x_ib_pred, x_pos_pred, x_mtopic_pred, x_features_pred, x_spId_pred, x_hub_pred, y_pred, handcraft_pred = data_helpers.load_data_and_labels_for_pred(FLAGS.Pred_Data)

        vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
        x_pred = data_helpers.fit_transform(x_text_pred[0], vocab_processor, vocab)
        vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
        x_utt1_pred = data_helpers.fit_transform(x_text_pred[1], vocab_processor, vocab)
        vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
        x_utt2_pred = data_helpers.fit_transform(x_text_pred[2], vocab_processor, vocab)
        vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
        x_utt3_pred = data_helpers.fit_transform(x_text_pred[3], vocab_processor, vocab)

        vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
        x_pos0_pred = data_helpers.fit_transform_pos(x_pos_pred[0], vocab_processor)
        vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
        x_pos1_pred = data_helpers.fit_transform_pos(x_pos_pred[1], vocab_processor)
        vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
        x_pos2_pred = data_helpers.fit_transform_pos(x_pos_pred[2], vocab_processor)
        vocab_processor = np.zeros([len(x_text_pred[0]), max_document_length])
        x_pos3_pred = data_helpers.fit_transform_pos(x_pos_pred[3], vocab_processor)

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
        print(predictions)


        print("Dev shape: {}, Train shape: {}, Pred shape {}".format(x_dev[:15].shape, x_train[:5].shape, x_pred[:15].shape))
