# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.python.ops import metrics
from tensorflow.contrib.framework.python.ops import variables as contrib_variables


from tensorflow.contrib.layers.python.layers.feature_column_ops import _input_from_feature_columns
from tensorflow.contrib.layers.python.layers import initializers
from module import *


class CRRN():

    def __init__(self):
        super(CRRN, self).__init__()

    def init(self, ctx):
        self.context = ctx
        self.logger = self.context.get_logger()
        self.config = self.context.get_config()
        self.mode = self.config.get_job_config("mode")
        self.sink = self.context.get_sink()
        self.fg = FgParser(self.config.get_fg_config())
        self.metrics = {}

        gl.init()
        gl.set_value('logger', self.logger)

        for (k, v) in self.config.get_all_algo_config().items():
            self.model_name = k
            self.algo_config = v
            self.opts_conf = v['optimizer']
            self.model_conf = v['modelx']

        if self.model_name is None:
            self.model_name = "CRRN"

    def build_graph(self, ctx, features, feature_columns, labels):
        self.logger.info("DwLog, build graph:: features :: " + str(features))
        self.logger.info("DwLog, build graph:: feature_columns :: " + str(feature_columns))

        self.set_global_step()
        self.inference(features, feature_columns)
        self.loss_esmm(labels)

    def set_global_step(self):
        """Sets up the global step Tensor."""
        with tf.name_scope("set_global_step"):
            self.global_step = training_util.get_or_create_global_step()
            self.global_step_reset = tf.assign(self.global_step, 0)
            self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
            tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def inference(self, features, feature_columns):
        self.feature_columns = feature_columns[self.model_name]
        self.features = features[self.model_name]
        with tf.name_scope("get_label"):
            # 加载获取label的值
            self.ctr_label = tf.identity(self.features["detail_label"])
            self.intent_label = tf.identity(self.features["tri_intent_label_pvid"])
            self.logger.info("DwLog : self.ctr_label " + str(self.ctr_label))
            self.logger.info("DwLog : self.intent_label " + str(self.intent_label))

            self.same_cate_label = tf.identity(self.features["is_same_cate"])

            # pairwise cate label
            is_same_cate = tf.identity(self.features["is_same_cate"])
            self.pairwise_rankloss_cate_label = tf.multiply(self.ctr_label, 2.0) + is_same_cate

        with tf.name_scope("get_batch_size"):
            self.batch_size = tf.shape(self.ctr_label)[0]

        self.interaction_layer()
        self.intention_layer()
        self.sim_layer()
        self.sequence_layer()
        self.dnn_layer()
        self.ctr_logits = self.ctr_main_net

    def interaction_layer(self):
        target_vec = self.layer_dict["item_id_columns"]  # (?, 48)
        trigger_vec = self.layer_dict["trigger_id_columns"]  # (?, 48)
        co_vec = self.layer_dict["co_action_common_columns"]
        weight_dim = target_vec.get_shape().as_list()[-1]
        intera_vec = gateNU(co_vec, weight_dim)
        self.intera_vec = tf.multiply(intera_vec, tf.multiply(target_vec, trigger_vec))


    def intention_layer(self):
        self.logger.info("XfLog, start intention_layer...")
        intent_features = []
        for block_name in set(self.user_column_block +
                              self.trigger_column_block
                              ):
            if block_name not in self.layer_dict:
                raise ValueError('[joint_features, layer dict] does not has block : {}'.format(block_name))
            self.logger.info(
                "XfLog, intent_features append emb, {} is {}".format(block_name, self.layer_dict[block_name]))
            intent_features.append(self.layer_dict[block_name])

        for block_name in self.sequence_layer_dict.keys():
            sequence_stack = self.sequence_layer_dict[block_name]  # (N,L,d)
            sequence_mean = tf.reduce_mean(sequence_stack, 1)  # (N,d)
            self.logger.info("XfLog, intent_features append seq mean, {} is {}".format(block_name, sequence_mean))
            intent_features.append(sequence_mean)

        intent_features = tf.concat(values=intent_features, axis=-1)
        self.logger.info("XfLog, intent_input_features is {}".format(intent_features))

        intent_features = layers.fully_connected(intent_features, 256, activation_fn=tf.nn.relu)
        intent_features = layers.fully_connected(intent_features, 128, activation_fn=tf.nn.relu)
        self.intent_feature_list.append(intent_features)
        self.intent_logits = layers.linear(intent_features, 1)
        self.intent_prob = tf.nn.sigmoid(self.intent_logits)  # (N,1)
        # self.intent_feature_list.append(self.intent_prob)

    def sim_layer(self):
        target_vec = self.layer_dict["item_id_columns"]  # (?, 48)
        trigger_vec = self.layer_dict["trigger_id_columns"]  # (?, 48)
        item_tri_cosin = cosinSim(target_vec, trigger_vec)
        self.sim_prob = tf.sigmoid(item_tri_cosin)  # (N,1)

    # 对序列进行进一步操作
    def sequence_layer(self):
        self.logger.info("DwLog, start sequence_layer...")



        for block_name in self.sequence_layer_dict.keys():
            with self.variable_scope(name_or_scope="{}_sequence_slat_{}".format(self.model_name, block_name)):
                max_len = self.fg.get_seq_len_by_sequence_name(block_name)

                sequence_stack = self.sequence_layer_dict[block_name]
                sequence_length = self.layer_dict[self.seq_column_len[block_name]]  # (?,1)
                sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)  # (?, max_len) bool

                with tf.name_scope("din_double"):
                    din_target_vec = self.layer_dict["item_id_columns"]  # (?, 48)
                    din_trigger_vec = self.layer_dict["trigger_id_columns"]  # (?, 48)
                    trigger_expand = tf.expand_dims(din_trigger_vec, 1)
                    target_expand = tf.expand_dims(din_target_vec, 1)

                    # trigger attention
                    q_trigger, k_trigger, v_trigger = trigger_expand, sequence_stack, sequence_stack
                    trigger_encoder = EncoderLayer(num_heads=4, num_units=256, output_num_units=128)
                    trigger_attention_output, trigger_attention_weights = trigger_encoder.call(q_trigger, k_trigger,
                                                                                               v_trigger,
                                                                                               sequence_mask,
                                                                                               self.is_training,
                                                                                               attention_type='target_attention')
                    # (N, L, 128)

                    # target attention
                    q_target, k_target, v_target = target_expand, trigger_attention_output, trigger_attention_output
                    target_encoder = EncoderLayer(num_heads=4, num_units=256, output_num_units=128)
                    target_attention_output, target_attention_weights = target_encoder.call(q_target, k_target,
                                                                                            v_target,
                                                                                            None,
                                                                                            self.is_training,
                                                                                            attention_type='target_attention')

                    # weighted
                    fuse_attention_output = tf.multiply(self.intent_prob * self.sim_prob, tf.squeeze(trigger_attention_output)) \
                                            + tf.multiply((tf.ones_like(self.intent_prob) - self.intent_prob) * (tf.ones_like(self.sim_prob) - self.sim_prob)
                                                          , tf.squeeze(target_attention_output))

                    fuse_attention_output = tf.reshape(fuse_attention_output, [-1, 128])

                    self.layer_dict["fuse_attention_output"] = fuse_attention_output
                    self.seq_block_name_list.append("fuse_attention_output")


    # main net
    def dnn_layer(self):
        self.main_net()

    def main_net(self):
        main_net_layer = []
        for block_name in set(self.user_column_block +
                              self.item_column_block +
                              self.trigger_column_block +
                              self.co_action_column_block +
                              self.context_colum_block +
                              self.seq_block_name_list
                              ):
            main_net_layer.append(self.layer_dict[block_name])

        main_net_layer = main_net_layer + self.intent_feature_list + [self.intera_vec]
        main_net = tf.concat(values=main_net_layer, axis=-1)
        with self.variable_scope(name_or_scope="{}_Main_Score_Network".format(self.model_name),
                                 partitioner=partitioned_variables.min_max_variable_partitioner(
                                     max_partitions=self.config.get_job_config("ps_num"),
                                     min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                 ):
            with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                dnn_hidden_units_list = self.model_conf['model_hyperparameter']['dnn_hidden_units']

                for layer_id, num_hidden_units in enumerate(dnn_hidden_units_list):
                    with self.variable_scope(
                            name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        main_net = layers.fully_connected(
                            main_net,
                            num_hidden_units,
                            getActivationFunctionOp(
                                self.model_conf['model_hyperparameter']['activation']),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm,
                            normalizer_params={"scale": True, "is_training": self.is_training})

                main_net = tf.layers.dense(main_net, 1, activation=None, name='deep')
                self.ctr_main_net = main_net


    def logits_layer(self, name):
        with self.variable_scope(name_or_scope="{}_{}_Logits".format(self.model_name, name)):
            if "CTR" in name:
                main_net = self.ctr_main_net
                bias_weight = "ctr_bias_weight"

            with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                main_logits = layers.linear(
                    main_net,
                    1,
                    scope="main_net",
                    variables_collections=[self.collections_dnn_hidden_layer],
                    outputs_collections=[self.collections_dnn_hidden_output],
                    biases_initializer=None)
                _logits = main_logits

                bias = contrib_variables.model_variable(
                    bias_weight,
                    shape=[1],
                    initializer=tf.zeros_initializer(),
                    trainable=True)

                logits = nn_ops.bias_add(_logits, bias)

                return logits

    def loss(self, logits, label):
        with tf.name_scope("{}_Loss_Op".format(self.model_name)):
            self.label = label
            self.logits = logits
            self.reg_loss_f()
            self.loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.label)) + self.reg_loss
            return self.loss_op


    def _rank_loss(self, logits, labels):
        pairwise_logits = logits - tf.transpose(logits)
        logging.info("[rank_loss] pairwise logits: {}".format(pairwise_logits))
        pairwise_mask = tf.greater(labels - tf.transpose(labels), 0)
        logging.info("[rank_loss] mask: {}".format(pairwise_mask))

        logging.info("[rank loss] use pvid mask.")
        # rn = tf.sparse_tensor_to_dense(self.features['id'], default_value='0')
        rn = tf.sparse_tensor_to_dense(self.features['pvid'], default_value='0')
        rn_mask = tf.equal(rn, tf.transpose(rn))
        pairwise_mask = tf.logical_and(rn_mask, pairwise_mask)
        pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
        logging.info("[rank_loss]: after masking: {}".format(pairwise_logits))
        pairwise_psudo_labels = tf.ones_like(pairwise_logits)
        rank_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pairwise_logits,
            labels=pairwise_psudo_labels
        ))
        # set rank loss to zero if a batch has no positive sample.
        rank_loss = tf.where(tf.is_nan(rank_loss), tf.zeros_like(rank_loss), rank_loss)
        return rank_loss

    def loss_esmm(self, label):
        self.label = label
        ctr_logits = self.ctr_logits
        ctr_label = self.ctr_label

        intent_prob = self.intent_prob
        intent_label = self.intent_label

        sim_label = self.same_cate_label

        with tf.name_scope("{}_Loss_Op".format(self.model_name)):
            ctr_prob = tf.sigmoid(ctr_logits)
            with tf.name_scope("ctr_loss"):
                self.ctr_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_label, logits=ctr_logits)
                )
            with tf.name_scope("intent_loss"):
                self.intent_loss = tf.reduce_mean(
                    tf.losses.log_loss(labels=intent_label, predictions=intent_prob)
                )

            with tf.name_scope("sim_loss"):
                self.sim_loss = tf.reduce_mean(
                    tf.losses.log_loss(labels=sim_label, predictions=self.sim_prob)
                )

            with tf.name_scope("rank_loss"):

                self.pairwise_cate_rankloss = self._rank_loss(ctr_prob, self.pairwise_rankloss_cate_label)


            with tf.name_scope("loss_sum"):
                self.loss_op = self.ctr_loss + 0.5 * self.intent_loss + 0.2 * self.pairwise_cate_rankloss

            return self.loss_op

