# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import layers


def gateNU(inputs, out_dim):
    outputs = layers.fully_connected(inputs, out_dim // 2, activation_fn=tf.nn.relu)
    outputs = layers.fully_connected(outputs, out_dim, activation_fn=tf.nn.sigmoid)
    return outputs

def cosinSim(a, b):
    """
    :param a: (N,d)
    :param b: (N,d)
    :return: (N,1)
    """
    dot_product = tf.reduce_sum(tf.multiply(a, b), axis=1)  # (N,)

    # 计算每个向量的模长
    norm_a = tf.sqrt(tf.reduce_sum(tf.square(a), axis=1))
    norm_b = tf.sqrt(tf.reduce_sum(tf.square(b), axis=1))

    # 防止除以0的情况发生
    epsilon = 1e-8
    sim_weight = dot_product / (norm_a * norm_b + epsilon)
    sim_weight = tf.expand_dims(sim_weight, -1)  # (N,1)
    return sim_weight

class EncoderLayer(object):
    def __init__(self, num_heads=4, num_units=256, output_num_units=128, rate=0.1):
        self.num_heads = num_heads
        self.num_units = num_units
        self.output_num_units = output_num_units
        self.dropout_rate = rate

    def call(self, q, k, v, mask, training, attention_type='self_attention'):
        """
        :param q: (?, seq_len_q, d_q)
        :param k: (?, seq_len_k, d_k)
        :param v: (?, seq_len_k, d_k)
        :param mask: (?, seq_len_k)
        :param training:
        :param attention_type: self_attention || target_attention
        :return: (?, seq_len_q, output_num_units)
        """
        mha = MultiHeadAttention(self.num_heads, self.num_units, self.output_num_units)

        # (?, seq_len_q, output_num_units)
        attn_output, attention_weights = mha.call(q, k, v, mask, attention_type)

        # (batch_size, seq_len_q, output_num_units)
        ffn_output = point_wise_feed_forward_network(attn_output, self.output_num_units)
        ffn_output = tf.layers.dropout(ffn_output, rate=self.dropout_rate, training=training)

        # (batch_size, seq_len_q, output_num_units)
        out = attn_output + ffn_output

        return out, attention_weights


class MultiHeadAttention(object):
    def __init__(self, num_heads, num_units, output_num_units):
        """
        num_heads: 要分成的头数
        num_units: q和k线性映射后的维度
        output_num_units: v映射后的维度
        depth: q和k在每个头下面的维度
        output_depth:v在每个头下面的维度
        """
        self.num_heads = num_heads
        self.num_units = num_units
        self.output_num_units = output_num_units

        assert num_units % num_heads == 0
        assert output_num_units % num_heads == 0

        self.depth = num_units // num_heads
        self.output_depth = output_num_units // num_heads

    def split_heads(self, x, depth):
        """
        x [?, seq_len, num_units] num_heads*depth = num_units
        分拆最后一个维度num_units 到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        # [?, seq_len, input_num_units] -> [?, seq_len, num_head, depth]
        x = tf.reshape(x, (-1, tf.shape(x)[1], self.num_heads, depth))
        # [?, num_head, seq_len, depth]
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask, attention_type='self_attention'):
        """
        :param q: (?, seq_len_q, d_q)
        :param k: (?, seq_len_k, d_k)
        :param v: (?, seq_len_k, d_k)
        :param mask: (?, seq_len_k)
        :param attention_type: self_attention || target_attention
        :return: (?, seq_len_q, output_num_units)
        """

        # Linear projections
        with tf.name_scope("linear_projection"):
            # (?, seq_len_q, num_units)
            q = layers.fully_connected(q, self.num_units, activation_fn=tf.nn.relu)
            # (?, seq_len_k, num_units)
            k = layers.fully_connected(k, self.num_units, activation_fn=tf.nn.relu)
            # (?, seq_len_k, output_num_units)
            v = layers.fully_connected(v, self.output_num_units, activation_fn=tf.nn.relu)



        with tf.name_scope("split_heads"):
            q = self.split_heads(q, self.depth)         # (batch_size, num_heads, seq_len_q, depth)
            k = self.split_heads(k, self.depth)         # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, self.output_depth)  # (batch_size, num_heads, seq_len_k, output_depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, output_depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        with tf.name_scope("scaled_dot_product_attention"):
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v,
                                                                               mask,
                                                                               self.num_heads,
                                                                               need_qk_ln=True,
                                                                               atten_type=attention_type)

        # (?, num_heads, seq_len_q, output_depth) -> (?, seq_len_q, num_heads, output_depth) -> (?, seq_len_q, output_num_units)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        scaled_attention = tf.reshape(scaled_attention, (-1, tf.shape(q)[-2], self.output_num_units))

        # (?, seq_len_q, output_num_units)
        return scaled_attention, attention_weights


def scaled_dot_product_attention(q, k, v,
                                 mask=None,
                                 num_heads=1,
                                 need_qk_ln=True,
                                 atten_type='self_attention'
                                 ):
    """
    计算注意力权重
    q, k, v 必须具有匹配的前置维度。 ...为(batch_size, num_heads) num_heads=1即不进行多头处理
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v = max_len。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，这里取填充型mask
    但是 mask 必须能进行广播转换以便求和。
    参数:
      q: 请求的形状 == (..., seq_len_q, depth) 如果seq_len_q=1, 即为target_attention
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, output_depth)
      mask: bool 张量，(batch_size, seq_len_k)。默认为None。
      num_heads: 1代表没有进行多头操作
    返回值:
      输出，注意力权重 (..., seq_len_q, output_depth)
    """
    seq_len_q = tf.shape(q)[-2]
    seq_len_k = tf.shape(k)[-2]
    if mask is not None:
        mask = tf.cast(mask, tf.bool)

    with tf.name_scope("qk_ln"):
        if need_qk_ln:
            q = layers.layer_norm(q, begin_norm_axis=-1, begin_params_axis=-1)
            k = layers.layer_norm(k, begin_norm_axis=-1, begin_params_axis=-1)

    with tf.name_scope("matmul_qk"):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        # 缩放 matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)  # (..., seq_len_q, seq_len_k)

    with tf.name_scope("key_mask"):
        # 将 mask 加入到缩放的张量上, 确保q只和有效的k计算相似度, 即将 列 置为0，将无效k置为无穷小，是softmax的权重为0
        # 加无穷小的数经过softmax后权重会变接近0
        if mask is not None:
            # (batch_size, seq_len_k) -> (batch_size, 1, 1, seq_len_k) -> (batch_size, num_heads, seq_len_q, seq_len_k)
            key_masks = tf.reshape(mask, [-1, 1, 1, seq_len_k])
            key_masks = tf.tile(key_masks, [1, num_heads, seq_len_q, 1])
            paddings = tf.fill(tf.shape(key_masks), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
            scaled_attention_logits = tf.where(key_masks, scaled_attention_logits, paddings)

    with tf.name_scope("softmax"):
        # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
        # 相加等于1。
        attention_weights = tf.nn.softmax(scaled_attention_logits)  # (..., seq_len_q, seq_len_k)

    # 如果是self attention, 还得取q中有效的序列, 此时seq_len_q=seq_len_k， 即将 行 置为0, 将无效q的权重置为0
    with tf.name_scope("query_mask"):
        if atten_type == "self_attention":
            query_masks = mask   # (batch_size, seq_len_k)
            if query_masks is not None:
                # (batch_size, 1, seq_len_k, 1) -> (batch_size, num_head, seq_len_k, seq_len_k)
                query_masks = tf.reshape(query_masks, [-1, 1, seq_len_k, 1])
                query_masks = tf.tile(query_masks, [1, num_heads, 1, seq_len_k])
                paddings = tf.fill(tf.shape(query_masks), tf.constant(0, dtype=tf.float32))
                attention_weights = tf.where(query_masks, attention_weights, paddings)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, output_depth)

    return output, attention_weights


# 点式前馈网络由两层全联接层组成，两层之间有一个 ReLU 激活函数。
# 先放大，再缩小 inputs (?, seq_len, num_units)
def point_wise_feed_forward_network(inputs, num_units):
    # (?, seq_len, num_units*4)
    outputs = layers.fully_connected(inputs, num_units * 4, activation_fn=tf.nn.relu)
    # (?, seq_len, num_units)
    outputs = layers.fully_connected(outputs, num_units, activation_fn=None)

    return outputs
