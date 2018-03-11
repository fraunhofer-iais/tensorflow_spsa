import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import graph_editor as ge
from tensorflow.contrib.distributions import Bernoulli

class SimultaneousPerturbationOptimizer(tf.train.Optimizer):

    def __init__(self, input_ops, a=0.01, c=0.01, alpha=1.0, gamma=0.4, global_step=None, use_locking=False, name="SPSA_Optimizer"):
        super(SimultaneousPerturbationOptimizer, self).__init__(use_locking, name)
        self.input_ops = {op.name.encode('ascii','ignore'): op for op in input_ops}
        self.input_op_names = [str.split(n,':')[0] for n in self.input_ops.keys()]      # names of input ops wihtout :index
        if global_step is None:
            self.global_step_tensor = tf.Variable(0, name='global_step', trainable=False)
        else:
            self.global_step_tensor = global_step
        self.a = tf.constant(a, dtype=tf.float32)
        self.c = tf.constant(c, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.a_t = self.a / tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32), tf.constant(1, dtype=tf.float32)), alpha)
        self.c_t = self.c / tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32), tf.constant(1, dtype=tf.float32)), gamma)

    def _clone_model(self, model, dst_scope):
        ''' make a copy of model and connect the resulting sub-graph to
            the input ops of the original graph.
        '''
        def not_input_op_filter(op):
            return op.name not in self.input_op_names

        ops_without_inputs = ge.filter_ops(model.ops, not_input_op_filter)
        clone_sgv = ge.make_view(ops_without_inputs)
        input_replacements = {orig_t : self.input_ops[orig_t.name] for orig_t in clone_sgv.inputs}  # replace geph_ inputs with original inputs
        return ge.copy_with_input_replacements(clone_sgv, input_replacements, dst_scope=dst_scope)

    def _make_perturbations(self, var, ninfo, pinfo):
        ''' add sub-graph that generates the perturbations for var.
        '''
        vshape = var.get_shape()
        random = Bernoulli(tf.fill(vshape, 0.5), dtype=tf.float32)
        delta = tf.subtract(tf.constant(1, dtype=tf.float32),
                            tf.scalar_mul(tf.constant(2, dtype=tf.float32), random.sample(1)[0] ))
        c_t_delta = tf.scalar_mul(tf.reshape(self.c_t, []), delta)
        n_var = ninfo.transformed(var.op)
        p_var = pinfo.transformed(var.op)
        n_tensor = tf.get_default_graph().get_tensor_by_name(n_var.name + ':0')
        p_tensor = tf.get_default_graph().get_tensor_by_name(p_var.name + ':0')
        W_n = tf.assign(n_tensor, tf.subtract(var, c_t_delta))
        W_p = tf.assign(p_tensor, tf.add(var, c_t_delta))
        return [W_p, W_n]

    def minimize(self, loss, graph=None, var_list=None, global_step=None):
        if graph is None: graph = tf.get_default_graph()

        # clone model twice
        orig_graph = ge.sgv(graph)
        nsg, ninfo = self._clone_model(orig_graph, 'Negative_Perturbation')
        psg, pinfo = self._clone_model(orig_graph, 'Positive_Perturbation')

        # create perturbations
        pops = []
        for var in tf.trainable_variables():
            pops += self._make_perturbations(var, ninfo, pinfo)

        # create weight updates

        return control_flow_ops.group(*(nsg.ops + psg.ops + pops), name='SPSA_Optimizer')
