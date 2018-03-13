import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import graph_editor as ge
from tensorflow.contrib.distributions import Bernoulli

class SimultaneousPerturbationOptimizer(tf.train.Optimizer):

    def __init__(self, input_ops, graph=None, a=0.01, c=0.01, alpha=1.0, gamma=0.4, global_step=None, use_locking=False, name="SPSA_Optimizer"):
        super(SimultaneousPerturbationOptimizer, self).__init__(use_locking, name)
        self.input_ops = {op.name.encode('ascii','ignore'): op for op in input_ops}
        self.input_op_names = [str.split(n,':')[0] for n in self.input_ops.keys()]      # names of input ops wihtout :index
        self.work_graph = tf.get_default_graph() if graph is None else graph

        # clone model twice
        self.orig_graph_view = ge.sgv(self.work_graph)
        self.nsg, self.ninfo = self._clone_model(self.orig_graph_view, 'N_Model')
        self.psg, self.pinfo = self._clone_model(self.orig_graph_view, 'P_Model')

        # create parameter assignment
        print("trainable vars: ")
        for var in tf.trainable_variables():
            print var
            nvar = self.ninfo.transformed(var.op)
            print nvar
            ge.add_control_inputs(nvar, var.initializer)
            self._make_perturbations(var)



    # def _prepare(self):
    #     if self.global_step is None:
    #         self.global_step_tensor = tf.Variable(0, name='global_step', trainable=False)
    #     else:
    #         self.global_step_tensor = global_step
    #     self.a = tf.constant(a, dtype=tf.float32)
    #     self.c = tf.constant(c, dtype=tf.float32)
    #     self.alpha = tf.constant(alpha, dtype=tf.float32)
    #     self.gamma = tf.constant(gamma, dtype=tf.float32)
    #     self.a_t = self.a / tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32), tf.constant(1, dtype=tf.float32)), alpha)
    #     self.c_t = self.c / tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32), tf.constant(1, dtype=tf.float32)), gamma)

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

    def _make_perturbations(self, var):
        ''' add ops that generate the perturbations for var.
        '''
        # random = Bernoulli(tf.fill(var.get_shape(), 0.5), dtype=tf.float32)
        # delta = tf.subtract(tf.constant(1, dtype=tf.float32),
        #                     tf.scalar_mul(tf.constant(2, dtype=tf.float32), random.sample(1)[0] ))
        # c_t_delta = tf.scalar_mul(tf.reshape(self.c_t, []), delta)
        print var
        print var.op
        n_var = self.ninfo.transformed(var.op)
        print n_var
        #p_var = self.pinfo.transformed(var.op)
        # sg = ge.make_view(nvar)
        # print sg
        # assign = self.work_graph.get_operation_by_name(n_var.name+"/Assign")
        # print assign


        # n_var.initializer = var.initialized_value()
        # p_var.initializer = var.initialized_value()
        n_tensor = tf.get_default_graph().get_tensor_by_name(n_var.name + ':0')
        print n_tensor
        # p_tensor = tf.get_default_graph().get_tensor_by_name(p_var.name + ':0')
        # # W_n = tf.assign(n_tensor, tf.subtract(var, c_t_delta))
        # # W_p = tf.assign(p_tensor, tf.add(var, c_t_delta))
        op = tf.assign(n_tensor, var).op
        print op
        # W_p = tf.assign(p_tensor, var.initial_value).op
        # return []

    def minimize(self, loss, graph=None, var_list=None, global_step=None):
        # create perturbations
        # pops = []
        # for var in tf.trainable_variables():
        #     pops += self._make_perturbations(var, ninfo, pinfo)

        # create weight updates

        return self.ninfo.transformed(loss)
        #return control_flow_ops.group(*(nsg.ops + psg.ops), name='SPSA_Optimizer')
