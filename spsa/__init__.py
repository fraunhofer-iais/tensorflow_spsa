import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import graph_editor as ge
from tensorflow.contrib.distributions import Bernoulli

class SimultaneousPerturbationOptimizer(tf.train.Optimizer):

    def __init__(self, graph=None, var_list=None, a=0.01, c=0.01, alpha=1.0, gamma=0.4, global_step=None, use_locking=False, name="SPSA_Optimizer"):
        super(SimultaneousPerturbationOptimizer, self).__init__(use_locking, name)
        self.work_graph = tf.get_default_graph() if graph is None else graph                                # the graph to work on
        self.orig_graph_view = ge.sgv(self.work_graph)                                                      # make view of original graph (before creating any other ops)
        self.tvars = [var.name.encode('ascii','ignore').split(':')[0] for var in tf.trainable_variables()]  # list of names of trainable variables
        self.global_step_tensor = tf.Variable(0, name='global_step', trainable=False) if global_step is None else global_step

        # optimizer parameters
        self.a = tf.constant(a, dtype=tf.float32)
        self.c = tf.constant(c, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)

        # create sub-graph computing perturbations
        n_perturbations, p_perturbations = self._make_perturbator()

        # copy model
        self.nsg, self.ninfo = self._clone_model(self.orig_graph_view, n_perturbations, 'N_Model')
        self.psg, self.pinfo = self._clone_model(self.orig_graph_view, p_perturbations, 'P_Model')


    def _make_perturbator(self):
        nps = {}
        pps = {}
        with tf.name_scope("Perturbator"):
            self.c_t = self.c / tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32), tf.constant(1, dtype=tf.float32)), self.gamma)
            for var in tf.trainable_variables():
                random = Bernoulli(tf.fill(var.get_shape(), 0.5), dtype=tf.float32)
                delta = tf.subtract(tf.constant(1, dtype=tf.float32),
                                    tf.scalar_mul(tf.constant(2, dtype=tf.float32), random.sample(1)[0] ))
                c_t_delta = tf.scalar_mul(tf.reshape(self.c_t, []), delta)
                var_name = var.name.encode('ascii','ignore').split(':')[0]
                nps[var_name+'/read:0'] = tf.subtract(var, c_t_delta)
                pps[var_name+'/read:0'] = tf.add(var, c_t_delta)
        return nps,pps


    def _clone_model(self, model, perturbations, dst_scope):
        ''' make a copy of model and connect the resulting sub-graph to
            input ops of the original graph and parameter assignments by
            perturbator.
        '''
        def not_placeholder_or_trainvar_filter(op):
            if op.type == 'Placeholder':
                return False
            for var_name in self.tvars:
                if op.name.startswith(var_name):            # remove Some/Var/(read,assign,...)
                    return False
            return True

        ops_without_inputs = ge.filter_ops(model.ops, not_placeholder_or_trainvar_filter)
        try:
            init = self.work_graph.get_operation_by_name("init")        # remove init op from clone if already present
            ops_without_inputs.remove(init)
        except:
            pass
        clone_sgv = ge.make_view(ops_without_inputs)
        clone_sgv = clone_sgv.remove_unused_ops(control_inputs=True)

        input_replacements = {}
        for t in clone_sgv.inputs:
            if t.name in perturbations.keys():                  # input from trainable var --> replace with perturbation
                input_replacements[t] = perturbations[t.name]
            else:                                               # otherwise take input from original graph
                input_replacements[t] = self.work_graph.get_tensor_by_name(t.name)
        return ge.copy_with_input_replacements(clone_sgv, input_replacements, dst_scope=dst_scope)


    def minimize(self, loss, global_step=None):
        return (self.ninfo.transformed(loss), self.pinfo.transformed(loss))
