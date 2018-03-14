import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import graph_editor as ge
from tensorflow.contrib.distributions import Bernoulli

class SimultaneousPerturbationOptimizer(tf.train.Optimizer):

    def __init__(self, input_ops, graph=None, var_list=None, a=0.01, c=0.01, alpha=1.0, gamma=0.4, global_step=None, use_locking=False, name="SPSA_Optimizer"):
        super(SimultaneousPerturbationOptimizer, self).__init__(use_locking, name)
        self.input_ops = {op.name.encode('ascii','ignore'): op for op in input_ops}
        self.input_op_names = [str.split(n,':')[0] for n in self.input_ops.keys()]      # names of input ops wihtout :index
        self.work_graph = tf.get_default_graph() if graph is None else graph            # the graph to work on
        self.orig_graph_view = ge.sgv(self.work_graph)                                  # make view of original graph (before creating any other ops)
        self.global_step_tensor = tf.Variable(0, name='global_step', trainable=False) if global_step is None else global_step

        # optimizer parameters
        self.a = tf.constant(a, dtype=tf.float32)
        self.c = tf.constant(c, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)

        # create sub-graph computing perturbations
        self.n_perturbations, self.p_perturbations = self._make_perturbator()

        # copy model
        self.nsg, self.ninfo = self._clone_model(self.orig_graph_view, self.n_perturbations, 'N_Model')
        self.psg, self.pinfo = self._clone_model(self.orig_graph_view, self.p_perturbations, 'P_Model')


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
                nps[var_name] = tf.subtract(var, c_t_delta).op
                pps[var_name] = tf.add(var, c_t_delta).op
        return nps,pps


    def _make_updater(self):
        #self.a_t = self.a / tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32), tf.constant(1, dtype=tf.float32)), self.alpha)
        pass


    def _clone_model(self, model, perturbations, dst_scope):
        ''' make a copy of model and connect the resulting sub-graph to
            input ops of the original graph and parameter assignments by
            perturbator.
        '''
        def not_input_op_filter(op):
            return op.name not in self.input_op_names

        def not_placeholder_filter(op):
            return not op.type == 'Placeholder'

        def not_placeholder_trainvar_filter(op):
            if op.type == 'Placeholder':
                print "removing Placeholder {}".format(op.name)
                self.input_ops[op.name.encode('ascii','ignore')] = op
                return False
            for s in self.tvars:
                if op.name.startswith(s):
                    print "removing trainable Variable {}".format(op.name)
                    return False
            return True

        self.input_ops = {}
        self.tvars = []
        print("Trainable variables:")
        for var in tf.trainable_variables():
            var_name = var.name.split(':')[0]
            print var_name
            self.tvars.append(var_name)

        #ops_without_inputs = ge.filter_ops(model.ops, not_input_op_filter)
        #ops_without_inputs = ge.filter_ops(model.ops, not_placeholder_filter)
        ops_without_inputs = ge.filter_ops(model.ops, not_placeholder_trainvar_filter)
        #print ops_without_inputs
        init = self.work_graph.get_operation_by_name("init")
        ops_without_inputs.remove(init)
        clone_sgv = ge.make_view(ops_without_inputs)
        #input_replacements = {orig : self.input_ops[orig.name] for orig in clone_sgv.inputs}  # replacements for input_ops

        print "---------------------"
        print self.input_ops
        print "---------------------"
        print perturbations
        print "---------------------"
        input_replacements = {}
        for op in clone_sgv.inputs:
            op_name = op.name.split(':')[0]
            print op_name
            if op_name in self.input_ops.keys():
                input_replacements[op_name] = self.input_ops[op_name]
            elif op_name.endswith('/read') and op_name[:-5] in perturbations.keys():
                input_replacements[op_name] = perturbations[op_name[:-5]]
            else:
                raise KeyError("Unable to find substitute for {}".format(op_name))

        print input_replacements


        # for var in tf.trainable_variables():
        #     print "making replacement for {} with {}".format(var.name, perturbations[var.name])
        #     input_replacements[var] = perturbations[var.name]
        #     ops_without_inputs.remove(var.op)
        #
        # clone_sgv = ge.make_view(ops_without_inputs)    # make new view now without initializers
        return ge.copy_with_input_replacements(clone_sgv, input_replacements, dst_scope=dst_scope)


    def _make_perturbations(self, var):
        ''' add ops that generate the perturbations for var.
        '''
        n_var = self.ninfo.transformed(var.op)
        #p_var = self.pinfo.transformed(var.op)
        # sg = ge.make_view(nvar)
        # print sg
        # assign = self.work_graph.get_operation_by_name(n_var.name+"/Assign")
        # print assign

        # n_var.initializer = var.initialized_value()
        # p_var.initializer = var.initialized_value()
        n_tensor = tf.get_default_graph().get_tensor_by_name(n_var.name + ':0')
        # p_tensor = tf.get_default_graph().get_tensor_by_name(p_var.name + ':0')
        # # W_n = tf.assign(n_tensor, tf.subtract(var, c_t_delta))
        # # W_p = tf.assign(p_tensor, tf.add(var, c_t_delta))
        op = tf.assign(n_tensor, var).op
        init = self.work_graph.get_operation_by_name("init")
        ge.add_control_inputs(init, [op])
        # W_p = tf.assign(p_tensor, var.initial_value).op
        # return []

        # n_var = self.ninfo.transformed(var.op)
        # n_tensor = tf.get_default_graph().get_tensor_by_name(n_var.name + ':0')
        # op = tf.assign(n_tensor, var).op
        # init = self.work_graph.get_operation_by_name("init")
        # ge.add_control_inputs(init, [op])


    def minimize(self, loss, global_step=None):
        # create perturbations
        # pops = []
        # for var in tf.trainable_variables():
        #     pops += self._make_perturbations(var, ninfo, pinfo)

        # create weight updates

        return []
        #return self.ninfo.transformed(loss)
        #return control_flow_ops.group(*(nsg.ops + psg.ops), name='SPSA_Optimizer')
