#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import itertools
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from copy import deepcopy

from .utils import fourierseries, _np_to_tensor, cache_data

class NNCDE(BaseEstimator):
    """
    Estimate univariate density using Bayesian Fourier Series.
    This method only works with data the lives in
    [0, 1], however, the class implements methods to automatically
    transform user inputted data to [0, 1]. See parameter `transform`
    below.

    Parameters
    ----------
    ncomponents : integer
        Maximum number of components of the Fourier series
        expansion.

    beta_loss_penal_exp : float
        Exponential term for penalizaing the size of beta's of the Fourier Series. This penalization occurs for training only (does not affect score method nor validation set if es=True).
    beta_loss_penal_base : float
        Base term for penalizaing the size of beta's of the Fourier Series. This penalization occurs for training only (does not affect score method nor validation set if es=True).
    nn_weight_decay : object
        Mulplier for penalizaing the size of neural network weights. This penalization occurs for training only (does not affect score method nor validation of early stopping).

    num_layers : integer
        Number of hidden layers for the neural network. If set to 0, then it degenerates to linear regression.
    hidden_size : integer
        Size of the hidden layers of the neural network.

    es : bool
        If true, then will split the training set into training and validation and calculate the validation internally on each epoch and check if the validation loss increases or not.
    es_validation_set_size : float, int
        Size of the validation set if es == True, given as proportion of train set or as absolute number. If None, then `round(min(x_train.shape[0] * 0.10, 5000))` will be used.
n_train = x_train.shape[0] - n_test
    es_give_up_after_nepochs : float
        Amount of epochs to try to decrease the validation loss before giving up and stoping training.
    es_splitter_random_state : float
        Random state to split the dataset into training and validation.

    nepoch : integer
        Number of epochs to run. Ignored if es == True.

    batch_initial : integer
        Initial batch size.
    batch_step_multiplier : float
        See batch_inital.
    batch_step_epoch_expon : float
        See batch_inital.
    batch_max_size : float
        See batch_inital.

    grid_size : integer
        Set grid size for calculating utility on score method.
    batch_test_size : integer
        Size of the batch for validation and score methods.
        Does not affect training efficiency, usefull when there's
        little GPU memory.
    gpu : bool
        If true, will use gpu for computation, if available.
    verbose : integer
        Level verbosity. Set to 0 for silent mode.
    """
    def __init__(self,
                 ncomponents=30,
                 beta_loss_penal_exp=0,
                 beta_loss_penal_base=0,
                 nn_weight_decay=0,
                 num_layers=10,
                 hidden_size=1000,
                 convolutional=False,

                 es = True,
                 es_validation_set_size = None,
                 es_give_up_after_nepochs = 50,
                 es_splitter_random_state = 0,

                 nepoch=200,

                 batch_initial=50,
                 batch_step_multiplier=1.1,
                 batch_step_epoch_expon=1.4,
                 batch_max_size=1000,

                 grid_size=1000,
                 batch_test_size=2000,
                 gpu=True,
                 verbose=1,
                 ):

        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    def fit(self, x_train, y_train):
        self.gpu = self.gpu and torch.cuda.is_available()

        #if self.divide_batch_max_size_by_nlayers:
        #    self.batch_max_size_c = self.batch_max_size // (
        #                                 self.num_layers + 1)
        #else:
        #    self.batch_max_size_c = self.batch_max_size

        self.ncomponents = int(self.ncomponents)
        max_penal = self.ncomponents ** self.beta_loss_penal_exp
        if max_penal > 1e4:
            new_val = np.log(1e4) / np.log(self.ncomponents)
            print("Warning: beta_loss_penal_exp is very large for "
                  "this amount of components (ncomponents).\n",
                  "Will automatically decrease it to", new_val,
                  "to avoid having the model blow up.")
            self.beta_loss_penal_exp = new_val

        self.x_dim = x_train.shape[1]
        self._construct_neural_net()
        self.epoch_count = 0

        if self.gpu:
            self.move_to_gpu()

        return self.improve_fit(x_train, y_train, self.nepoch)

    def move_to_gpu(self):
        self.neural_net.cuda()
        if hasattr(self, "phi_grid"):
            self.phi_grid = self.phi_grid.cuda()
        self.gpu = True

        return self

    def move_to_cpu(self):
        self.neural_net.cpu()
        if hasattr(self, "phi_grid"):
            self.phi_grid = self.phi_grid.cpu()
        self.gpu = False

        return self

    def improve_fit(self, x_train, y_train, nepoch):
        criterion = nn.MSELoss()

        assert(self.batch_initial >= 1)
        assert(self.batch_step_multiplier > 0)
        assert(self.batch_step_epoch_expon > 0)
        assert(self.batch_max_size >= 1)
        assert(self.batch_test_size >= 1)

        assert(self.beta_loss_penal_exp >= 0)
        assert(self.beta_loss_penal_base >= 0)
        assert(self.nn_weight_decay >= 0)

        assert(self.num_layers >= 0)
        assert(self.hidden_size > 0)

        inputv_train = np.array(x_train, dtype='f4')
        target_train = np.array(fourierseries(y_train,
            self.ncomponents), dtype='f4')

        range_epoch = range(nepoch)
        if self.es:
            es_validation_set_size = self.es_validation_set_size
            if es_validation_set_size is None:
                es_validation_set_size = round(
                    min(x_train.shape[0] * 0.10, 5000))
            splitter = ShuffleSplit(n_splits=1,
                test_size=es_validation_set_size,
                random_state=self.es_splitter_random_state)
            index_train, index_val = next(iter(splitter.split(x_train,
                y_train)))
            self.index_train = index_train
            self.index_val = index_val

            inputv_val = inputv_train[index_val]
            target_val = target_train[index_val]
            inputv_val = np.ascontiguousarray(inputv_val)
            target_val = np.ascontiguousarray(target_val)

            inputv_train = inputv_train[index_train]
            target_train = target_train[index_train]
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            self.best_loss_val = np.infty
            es_tries = 0
            range_epoch = itertools.count() # infty iterator

            batch_test_size = min(self.batch_test_size,
                                  inputv_val.shape[0])
            self.loss_history_validation = []

        batch_max_size = min(self.batch_max_size, inputv_train.shape[0])
        self.loss_history_train = []

        start_time = time.process_time()

        lr = 0.01
        optimizer = optim.Adamax(self.neural_net.parameters(), lr=lr,
                                 weight_decay=self.nn_weight_decay)
        es_penal_tries = 0
        for _ in range_epoch:
            batch_size = int(min(batch_max_size,
                self.batch_initial +
                self.batch_step_multiplier *
                self.epoch_count ** self.batch_step_epoch_expon))

            permutation = np.random.permutation(target_train.shape[0])
            inputv_train = inputv_train[permutation]
            target_train = target_train[permutation]
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            try:
                self.neural_net.train()
                self._one_epoch("train", batch_size, batch_test_size,
                                inputv_train, target_train, optimizer,
                                criterion, volatile=False)

                self.neural_net.eval()
                avloss = self._one_epoch("train", batch_size,
                              batch_test_size, inputv_train,
                              target_train, optimizer, criterion,
                              volatile=True)
                self.loss_history_train.append(avloss)

                if self.es:
                    self.neural_net.eval()
                    avloss = self._one_epoch("val", batch_size,
                        batch_test_size, inputv_val, target_val,
                        optimizer, criterion, volatile=True)
                    self.loss_history_validation.append(avloss)
                    if avloss <= self.best_loss_val:
                        self.best_loss_val = avloss
                        best_state_dict = self.neural_net.state_dict()
                        best_state_dict = deepcopy(best_state_dict)
                        es_tries = 0
                        if self.verbose >= 2:
                            print("This is the lowest validation loss",
                                  "so far.")
                    else:
                        es_tries += 1

                    if (es_tries == self.es_give_up_after_nepochs // 3
                        or
                        es_tries == self.es_give_up_after_nepochs // 3
                        * 2):
                        if self.verbose >= 2:
                            print("Decreasing learning rate by half.")
                        optimizer.param_groups[0]['lr'] *= 0.5
                        self.neural_net.load_state_dict(best_state_dict)
                    elif es_tries >= self.es_give_up_after_nepochs:
                        self.neural_net.load_state_dict(best_state_dict)
                        if self.verbose >= 1:
                            print("Validation loss did not improve after",
                                  self.es_give_up_after_nepochs, "tries.",
                                  "Stopping")
                        break

                self.epoch_count += 1
            except RuntimeError as err:
                #if self.epoch_count == 0:
                #    raise err
                if self.verbose >= 2:
                    print("Runtime error problem probably due to",
                           "high learning rate.")
                    print("Decreasing learning rate by half.")

                self._construct_neural_net()
                if self.gpu:
                    self.move_to_gpu()
                lr /= 2
                optimizer = optim.Adamax(self.neural_net.parameters(),
                    lr=lr, weight_decay=self.nn_weight_decay)
                self.epoch_count = 0

                continue
            except KeyboardInterrupt:
                if self.epoch_count > 0 and self.es:
                    print("Keyboard interrupt detected.",
                          "Switching weights to lowest validation loss",
                          "and exiting")
                    self.neural_net.load_state_dict(best_state_dict)
                break

        self._find_cut_low_density(x_train[index_val],
                                   y_train[index_val])

        elapsed_time = time.process_time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", elapsed_time, flush=True)

        return self

    def _find_cut_low_density(self, x_train, y_train, grid_size=100):
        if self.verbose >= 2:
            print("Looking for best cutpoint.")

        scores = np.empty(grid_size)
        cut_values = np.logspace(0, 4, grid_size)-1
        for i, cut_value in enumerate(cut_values):
            self.cut_low_density = cut_value
            scores[i] = self.score(x_train, y_train)
        self.cut_low_density = cut_values[np.argmax(scores)]

        if self.verbose >= 2:
            print("Best cutpoint is", self.cut_low_density)

        return self

    def _one_epoch(self, ftype, batch_train_size, batch_test_size,
                   inputv, target, optimizer, criterion, volatile):
        if volatile:
            c_phi_grid = self.phi_grid
        else:
            self._create_random_phi_grid()
            c_phi_grid = self.r_phi_grid
        with torch.set_grad_enabled(not volatile):
            if volatile:
                batch_size = batch_test_size
            else:
                batch_size = batch_train_size

            if ftype == "train":
                batch_show_size = batch_train_size
            else:
                batch_show_size = batch_test_size

            inputv = torch.from_numpy(inputv)
            target = torch.from_numpy(target)
            if self.gpu:
                inputv = inputv.pin_memory()
                target = target.pin_memory()

            loss_vals = []
            batch_sizes = []
            for i in range(0, target.shape[0] + batch_size, batch_size):
                if i < target.shape[0]:
                    inputv_next = inputv[i:i+batch_size]
                    target_next = target[i:i+batch_size]

                    if self.gpu:
                        inputv_next = inputv_next.cuda(async=True)
                        target_next = target_next.cuda(async=True)

                if i != 0:
                    batch_actual_size = inputv_this.shape[0]
                    if batch_actual_size != batch_size and not volatile:
                        continue

                    optimizer.zero_grad()
                    output = self.neural_net(inputv_this)

                    self._create_phi_grid()
                    output_grid = torch.mm(output, c_phi_grid)
                    output_grid = F.softplus(output_grid + 1)

                    normalizing = output_grid.mean(1)

                    loss1 = (output * target_this).sum(1)
                    loss1 = F.softplus(loss1 + 1)
                    loss1 = -2 * loss1 / normalizing
                    loss1 = loss1.mean()

                    loss2 = output_grid / normalizing[:,None]
                    loss2 = loss2 ** 2
                    loss2 = loss2.mean()

                    loss = loss1 + loss2

                    #alpha = min(0.1 * (self.epoch_count + 1) ** 2, 100)
                    #loss = (output * target_this).sum(1) + 1
                    #loss = F.softplus(loss, alpha)
                    #loss = torch.clamp(loss, 1e-30)
                    #loss = - loss.log().mean()

                    # Penalize on betas
                    if self.beta_loss_penal_base != 0 and not volatile:
                        penal = output ** 2
                        if self.beta_loss_penal_exp != 0:
                            aranged = (loss.data.new(
                                    range(1, self.ncomponents + 1))
                                ** self.beta_loss_penal_exp
                                )
                            penal = penal * aranged
                        penal = penal.mean()
                        penal = penal * self.beta_loss_penal_base
                        loss += penal

                    # Correction for last batch as it might be smaller
                    #if batch_actual_size != batch_size:
                    #    loss *= batch_actual_size / batch_size

                    np_loss = loss.data.cpu().numpy()
                    if np.isnan(np_loss):
                        raise RuntimeError("Loss is NaN")

                    loss_vals.append(np_loss)
                    batch_sizes.append(batch_actual_size)

                    if not volatile:
                        loss.backward()
                        optimizer.step()

                inputv_this = inputv_next
                target_this = target_next

            avgloss = np.average(loss_vals, weights=batch_sizes)
            if self.verbose >= 2 and not volatile:
                print("Finished training for epoch", self.epoch_count,
                      "now calculating statistics.", flush=True)
            if self.verbose >= 2 and volatile:
                print("Finished epoch", self.epoch_count,
                      "with batch size", batch_show_size,
                      "and", ftype + " loss",
                      avgloss, flush=True)

            return avgloss

    def score(self, x_test, y_test):
        with torch.no_grad():
            self.neural_net.eval()
            inputv = _np_to_tensor(np.ascontiguousarray(x_test))
            target = _np_to_tensor(fourierseries(y_test, self.ncomponents))

            if self.gpu:
                inputv = inputv.pin_memory()
                target = target.pin_memory()

            batch_size = min(self.batch_test_size, x_test.shape[0])

            loss_vals = []
            batch_sizes = []
            for i in range(0, target.shape[0] + batch_size, batch_size):
                if i < target.shape[0]:
                    inputv_next = inputv[i:i+batch_size]
                    target_next = target[i:i+batch_size]

                    if self.gpu:
                        inputv_next = inputv_next.cuda(async=True)
                        target_next = target_next.cuda(async=True)

                if i != 0:
                    output = self.neural_net(inputv_this)

                    self._create_phi_grid()
                    output_grid = torch.mm(output, self.phi_grid)
                    output_grid = F.softplus(output_grid + 1)

                    normalizing = output_grid.mean(1)

                    loss1 = (output * target_this).sum(1)
                    loss1 = F.softplus(loss1 + 1)
                    loss1 = loss1 / normalizing

                    indch = loss1 <= self.cut_low_density
                    loss1[indch] = 0

                    loss1 = -2 * loss1
                    loss1 = loss1.mean()

                    loss2 = output_grid / normalizing[:,None]
                    del(normalizing)

                    indch = loss2 <= self.cut_low_density
                    loss2[indch] = 0

                    loss2 = loss2 ** 2
                    loss2 = loss2.mean()

                    loss = loss1 + loss2

                    loss_vals.append(loss.data.cpu().numpy())
                    batch_sizes.append(inputv_this.shape[0])

                inputv_this = inputv_next
                target_this = target_next

            return -1 * np.average(loss_vals, weights=batch_sizes)

    def predict(self, x_pred):
        with torch.no_grad():
            self.neural_net.eval()
            inputv = _np_to_tensor(x_pred)
            self._create_phi_grid()
            target = self.phi_grid

            if self.gpu:
                inputv = inputv.cuda()
                target = target.cuda()

            #Normalize by min
            #x_output_pred = self.neural_net(inputv)
            #output_pred = torch.mm(x_output_pred, target)
            #output_pred += 1

            #output_pred = output_pred.data.cpu().numpy()

            #normalizer = np.min(output_pred, 1)
            #normalizer = torch.from_numpy(normalizer)
            #normalizer = torch.clamp(normalizer, max=0)
            #normalizer = normalizer.numpy()

            #output_pred = output_pred - normalizer[:, None]
            #output_pred /= output_pred.mean(1)[:, None]
            #return output_pred

            x_output_pred = self.neural_net(inputv)
            output_pred = torch.mm(x_output_pred, target)

            output_pred = F.softplus(output_pred + 1)
            output_pred /= output_pred.mean(1)[:,None]

            indch = output_pred <= self.cut_low_density
            output_pred[indch] = 0

            return output_pred.data.cpu().numpy()

    def change_grid_size(self, new_grid_size):
        self.grid_size = new_grid_size
        if hasattr(self, "phi_grid"):
            del(self.phi_grid)
            del(self.y_grid)
            self._create_phi_grid()

    def _create_phi_grid(self):
        if not hasattr(self, "phi_grid"):
            self.y_grid = np.linspace(0, 1, self.grid_size,
                                      dtype=np.float32)[1:-1]
            self.phi_grid = np.array(fourierseries(self.y_grid,
                                     self.ncomponents).T)
            self.phi_grid = _np_to_tensor(self.phi_grid)
            if self.gpu:
                self.phi_grid = self.phi_grid.cuda()

    def _create_random_phi_grid(self):
        if not hasattr(self, "phi_grid"):
            self.r_y_grid = np.linspace(0, 1, 1002,
                                        dtype=np.float32)[1:-1]
            self.r_y_grid += np.random.uniform(-4.9e-5, 4.9e-5, 1)
            self.r_phi_grid = np.array(fourierseries(self.r_y_grid,
                                       self.ncomponents).T)
            self.r_phi_grid = _np_to_tensor(self.r_phi_grid)
            if self.gpu:
                self.r_phi_grid = self.r_phi_grid.cuda()

    def _construct_neural_net(self):
        class NeuralNet(nn.Module):
            def __init__(self, x_dim, ncomponents, num_layers,
                         hidden_size, convolutional):
                super(NeuralNet, self).__init__()

                output_hl_size = int(hidden_size)
                self.dropl = nn.Dropout(p=0.5)
                self.convolutional = convolutional
                next_input_l_size = x_dim

                if self.convolutional:
                    next_input_l_size = 1
                    self.nclayers = 4
                    clayers = []
                    polayers = []
                    normclayers = []
                    for i in range(self.nclayers):
                        if next_input_l_size == 1:
                            output_hl_size = 16
                        else:
                            output_hl_size = 32
                        clayers.append(nn.Conv1d(next_input_l_size,
                            output_hl_size, kernel_size=5, stride=1,
                            padding=2))
                        polayers.append(nn.MaxPool1d(stride=1,
                            kernel_size=5, padding=2))
                        normclayers.append(nn.BatchNorm1d(output_hl_size))
                        next_input_l_size = output_hl_size
                        self._initialize_layer(clayers[i])
                    self.clayers = nn.ModuleList(clayers)
                    self.polayers = nn.ModuleList(polayers)
                    self.normclayers = nn.ModuleList(normclayers)

                    faked = torch.randn(2, 1, x_dim)
                    for i in range(self.nclayers):
                        faked = polayers[i](clayers[i](faked))
                    faked = faked.view(faked.size(0), -1)
                    next_input_l_size = faked.size(1)
                    del(faked)

                llayers = []
                normllayers = []
                for i in range(num_layers):
                    llayers.append(nn.Linear(next_input_l_size,
                                             output_hl_size))
                    normllayers.append(nn.BatchNorm1d(output_hl_size))
                    next_input_l_size = output_hl_size
                    self._initialize_layer(llayers[i])
                self.llayers = nn.ModuleList(llayers)
                self.normllayers = nn.ModuleList(normllayers)

                self.fc_last = nn.Linear(next_input_l_size, ncomponents)
                self._initialize_layer(self.fc_last)
                self.exp_decay = nn.Parameter(torch.Tensor([0.0]))
                self.base_decay = nn.Parameter(torch.Tensor([-5.0]))

                self.num_layers = num_layers
                self.ncomponents = ncomponents
                self.np_sqrt2 = np.sqrt(2)

            def _decay_x(self, x):
                exp_decay = - torch.exp(self.exp_decay)
                base_decay = torch.exp(self.base_decay) + 1

                decay = (x.data.new(range(1, self.ncomponents + 1))
                     * exp_decay)
                decay = base_decay ** decay

                return x * decay

            def forward(self, x):
                if self.convolutional:
                    x = x[:, None]
                    for i in range(self.nclayers):
                        fc = self.clayers[i]
                        fpo = self.polayers[i]
                        fcn = self.normclayers[i]
                        x = fcn(F.elu(fc(x)))
                        x = fpo(x)
                    x = x.view(x.size(0), -1)

                for i in range(self.num_layers):
                    fc = self.llayers[i]
                    fcn = self.normllayers[i]
                    x = fcn(F.elu(fc(x)))
                    x = self.dropl(x)
                x = self.fc_last(x)
                x = F.tanh(x) * self.np_sqrt2
                #x = self._decay_x(x)
                return x

            def _initialize_layer(self, layer):
                nn.init.constant_(layer.bias, 0)
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_normal_(layer.weight, gain=gain)

        self.neural_net = NeuralNet(self.x_dim, self.ncomponents,
                                    self.num_layers, self.hidden_size,
                                    self.convolutional)

    def __getstate__(self):
        d = self.__dict__.copy()
        if hasattr(self, "neural_net"):
            state_dict = self.neural_net.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].cpu()
            d["neural_net_params"] = state_dict
            del(d["neural_net"])

        #Delete phi_grid (will recreate on load)
        if hasattr(self, "phi_grid"):
            del(d["phi_grid"])
            d["y_grid"] = None

        return d

    def __setstate__(self, d):
        self.__dict__ = d

        if "neural_net_params" in d.keys():
            self._construct_neural_net()
            self.neural_net.load_state_dict(self.neural_net_params)
            del(self.neural_net_params)
            if self.gpu:
                if torch.cuda.is_available():
                    self.move_to_gpu()
                else:
                    self.gpu = False
                    print("Warning: GPU was used to train this model, "
                          "but is not currently available and will "
                          "be disabled "
                          "(renable with method move_to_gpu)")
        #Recreate phi_grid
        if "y_grid" in d.keys():
            del(self.y_grid)
            self._create_phi_grid()

        #Backward compatibility
        if "cut_low_density" in d.keys():
            self.cut_low_density = 0

class NNCDECached(NNCDE):
    def fit(self, x_train, y_train):
        cache = cache_data.cache
        if cache is None:
            raise("Must set cache first!")
            # return super.fit(x_train, y_train)

        fit_cached = cache(self.fit_cacheable, ignore=["self"])
        new_self = fit_cached(x_train, y_train, self.get_params())
        self.__dict__ = new_self.__dict__
        return self

    def fit_cacheable(self, x_train, y_train, params):
        return super().fit(x_train, y_train)

    def score(self, x_test, y_test):
        cache = cache_data.cache
        if cache is None:
            raise("Must set cache first!")
            # return super.score(x_train, y_train)

        score_cached = cache(self.score_cacheable, ignore=["self"])
        return score_cached(x_test, y_test, self.get_params())

    def score_cacheable(self, x_test, y_test, params):
        return super().score(x_test, y_test)
