import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from datetime import date
import time

def multinomialSampler(Nc, M, CD_min, CD_max):
    """A multinomial sampler with a temperatured step function, making sampling of classes/cell types go from equally likely to more extreme (singleton-like).

    :param Nc: The number of cell types. Usually in the range of 5-50.
    :type Nc: int
    :param M: The number of profiles generated, for each `CD`.  see `CD_min` and `CD_max` for more information.
    :type M: int
    :param CD_min: CD is cell density, and it is the measure of how many cells contribute to a particular profile. `CD_min` is the miniumum number of cells contributing to a profile, and together with `CD_max` they form a range of CDs, going from `CD_min` to `CD_max`.
    :type CD_min: int
    :param CD_max: CD is cell density, and it is the measure of how many cells contribute to a particular profile. `CD_max` is the maximum number of cells contributing to a profile, and together with `CD_min` they form a range of CDs, going from `CD_min` to `CD_max`.
    :type CD_max: int
    :return: Return a list of profiles, with the number of profiles equal to `Nc x M x (CD_max - CD_min + 1)`. Each profile contains a count value (positive integer, including 0) for each class/cell type.
    :rtype: List
    """
    profiles=[]
    # Sample across cell densites (S)
    for S in range(CD_min, CD_max+1):
        for temp_ in range(M):
            # Assign temperatures
            temps = []
            for t in range(Nc):
                power = (S**t)**((temp_+1)/(M)) # temp_ goes from 0 to M-1
                temps.append(power)
            temps = np.array(temps)/np.sum(temps) # Scale temperatures to 1
            # Extract M samples for current temperature step
            profile = np.random.multinomial(S, temps)
            # shuffle weights among cell types indices
            np.random.shuffle(profile)
            profiles.append(profile)
    return profiles

def getConvolutedProfilesFromDistributions(adata, cell_types, cell_type_key, distributions, normalize_X=False):
    """A function that converts the profiles generated with `multinomialSampler`, into gene-based profiles by sampling cells from the `SC` dataset, corresponding to the number of counts found in each profile.

    :param adata: An AnnData object, this is usually the `SC` dataset, in the experiment class `DeconvolutionExperiment`.
    :type adata: AnnData
    :param cell_types: A ordered list of cell types, found in `adata`s `cell_type_key`.
    :type cell_types: List
    :param cell_type_key: The key/column found in `adata`, the method will look for in the observations (`obs`) data frame.
    :type cell_type_key: str
    :param distributions: The profiles that should be processed to be convoluted, usually generated using `multinomialSampler`.
    :type distributions: [ParamType]
    :param normalize_X: If `True`, each convoluted profile is scaled to sum to 1 (assuming cell types already are scaled to 1), defaults to False.
    :type normalize_X: bool (False, optional)
    :return: A dict containing three lists. `X_list`, a gene-based list of convoluted profiles, each profile is a list of genes. `Y_list`, a list of cell types used to produce `X_list`, each element is class-based list. `I_list`, a list of indicies, to traceback what cells were used to generate the `X_list`. Each list is index-based related, so the first element of `X_list` is related to the first element of `Y_list` and the first element of `I_list`.
    :rtype: Dict
    """
    # make a copy of the adata
    adata_copy = adata.copy()
    # pre subset cell_types save expensive operations
    cell_types_cache = {}
    desified_X_cache = {}
    for cell_type in cell_types:
        cell_types_cache[cell_type] = adata_copy[adata_copy.obs[cell_type_key] == cell_type,:]
        desified_X_cache[cell_type] = cell_types_cache[cell_type].X.toarray()

    # sample Xs and Ys, and record indices used to redo the profiles
    X_list = []
    Y_list = []
    I_list = [] # indices
    for dist_ in distributions:
        cur_x = []
        cur_y = []
        cur_I = []
        for i in range(len(dist_)):
            type_ = cell_types[i]
            amount = dist_[i]
            #print(i, type_, amount)

            indices = np.random.choice(cell_types_cache[type_].n_obs, amount)
            #print("indices:", indices)
            for index in indices:
                cur_x.append(desified_X_cache[type_][index,:])
                cur_y.append(type_)
                cur_I.append([type_, index]) # use two keys for indexing based on cached cell-type


        # scale x by counts in y
        cur_x = np.sum(cur_x, axis=0) # only sum (this is discarded) -> /len(cur_y)
        # normalize X to the same scale, independant of cell density
        if normalize_X:
            cur_x /= len(cur_y)


        # convert to numpy
        cur_x = np.array(cur_x)
        cur_y = np.array(cur_y)

        # add x and y
        X_list.append(cur_x)
        Y_list.append(cur_y)
        I_list.append(cur_I)

    # return all three lists
    return (X_list, Y_list, I_list)


def getProportionFromCountVector(Y_list):
    """A function that will convert the count vectors into proportions. This is used to go from count vectors of cell types to proportions of cell types. Each profile will sum to 1.

    :param Y_list: Converts count profiles to proportion profiles.
    :type Y_list: List
    :return: A list of proportion profiles.
    :rtype: List
    """
    ret_list = [] # return list
    for i in range(len(Y_list)):
        # this is a single profile
        cur_elem = Y_list[i]
        # total count of the profile
        total_cells = np.sum(cur_elem)
        # element-wise scale to proportion in the profile
        ret_list.append(cur_elem/total_cells)

    return ret_list


class SingleCellDataset(Dataset):
    """A simple class used to store X and y relations as a paired dataset.
    We use it to store gene-based profiles (X) that are related to class-based profiles (y).
    This function is used to store tensors intended to train, validate or test the models generated.

    :param X_data: A tensor where each element is a list of gene counts or gene proportions.
    :type X_data: Tensor
    :param Y_data: A tensor where each element is a list of cell type counts or cell type proportions
    :type Y_data: Tensor
    """
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

class CelltypeDeconvolver(nn.Module):
    """A class extending the `nn.Module` from pytorch.

    :param num_feature: Number of input elements/features (usually gene-based).
    :type num_feature: int
    :param num_class: Number of output elements, usually cell types or similr classes.
    :type num_class: int
    :param number_of_layers_per_part: Number of hidden layers per layer block/part.
    :type number_of_layers_per_part: int
    :param first_part_size: Number of neurons per layer in the first block/part.
    :type first_part_size: int
    :param second_part_size: Number of neurons per layer in the second block/part.
    :type second_part_size: int
    :param last_part_size: Number of neurons per layer in the last block/part.
    :type last_part_size: int
    :param out_part_size: Number of neurons in the last layer immediate before the final output layer.
    :type out_part_size: int
    :param input_dropout: Dropout in the input layer, used to simulate spareness or missing genes during training.
    :type input_dropout: float
    :param normalize_output: Normalize output by scaling each tensor to 1, directly from the model and before computing the error. This sometimes speeds up the training for datasets with low number of classes.
    :type normalize_output: bool
    """
    def __init__(self, num_feature, num_class, number_of_layers_per_part, first_part_size, second_part_size, last_part_size, out_part_size, input_dropout, normalize_output=False):
        super(CelltypeDeconvolver, self).__init__()

        # helpers
        self.members = {}

        self.input_params = locals()
        # stores all layers
        self.layers=nn.ModuleList()

        # go from features to first part
        self.layers.append(nn.Dropout(p=input_dropout)) # add dropout to simulate sparseness
        self.layers.append(nn.Linear(num_feature, first_part_size))
        self.layers.append(nn.ReLU())

        # part 1
        self.layers.append(nn.BatchNorm1d(first_part_size))
        for i in range(number_of_layers_per_part):
            self.layers.append(nn.Linear(first_part_size, first_part_size))
        self.layers.append(nn.Linear(first_part_size, second_part_size))
        self.layers.append(nn.ReLU())

        # part 2
        self.layers.append(nn.BatchNorm1d(second_part_size))
        for i in range(number_of_layers_per_part):
            self.layers.append(nn.Linear(second_part_size, second_part_size))
        self.layers.append(nn.Linear(second_part_size, last_part_size))
        self.layers.append(nn.ReLU())

        # part 3
        self.layers.append(nn.BatchNorm1d(last_part_size))
        for i in range(number_of_layers_per_part):
            self.layers.append(nn.Linear(last_part_size, last_part_size))
        self.layers.append(nn.Linear(last_part_size, out_part_size))
        self.layers.append(nn.ReLU())
        #self.layers.append(nn.Dropout(p=0.1))

        # go from last part to num classes
        self.layers.append(nn.Linear(out_part_size, num_class))

        # leaky relu used to penalize values below 0
        self.layers.append(nn.LeakyReLU(negative_slope=0.1))

        # used in forward
        self.normalize_output = normalize_output
        self.num_class = num_class

    def forward(self, x):
        # iterate through all defined layers
        for layer in self.layers:
            x = layer(x)

        if normalize_output:
            x = nn.functional.relu(x)
            sums_ = torch.sum(x, 1) # check for invalid
            # set all invalid tensors to baseline
            x[sums_ <= 0] = torch.tensor([1/self.num_class]*self.num_class)

            sums_ = torch.sum(x, 1) # used for scaling
            # scale to 1
            x = torch.transpose(x, 0, 1)
            x = torch.div(x, sums_)
            x = torch.transpose(x, 0, 1)

        return x

    def Set(self, key, val):
        """Used to store members in the class's `member` dictionary.

        :param key: Key in the `member` dictionary.
        :type key: str
        :param val: Value to be stored.
        :type val: Anything
        """
        self.members[key] = val

    def Get(self, key):
        """Used to retrieve members in the class's `member` dictionary.

        :param key: Key in the `member` dictionary.
        :type key: str
        :return: Returns the value of the member.
        :rtype: Anything
        """
        return self.members[key]

class DeconvolutionExperiment:
    """A deconvolution experiment class used to keep track of everything that is required to do a full AntiSplodge experiment.

    :param SC: A single-cell dataset, formatted as an AnnData object.
    :type SC: AnnData
    """
    def __init__(self, SC):
        # h5ad/AnnData formated single-cell dataset
        self.SC = SC

        self.celltypes_column = ""
        self.celltypes = None
        self.num_classes = -1

        self.verbose = False

    def setVerbosity(self, verbose):
        """Sets the verbosity level of the prints of the experiment, either True or False.

        :param verbose: Verboisty of the prints (True or False), this is False when the experiment is inititalized.
        :type verbose: bool
        """
        self.verbose = verbose

    def setCellTypeColumn(self, name):
        """Column in the `SC` dataset, that holds the cell types. This create members: `celltypes_column`, `celltypes`, `num_classes`.

        :param name: Name (key) of the column.
        :type name: str
        """
        self.celltypes_column = name
        self.celltypes = np.array(np.unique(self.SC.obs[name]))
        self.num_classes = len(self.celltypes)

    def splitTrainTestValidation(self, train=0.9, rest=0.5):
        """Split the `SC` dataset into training, validation and test dataset, the splits are strattified on the cell types.
        This create members: `trainIndex`, `valIndex`, `testIndex`, `SC_train`, `SC_val`, `SC_test`.

        :param train: A number between 0 and 1 controlling the proportion of samples used in the training dataset, defaults to 0.9 (90%)
        :type train: float (0.9, optional)
        :param rest: A number between 0 and 1 controlling the proportion of samples used in the training dataset (the rest will be in the validation dataset), defaults to 0.5 (A 50%/50% split)
        :type rest: float (0.5, optional)
        """
        #
        # Split into train and rest
        #
        TrainIndex, RestIndex, _, _ = train_test_split(range(0,self.SC.n_obs),
                                                       self.SC.obs[self.celltypes_column],
                                                       test_size=1-train,
                                                       stratify=self.SC.obs[self.celltypes_column])

        #
        # Use the rest to split into validation and test
        #
        SC_rest = self.SC[RestIndex,:]
        ValIndex, TestIndex, _, _ = train_test_split(range(0,SC_rest.n_obs),
                                                     SC_rest.obs[self.celltypes_column],
                                                     test_size=rest,
                                                     stratify=SC_rest.obs[self.celltypes_column])
        # Final AnnData objects
        self.trainIndex = TrainIndex
        self.valIndex = ValIndex
        self.testIndex = TestIndex

        self.SC_train = self.SC[TrainIndex,:]
        self.SC_val = SC_rest[ValIndex,:]
        self.SC_test = SC_rest[TestIndex,:]

    def generateTrainTestValidation(self, num_profiles, CD):
        """Generate training, testing, and, validation profiles.
        This function will call `multinomialSampler`, `getConvolutedProfilesFromDistributions`, and, `getProportionFromCountVector`, in that order, for each dataset.
        This create members: `X_train_counts`, `X_val_counts`, `X_test_counts`, `X_train`, `X_val`, `X_test`, `Y_train`, `Y_val`, `Y_test`, `Y_train_prop`, `Y_val_prop`, `Y_test_prop`, `num_features`.

        :param num_profiles: A list of lengths 3, controlling the number of profiles used for training, testing, and, validation (index 0, 1, and, 2, respectively).
        :type num_profiles: list of ints, length = 3
        :param CD: A list of lengths 2, controlling the number of cell densities used (index 0 is the minimum number of CDs, and index 1 is the maximum number of CDs). The same CD will be used for the training, testing, and, validation dataset, respectively.
        :type CD: list of ints, length = 2
        """
        # SAMPLE PROFILES
        if self.verbose:
            print("GENERATING PROFILES")
        X_train_profiles = multinomialSampler(self.num_classes, num_profiles[0], CD[0], CD[1])
        X_val_profiles =   multinomialSampler(self.num_classes, num_profiles[1], CD[0], CD[1])
        X_test_profiles =  multinomialSampler(self.num_classes, num_profiles[2], CD[0], CD[1])

        if self.verbose:
            print("GENERATING TRAIN DATASET (N={})".format(len(X_train_profiles)))
        X_train, Y_train, I_ = getConvolutedProfilesFromDistributions(self.SC_train, self.celltypes, self.celltypes_column, X_train_profiles, normalize_X=True)
        Y_train_prop = getProportionFromCountVector(X_train_profiles)

        if self.verbose:
            print("GENERATING VALIDATION DATASET (N={})".format(len(X_val_profiles)))
        X_val, Y_val, I_ =     getConvolutedProfilesFromDistributions(self.SC_val, self.celltypes, self.celltypes_column, X_val_profiles, normalize_X=True)
        Y_val_prop = getProportionFromCountVector(X_val_profiles)

        if self.verbose:
            print("GENERATING TEST DATASET (N={})".format(len(X_test_profiles)))
        X_test, Y_test, I_ =   getConvolutedProfilesFromDistributions(self.SC_test, self.celltypes, self.celltypes_column, X_test_profiles, normalize_X=True)
        Y_test_prop = getProportionFromCountVector(X_test_profiles)

        # bind counts, proportions and convoluted profiles
        self.X_train_counts = X_train_profiles
        self.X_val_counts = X_val_profiles
        self.X_test_counts = X_test_profiles
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_test = Y_test
        self.Y_train_prop = Y_train_prop
        self.Y_val_prop = Y_val_prop
        self.Y_test_prop = Y_test_prop

        # set features to the number of elements in X_train
        self.num_features = X_train[0].shape[0]


    def setupDataLoaders(self, batch_size=1000):
        """Will process the profiles generated by the `generateTrainTestValidation` method into ready-to-use data loaders. This create members: `train_loader`, `val_loader`, `test_loader`.

        :param batch_size: The number of samples in each batch, defaults to 1000
        :type batch_size: int (1000, optional)
        """
        # batch size for data loaders
        self.batch_size = batch_size

        dataset_train = SingleCellDataset(torch.from_numpy(np.array(self.X_train)).float(), torch.from_numpy(np.array(self.Y_train_prop)).float())
        train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True
        )

        dataset_val = SingleCellDataset(torch.from_numpy(np.array(self.X_val)).float(), torch.from_numpy(np.array(self.Y_val_prop)).float())
        val_loader = DataLoader(
            dataset=dataset_val,
            batch_size=batch_size,
            shuffle=True
        )

        dataset_test = SingleCellDataset(torch.from_numpy(np.array(self.X_test)).float(), torch.from_numpy(np.array(self.Y_test_prop)).float())
        test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=batch_size
            # we don't shuffle test data
        )

        # bind loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def setupModel(self, cuda_id=1, dropout=0.33, fps=512, sps=256, lps=128, ops=64, lp=1):
        """Initialize the feed forward neural network model. We recommend about half number of nodes per part for each subsequent layer part.
        The first layer should be smaller than the input. Check out the member variable `num_features`.
        This create members: `model`, `device`.

        :param cuda_id: The id of the CUDA device, this can be either an int for the id or "cpu" (to use CPU device), defaults to 1
        :type cuda_id: int (or "cpu") (1, optional)
        :param dropout: [ParamDescription], defaults to 0.33
        :type dropout: float (0.33, optional)
        :param fps: Nodes for each layer for the first part/block, defaults to 512
        :type fps: int (512, optional)
        :param sps: Nodes for each layer for the second part/block, defaults to 256
        :type sps: int (256, optional)
        :param lps: Nodes for each layer for the last part/block, defaults to 128
        :type lps: int (128, optional)
        :param ops: Number of nodes in the last hidden layer just before the output layer, defaults to 64
        :type ops: int (64, optional)
        :param lp: Layers per part/block, defaults to 1
        :type lp: int (1, optional)
        """
        # CUDA SETTINGS
        device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print("(CUDA) device is: {}".format(device))

        # setup the NN model
        model = CelltypeDeconvolver(
            num_feature = self.num_features,
            num_class=self.num_classes,
            number_of_layers_per_part = lp,
            first_part_size = fps,
            second_part_size = sps,
            last_part_size = lps,
            out_part_size = ops,
            input_dropout = dropout
        )
        # bind to device
        model.Set("device", device)
        model.to(device)

        # bind settings and models
        self.device = device
        self.model = model

    def setupOptimizerAndCriterion(self, learning_rate = 0.001, optimizer=None, criterion=None):
        """Set the optimizer and criterion, and bind it to the model. This create members: `optimizer`, `criterion`.

        :param learning_rate: The learning rate of the optimizer, if you supply another optimizer, remember to set it yourself, defaults to 0.001
        :type learning_rate: float (0.001, optional)
        :param optimizer: The neural network optimizer, defaults to `None`, and will then use pytorch's `optim.Adam`.
        :type optimizer: Pytorch optimizer (None, optional)
        :param criterion: The neural network criterion, defaults to `None`, and will then use pytorch's `nn.SmoothL1Loss`.
        :type criterion: Pytorch criterion or loss function (None, optional)
        """
        # define optimizer and criterion if not set
        if optimizer == None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        if criterion == None:
            criterion = torch.nn.SmoothL1Loss(beta=0.25)

        # attach members as to the model object
        self.model.Set("optimizer", optimizer)
        self.model.Set("criterion", criterion)

        # bind optimizers
        self.optimizer = optimizer
        self.criterion = criterion


    def loadCheckpoint(self, checkpoint):
        """Loads a checkpoint file (.pt) containing the state of a neural network onto the `model` member variable.

        :param checkpoint: The path to the checkpoint file
        :type checkpoint: str
        """
        print("Restoring checkpoint:", checkpoint)
        self.model.load_state_dict(torch.load(checkpoint))

def train(experiment, patience=25, save_file=None, auto_load_model_on_finish=True, best_loss=None):
    """Train the model found in an experiment, this will utilize the train and validation dataset.

    :param patience: Patience counter, the training will stop once a new better loss hasn't been seen in the last `patience` epochs, defaults to 25
    :type patience: int (25, optional)
    :param save_file: The file to save the model parameters each time a better setting has been found. This is done each time the validation error is better (lower) than the best seen. Defaults to None, in which case a time-stamped file will be used.
    :type save_file: str or None (None, optional)
    :param auto_load_model_on_finish: If the best model settings should be loaded back onto the model when the training stops, defaults to True
    :type auto_load_model_on_finish: bool (True, optional)
    :param best_loss: A loss function to beat in order to save the model as the new best, used for warm restarts, defaults to None.
    :type best_loss: float or None (None, optional)
    :return: A dictionary with keys: `train_loss` and `validation_loss`, containing the train and validation loss for each epoch.
    :rtype: Dict
    """
    # time the function
    t0 = time.time()

    # retrieve experiment elements
    model        = experiment.model
    train_loader = experiment.train_loader
    val_loader   = experiment.val_loader

    # a save file is generated if not specified
    if save_file == None:
        today = date.today()
        d_ = today.strftime("%d-%m-%Y")
        save_file = "CelltypeDeconvolver_{0}_{1}.pt".format(d_, int(time.time()))
        print("Model will be saved in:", save_file)

    stats = {
        # TRAIN STATISTICS
        'train_loss': [],
        # VALIDATION STATS
        'validation_loss': []
    }

    # extract torch attributes
    device = model.Get("device")
    optimizer = model.Get("optimizer")
    criterion = model.Get("criterion")

    p_loss_value = best_loss # this will change on the first passthrough
    p_ = 0 # patience counter
    e_ = 0 # epoch counter
    while p_ < patience+1:
        # flags
        nans_found = 0

        #
        # TRAINING
        #
        model.train()
        train_epoch_loss = 0
        train_loss_counter = 0
        for X_, Y_ in train_loader:
            optimizer.zero_grad()

            # passthrough and backprop
            X_, Y_ = X_.to(device), Y_.to(device)
            Y_pred = model(X_)
            loss_ = criterion(Y_pred, Y_)
            loss_.backward()
            optimizer.step()

            # SCALE TO 1 (unit vector)
            Y_pred = nn.functional.relu(Y_pred) # first remove negatives
            sums_ = torch.sum(Y_pred, 1) # use the sum to scale
            Y_pred = torch.transpose(Y_pred, 0, 1)
            Y_pred = torch.div(Y_pred, sums_)
            Y_pred = torch.transpose(Y_pred, 0, 1)
            # END OF SCALING

            # With large sparse outputs NaNs can occur, simply report this for now (as a result of sums_ == 0)
            if np.sum(np.isnan(Y_pred.detach().cpu().numpy())) > 0:
                nans_found = 1

            # compute batch loss
            loss_ = loss_.item()
            if np.isnan(loss_):
                # use counter to revoke loss without values
                train_loss_counter += 1
            else:
                train_epoch_loss += loss_

        #
        # VALIDATION
        #
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_loss_counter = 0
            for X_, Y_ in val_loader:
                # passthrough
                X_, Y_ = X_.to(device), Y_.to(device)
                Y_pred = model(X_)
                loss_ = criterion(Y_pred, Y_)

                # SCALE TO 1 (unit vector)
                Y_pred = nn.functional.relu(Y_pred) # first remove negatives
                sums_ = torch.sum(Y_pred, 1) # use the sum to scale
                Y_pred = torch.transpose(Y_pred, 0, 1)
                Y_pred = torch.div(Y_pred, sums_)
                Y_pred = torch.transpose(Y_pred, 0, 1)
                # END OF SCALING

                # With large sparse outputs NaNs can occur, simply report this for now (as a result of sums_ == 0)
                if np.sum(np.isnan(Y_pred.detach().cpu().numpy())) > 0:
                    nans_found = 1

                # compute batch loss
                loss_ = loss_.item()
                if np.isnan(loss_):
                    val_loss_counter += 1
                else:
                    val_epoch_loss += loss_

        # STATS
        tel = train_epoch_loss/(len(train_loader)-train_loss_counter) # reduce by NaNs found
        vel = val_epoch_loss/(len(val_loader)-val_loss_counter) # reduce by NaNs found

        # Check validation loss
        if p_loss_value == None: # set target loss value
            p_loss_value = vel
        else:
            # if new loss is better than old then update
            if vel < p_loss_value:
                p_loss_value = vel
                p_ = 0 # reset patience

                # the model is better, save it as the current best for this timestep
                torch.save(model.state_dict(), save_file)


        # ADD STATS
        stats['train_loss'].append(tel)
        stats['validation_loss'].append(vel)

        # increase counters
        e_ += 1
        p_ += 1

        # report current stats
        if experiment.verbose:
            print(f'Epoch: {e_+0:03} | Epochs since last increase: {(p_-1)+0:03}' + ('| !!NaNs vectors produced!!' if nans_found else ''))
            print(f'Loss: (Train) {tel:.5f} | (Valid): {vel:.5f}')
            print("")

    print("Finished training (checkpoint saved in: {})".format(save_file))
    print(f"Time elapsed: {(time.time() - t0):.2f} ({(time.time() - t0)/60:.2f} Minutes)")
    if auto_load_model_on_finish:
        print("Autoloading best parameters onto model (auto_load_model_on_finish==True)")
        experiment.loadCheckpoint(save_file) # restore the best checkpoint

    return stats



def predict(experiment, test_loader=None):
    """Predict profiles using the current model found in the experiment, this will test dataset, if `test_loader` has not been set. You should load a loader yourself if you want to predict spots.

    :param test_loader: A test_loader with profiles to deconvolute, defaults to None, in which case the test profiles will be used.
    :type test_loader: Dataloader (None, optional)

    :return: A list of deconvoluted cell types (profiles).
    :rtype: List
    """
    profiles = []

    # retrieve experiment elements
    model = experiment.model
    # if test_loader is not set, use the one from the experiment
    if test_loader == None:
        test_loader = experiment.test_loader

    device = model.Get("device")
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            #
            # SCALE TO 1
            #
            # debug added
            y_pred = nn.functional.relu(y_pred) # first remove negatives

            if np.sum(np.isnan(y_pred.detach().cpu().numpy())) > 0:
                if experiment.verbose:
                    print("y_pred is nan (before)")

            # scale to 1
            sums_ = torch.sum(y_pred, 1)
            y_pred = torch.transpose(y_pred, 0, 1)
            y_pred = torch.div(y_pred, sums_)
            y_pred = torch.transpose(y_pred, 0, 1)

            if np.sum(np.isnan(y_pred.detach().cpu().numpy())) > 0:
                if experiment.verbose:
                    print("y_pred is nan (after)")

            #
            # END OF SCALING
            #

            for prof in y_pred:
                profiles.append(prof.detach().cpu().numpy())


    return profiles
