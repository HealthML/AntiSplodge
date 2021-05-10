import torch # TODO USED
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler # TODO USED
import numpy as np # TODO USED
import scanpy as sc
from scipy.spatial import distance
import torch.nn as nn
import torch.optim as optim
from datetime import date
import time

#
# Nc: is the number of cell types
# M: is the number of profiles for each CD we want to sample
#
def multinomialSampler(Nc, M, CD_min, CD_max):
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

#
#
#
#
def getConvolutedProfilesFromDistributions(adata, cell_types, cell_type_key, distributions, normalize_X=False):
    # make a copy of the adata
    adata_copy = adata.copy()
    # pre subset cell_types
    cell_types_cache = {}
    desified_X_cache = {}
    for cell_type in cell_types:
        print(cell_type)
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

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

class CelltypeDeconvolver(nn.Module):
    def __init__(self, num_feature, num_class, number_of_layers_per_part, first_part_size, second_part_size, last_part_size, out_part_size, input_dropout):
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

    def forward(self, x):
        # iterate through all defined layers
        for layer in self.layers:
            x = layer(x)
        return x

    def Set(self, key, val):
        self.members[key] = val

    def Get(self, key):
        return self.members[key]

class DeconvolutionExperiment:
    def __init__(self, SC):
        # h5ad/AnnData formated single-cell dataset
        self.SC = SC

        self.celltypes_column = ""
        self.celltypes = None
        self.num_classes = -1

        self.verbose = False

    def setVerbosity(self, verbose):
        self.verbose = verbose

    def setCellTypeColumn(self, name):
        print(self)
        self.celltypes_column = name
        self.celltypes = np.array(np.unique(self.SC.obs[name]))
        self.num_classes = len(self.celltypes)

    def splitTrainTestValidation(self, train=0.9, rest=0.5):
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
        SC_rest = SC_subset[RestIndex,:]
        ValIndex, TestIndex, _, _ = train_test_split(range(0,self.SC.n_obs),
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
        # SAMPLE PROFILES
        if self.verbose:
            print("GENERATING PROFILES")
        X_train_profiles = multinomialSampler(self.num_classes, num_profiles[0], CD[0], CD[1])
        X_val_profiles =   multinomialSampler(self.num_classes, num_profiles[1], CD[0], CD[1])
        X_test_profiles =  multinomialSampler(self.num_classes, num_profiles[2], CD[0], CD[1])

        if self.verbose:
            print("GENERATING TRAIN DATASET")
        X_train, Y_train, I_ = getConvolutedProfilesFromDistributions(self.SC_train, self.celltypes, self.celltypes_column, X_train_profiles)
        Y_train_prop = getProportionFromCountVector(X_train_profiles)

        if self.verbose:
            print("GENERATING VALIDATION DATASET")
        X_val, Y_val, I_ =     getConvolutedProfilesFromDistributions(self.SC_val, self.celltypes, self.celltypes_column, X_val_profiles)
        Y_val_prop = getProportionFromCountVector(X_val_profiles)

        if self.verbose:
            print("GENERATING TEST DATASET")
        X_test, Y_test, I_ =   getConvolutedProfilesFromDistributions(self.SC_test, self.celltypes, self.celltypes_column, X_test_profiles)
        Y_test_prop = getProportionFromCountVector(X_test_profiles)

        # bind counts, proportions and convoluted profiles
        self.X_train_counts = X_train_profiles
        self.X_val_counts = X_val_profiles
        self.X_test_counts = X_test_profiles
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.Y_train_prop = Y_train_prop
        self.Y_val_prop = Y_val_prop
        self.Y_test_prop = Y_test_prop

        # set features to the number of elements in X_train
        self.num_features = X_train[0].shape[0]


    def setupDataLoaders(self, batch_size=1000):
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
        model.to(device)

        # define optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.SmoothL1Loss(beta=0.25)

        # attach members as to the model object
        model.Set("device", device)
        model.Set("optimizer", optimizer)
        model.Set("criterion", criterion)

        # bind settings and models
        self.device = device
        self.model = model


    def loadCheckpoint(self, checkpoint):
        print("Restoring checkpoint:", checkpoint)
        self.model.load_state_dict(torch.load(checkpoint))


# patience is the number of epochs before stopping
def train(experiment, patience=25, save_file=None, auto_load_model_on_finish=True):
    # time the function
    t0 = time.time()

    # retrieve experiment elements
    model        = experiment.model
    train_loader = experiment.train_loader
    val_loader   = train_loader.val_loader

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

    p_loss_value = 0 # this will change on the first passthrough
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
        tel = train_epoch_loss/(len(data_loader)-train_loss_counter) # reduce by NaNs found
        vel = val_epoch_loss/(len(val_loader)-val_loss_counter) # reduce by NaNs found

        # Check validation loss
        if e_ == 0: # set target loss value at first epoch
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
            print(f'Loss: (Train) {tel:.5f} | (Valid): {tea:.3f}')
            print("")

    print("Finished training (checkpoint saved in: {})".format(save_file))
    print(f"Time elapsed: {(time.time() - t0):.2f} ({(time.time() - t0)/60:.2f} Minutes)")
    if auto_load_model_on_finish:
        print("Autoloading best parameters onto model (auto_load_model_on_finish==True)")
        experiment:loadCheckpoint(model, save_file) # restore the best checkpoint

    return stats



def predict(experiment):
    profiles = []

    # retrieve experiment elements
    model        = experiment.model
    test_loader  = experiment.test_loader

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
                print("y_pred is nan (before)")

            # scale to 1
            sums_ = torch.sum(y_pred, 1)
            y_pred = torch.transpose(y_pred, 0, 1)
            y_pred = torch.div(y_pred, sums_)
            y_pred = torch.transpose(y_pred, 0, 1)

            if np.sum(np.isnan(y_pred.detach().cpu().numpy())) > 0:
                print("y_pred is nan (after)")

            #
            # END OF SCALING
            #

            for prof in y_pred:
                profiles.append(prof.detach().cpu().numpy())


    return profiles
