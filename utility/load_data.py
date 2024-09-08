import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import csv 
import torch
import pandas as pd
import os
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import StandardScaler




class Data:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/TE.csv'
        test_file = path + '/RS.csv'

        # Print the file paths
        print("Training file path:", train_file)
        print("Testing file path:", test_file)
        # Print the shapes of the loaded data
        #print("Shape of training data:", train_file.shape)
        #print("Shape of testing data:", test_file.shape)

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []   # Unique users
        
        
        #add a dictionary to store the recommendation result.
        self.recommendResult = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {} #Add training set and test set
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for idx, i in enumerate(train_items):
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

   

    def print_number_of_users(self):
        """Print the total number of users in the dataset."""
        print(f"Total number of users: {len(self.exist_users)}")

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    
    def load_node_features(self):
        features_path = os.path.join(self.path, 'node_features.csv')
        if os.path.exists(features_path):
            features = pd.read_csv(features_path)
        else:
            features = self.generate_node_features()  # This calls your updated feature generation method

        # No need for PCA here since generate_node_features uses SVD
        # If you still want to apply PCA on the matrix factorization results for some reason, you can include it
        print(f"Node features DataFrame shape: {features.shape}")

        return features
    
    def load_userss_features(self):
        features_path = os.path.join(self.path, 'user_features.csv')
        if os.path.exists(features_path):
            features = pd.read_csv(features_path)
        else:
            features = self.generate_user_features()  # This now calls the updated user feature generation method

        print(f"User features DataFrame shape: {features.shape}")
        return features
        

    def ggenerate_node_features(self):
        # Use SVD to perform matrix factorization on the interaction matrix
        interaction_matrix_csr = self.R.tocsr()
        print("Interaction matrix shape:", interaction_matrix_csr.shape)
        
        # Choose the number of latent factors
        # It should not exceed the number of items and should match the original feature count
        num_features = min(interaction_matrix_csr.shape[1], 20)  # Here, we have 20 original features
        u, s, vt = svds(interaction_matrix_csr, k=num_features)
        
        # Use the matrix 'vt' (item latent factors) as item features
        item_features = vt.T
        print("Item features matrix shape from SVD:", item_features.shape)
        
        # Turn item features into a DataFrame
        features_df = pd.DataFrame(item_features, columns=[f'feature{i}' for i in range(num_features)])
        features_df['node_id'] = range(self.n_items)
        
        # If you want to match the shape with the original features + PCs, uncomment the following line
        # features_df = features_df[['node_id'] + [f'feature{i}' for i in range(num_features)]]
        
        # Save the features DataFrame to a CSV file
        features_path = os.path.join(self.path, 'node_features.csv')
        features_df.to_csv(features_path, index=False)
        
        print(f"Generated node features DataFrame shape: {features_df.shape}")
        print(f"Saved the node features to {features_path}")
        
        return features_df
    

    def ggenerate_user_features(self):
        # Use SVD to perform matrix factorization on the interaction matrix
        interaction_matrix_csr = self.R.tocsr()
        print("Interaction matrix shape:", interaction_matrix_csr.shape)
        
        # Ensure we choose the minimum between the number of users and the desired feature count
        num_features = min(interaction_matrix_csr.shape[0], 20)  # Here, we have 20 original features
        u, s, vt = svds(interaction_matrix_csr, k=num_features)
        
        # Use the matrix 'u' as user features
        user_features = u
        print("User features matrix shape from SVD:", user_features.shape)
        
        # Turn user features into a DataFrame
        features_df = pd.DataFrame(user_features, columns=[f'feature{i}' for i in range(num_features)])
        features_df['node_id'] = range(interaction_matrix_csr.shape[0])  # Assuming node_id corresponds to user indices
        
        # Save the features DataFrame to a CSV file
        features_path = os.path.join(self.path, 'user_features.csv')
        features_df.to_csv(features_path, index=False)
        
        print(f"Generated user features DataFrame shape: {features_df.shape}")
        print(f"Saved the user features to {features_path}")
        
        return features_df

    def generate_user_features(self):
        interaction_matrix_csr = self.R.tocsr()
        print("Interaction matrix shape:", interaction_matrix_csr.shape)
        
        num_features = min(interaction_matrix_csr.shape[0], 20)  # NMF latent factors
        nmf_model = NMF(n_components=num_features, init='nndsvda', random_state=42)
        user_features_nmf = nmf_model.fit_transform(interaction_matrix_csr)
        print("User features matrix shape from NMF:", user_features_nmf.shape)

        # Ensure we get the same number of PCA features
        pca = PCA(n_components=num_features)
        user_features_pca = pca.fit_transform(interaction_matrix_csr.toarray())
        print("User features matrix shape from PCA:", user_features_pca.shape)

        # Ensure the PCA features have the same number of rows as the NMF features
        if user_features_pca.shape[0] != user_features_nmf.shape[0]:
            raise ValueError("The number of rows in PCA and NMF features must match")

        # Combine NMF and PCA features
        user_features_combined = np.hstack((user_features_nmf, user_features_pca))
        print("Combined user features matrix shape:", user_features_combined.shape)

        # Normalize the combined features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(user_features_combined)

        # Turn combined features into a DataFrame
        features_df = pd.DataFrame(normalized_features, columns=[f'feature{i}' for i in range(normalized_features.shape[1])])
        features_df['node_id'] = range(self.n_users)
        
        features_path = os.path.join(self.path, 'user_features.csv')
        features_df.to_csv(features_path, index=False)
        
        print(f"Generated user features DataFrame shape: {features_df.shape}")
        print(f"Saved the user features to {features_path}")
        
        return features_df


    def generate_node_features(self):
        interaction_matrix_csr = self.R.tocsr()
        print("Interaction matrix shape:", interaction_matrix_csr.shape)
        
        num_features = min(interaction_matrix_csr.shape[1], 20)  # NMF latent factors
        nmf_model = NMF(n_components=num_features, init='nndsvda', random_state=42)
        item_features_nmf = nmf_model.fit_transform(interaction_matrix_csr.T)
        print("Node features matrix shape from NMF:", item_features_nmf.shape)

        # Ensure we get the same number of PCA features
        pca = PCA(n_components=num_features)
        item_features_pca = pca.fit_transform(interaction_matrix_csr.toarray().T)
        print("Node features matrix shape from PCA:", item_features_pca.shape)

        # Ensure the PCA features have the same number of rows as the NMF features
        if item_features_pca.shape[0] != item_features_nmf.shape[0]:
            raise ValueError("The number of rows in PCA and NMF features must match")

        # Combine NMF and PCA features
        item_features_combined = np.hstack((item_features_nmf, item_features_pca))
        print("Combined node features matrix shape:", item_features_combined.shape)

        # Normalize the combined features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(item_features_combined)

        # Turn combined features into a DataFrame
        features_df = pd.DataFrame(normalized_features, columns=[f'feature{i}' for i in range(normalized_features.shape[1])])
        features_df['node_id'] = range(self.n_items)
        
        features_path = os.path.join(self.path, 'node_features.csv')
        features_df.to_csv(features_path, index=False)
        
        print(f"Generated node features DataFrame shape: {features_df.shape}")
        print(f"Saved the node features to {features_path}")
        
        return features_df


    


    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            #print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat) #+ sp.eye(adj_mat.shape[0])
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    #Generate a negative sampling pool for each user, sampling 100 unrelated items for each user.
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)#random.sample()Randomly extract a sequence of specified length from the given sequence.
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
            #ramdom.choice()Randomly return a value from the given sequence
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]#The return is a list, and [0] retrieves the value from the list
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)  #Sample a positively correlated item.
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items


    def sample_all_users_pos_items(self):
        self.all_train_users = []
        self.all_train_pos_items = []
        for u in self.exist_users:
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        #Sample a certain number of negative items for the user.
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        neg_items = []
        for u in self.all_train_users:
            neg_items += sample_neg_items_for_u(u,1)

        perm = np.random.permutation(len(self.all_train_users))
        # Shuffle the training users, positive samples, and negative samples
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')
        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

