import torch.optim as optim
import sys
import math
from Models import *
from utility.helper import *
from utility.batch_test import *
from sklearn.mixture import GaussianMixture
from utility.parser import *
from scipy.sparse import lil_matrix, coo_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import torch
import numpy as np
import random
#import wandb
#wandb.init(project="MSHGNN_ALpha")


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def lamb(epoch):
    epoch += 0
    return 0.95 ** (epoch / 14)

result = []
txt = open("./result.txt", "a")
alpha1=args.alpha1
alpha2=args.alpha2

data_loader = Data(args.data_path + args.dataset, batch_size=args.batch_size)
data_loader.print_number_of_users()


def create_adjacency_matrix(features, n_components, device, verbose=False):
    if verbose:
        print(f"Starting GMM clustering with n_components: {n_components}")
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
        tol=1e-4,                # Convergence threshold
        reg_covar=1e-04,         # Regularization
        max_iter=60,             # Maximum number of iterations
        n_init=30,               # Number of initializations
        init_params='kmeans',    # Method for initialization: 'kmeans' or 'random'
        random_state=45,         # Seed for reproducibility
        warm_start=False,
        verbose=verbose
    )
    labels = gmm.fit_predict(features)
    
    if verbose:
        print("GMM clustering completed. Labels:", np.unique(labels, return_counts=True))
    
    n_nodes = features.shape[0]
    adj_cat = lil_matrix((n_nodes, n_nodes), dtype=float)  # Using sparse matrix representation

    # Vectorized approach to create adjacency matrix
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        adj_cat[indices[:, None], indices] = 1

    adj_cat = coo_matrix(adj_cat)  # Convert to COO format for efficient operations with PyTorch
    adj_cat = torch.sparse.FloatTensor(
        torch.LongTensor([adj_cat.row, adj_cat.col]),
        torch.FloatTensor(adj_cat.data),
        torch.Size(adj_cat.shape)
    ).to(device)  # Ensure it's on the correct device
    
    if verbose:
        print("Adjacency matrix shape:", adj_cat.shape)
    
    return adj_cat

# Node Features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node_features_df = data_loader.load_node_features()
node_features = node_features_df[[f'feature{i}' for i in range(20)]].values
print("Shape of Node/App features:", node_features.shape)
adj_cat = create_adjacency_matrix(node_features, n_components=45, device=device, verbose=True)

# User Features
user_features_df1 = data_loader.load_userss_features()
user_features = user_features_df1[[f'feature{i}' for i in range(20)]].values
print("Shape of Library/User features:", user_features.shape)
adj_cat_user = create_adjacency_matrix(user_features, n_components=45, device=device, verbose=True)




def jaccard_similarity(matrix):
    intersection = np.dot(matrix, matrix.T)
    square_sum = np.diag(intersection)  # Get the elements on the diagonal
    union = square_sum[:, None] + square_sum - intersection
    return np.divide(intersection, union)

class Model_Wrapper(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.mess_dropout = eval(args.mess_dropout)
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.record_alphas = False
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.model_type += '_%s_%s_layers%d' % (self.adj_type, self.alg_type, self.layer_num)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        print('model_type is {}'.format(self.model_type))

        self.weights_save_path = '%sweights/%s/%s/emb_size%s/layer_num%s/mess_dropout%s/drop_edge%s/lr%s/reg%s' % (
            args.weights_path, args.dataset, self.model_type,
            str(args.embed_size), str(args.layer_num), str(args.mess_dropout), str(args.drop_edge), str(args.lr),
            '-'.join([str(r) for r in eval(args.regs)]))
        self.result_message = []

        print('----self.alg_type is {}----'.format(self.alg_type))

        if self.alg_type in ['hcf']:
            self.model = HCF(self.n_users, self.n_items, self.emb_dim, self.layer_num, self.mess_dropout)
        else:
            raise Exception('Dont know which model to train')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2,self.adj_cat,self.adj_cat_user = self.build_hyper_edge(
            args.data_path + args.dataset + '/TE.csv')

        self.model = self.model.cuda()
        self.norm_u1 = self.norm_u1.cuda()
        self.norm_u2 = self.norm_u2.cuda()
        self.norm_i1 = self.norm_i1.cuda()
        self.norm_i2 = self.norm_i2.cuda()
        self.lr_scheduler = self.set_lr_scheduler()
        self.adj_cat = self.adj_cat.cuda() 
        self.adj_cat_user = self.adj_cat_user.cuda() 

    def get_D_inv(self, Hadj):

        H = sp.coo_matrix(Hadj.shape)
        H.row = Hadj.row.copy()
        H.col = Hadj.col.copy()
        H.data = Hadj.data.copy()
        rowsum = np.array(H.sum(1))
        columnsum = np.array(H.sum(0))

        Dv_inv = np.power(rowsum, -1).flatten()
        De_inv = np.power(columnsum, -1).flatten()
        Dv_inv[np.isinf(Dv_inv)] = 0.
        De_inv[np.isinf(De_inv)] = 0.

        Dv_mat_inv = sp.diags(Dv_inv)
        De_mat_inv = sp.diags(De_inv)
        return Dv_mat_inv, De_mat_inv

    def build_hyper_edge(self, file):
        user_inter = np.zeros((USR_NUM, ITEM_NUM))
        items_inter = np.zeros((ITEM_NUM, USR_NUM))
        self.adj_cat = adj_cat
        self.adj_cat_user = adj_cat_user
        with open(file) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip("\n").split(" ")
                uid = int(l[0])
                items = [int(j) for j in l[1:]]
                user_inter[uid, items] = 1
                items_inter[items, uid] = 1

        # User Similarity Matrix
        J_u = jaccard_similarity(user_inter)
        # the index of each hyperedge
        indices = np.where(J_u > alpha1)
        # the weight of each hyperedge node
        values = J_u[indices]
        # generate the hyperedge matrix.
        HEdge = sp.coo_matrix((values, indices), (USR_NUM, USR_NUM))
        self.HuEdge = (HEdge).T
        Dv_1, De_1 = self.get_D_inv(self.HuEdge)

        Dv_1 = self.sparse_mx_to_torch_sparse_tensor(Dv_1)
        De_1 = self.sparse_mx_to_torch_sparse_tensor(De_1)
        self.HuEdge = self.sparse_mx_to_torch_sparse_tensor(self.HuEdge)
        self.HuEdge_T = self.sparse_mx_to_torch_sparse_tensor(HEdge)

        spm1 = sparse.mm(Dv_1, self.HuEdge)
        self.norm_u1 = sparse.mm(spm1, De_1)
        self.norm_u2 = self.HuEdge_T

        J_i = jaccard_similarity(items_inter)
        # the index of each hyperedge
        indices = np.where(J_i >alpha2)
        # the weight of each node in each hyperedge
        values = J_i[indices]
        # generate the hyperedge matrix
        HEdge = sp.coo_matrix((values, indices), (ITEM_NUM, ITEM_NUM))
        self.HiEdge = (HEdge).T
        Dv_1, De_1 = self.get_D_inv(self.HiEdge)

        Dv_1 = self.sparse_mx_to_torch_sparse_tensor(Dv_1)
        De_1 = self.sparse_mx_to_torch_sparse_tensor(De_1)
        self.HiEdge = self.sparse_mx_to_torch_sparse_tensor(self.HiEdge)
        self.HiEdge_T = self.sparse_mx_to_torch_sparse_tensor(HEdge)

        spm1 = sparse.mm(Dv_1, self.HiEdge)
        self.norm_i1 = sparse.mm(spm1, De_1)
        self.norm_i2 = self.HiEdge_T

        return self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2,self.adj_cat,self.adj_cat_user

    def set_lr_scheduler(self):  # lr_scheduler: learning rate scheduler.
        fac = lamb
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        # Each lr value: the initial lr from the optimizer multiplied by a lambda.
        return scheduler

    def save_model(self):
        ensureDir(self.weights_save_path)
        torch.save(self.model.state_dict(), self.weights_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.weights_save_path))

    def test(self, users_to_test, drop_flag=False, batch_test_flag=False):
        self.model.eval()  # In evaluation mode, batch normalization and dropout layers are not active, equivalent to self.model.train(False)
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2,self.adj_cat,self.adj_cat_user)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test)
        return result
    
    def train(self):
        result = []
        txt = open("./result.txt", "a")
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger = [], [], [], [], [], [], [], []
        stopping_step = 10
        should_stop = False
        cur_best_pre_0 = 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        
        # Learning rate warm-up and scheduler
        def lr_lambda(epoch):
            return 0.01 if epoch < 10 else 1.0  # Warm-up for first 10 epochs
        
        base_lr = args.lr
        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        reduce_lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        for epoch in range(args.epoch):
            t1 = time()

            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.

            for idx in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                ua_embeddings, ia_embeddings = self.model(self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2, self.adj_cat, self.adj_cat_user)

                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.hinge_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)  # Gradient clipping
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

            warmup_scheduler.step()  # Apply learning rate warm-up
            self.lr_scheduler.step()  # Original learning rate scheduler step
            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings
            torch.cuda.empty_cache()

            if math.isnan(loss):
                print('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, emb_loss)
                    print(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            ret = self.test(users_to_test, drop_flag=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            map_loger.append(ret['map'])
            mrr_loger.append(ret['mrr'])
            fone_loger.append(ret['fone'])
            
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], "Recall :" recall=[%.5f, %.5f], ' \
                        'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], map=[%.5f, %.5f], mrr=[%.5f, %.5f], f1=[%.5f, %.5f]' % \
                        (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1], ret['map'][0], ret['map'][-1], ret['mrr'][0],
                            ret['mrr'][-1], ret['fone'][0], ret['fone'][-1]) 

                # wandb.log({
                #     'Epoch': epoch,
                #     'Training Time': t2 - t1,
                #     'Loss': loss,
                #     'MF Loss': mf_loss,
                #     'Embedding Loss': emb_loss,
                #     'Regularization Loss': reg_loss,
                #     'Recall@K': ret['recall'][0],
                #     'Recall@K_end': ret['recall'][-1],
                #     'Precision@K': ret['precision'][0],
                #     'Precision@K_end': ret['precision'][-1],
                #     'Hit Ratio@K': ret['hit_ratio'][0],
                #     'Hit Ratio@K_end': ret['hit_ratio'][-1],
                #     'NDCG@K': ret['ndcg'][0],
                #     'NDCG@K_end': ret['ndcg'][-1],
                #     'MAP@K': ret['map'][0],
                #     'MAP@K_end': ret['map'][-1],
                #     'MRR@K': ret['mrr'][0],
                #     'MRR@K_end': ret['mrr'][-1],
                #     'F1@K': ret['fone'][0],
                #     'F1@K_end': ret['fone'][-1]
                # })
            
                result.append(perf_str + "\n")
                txt.write(perf_str + "\n")
                txt.close()
                txt = open("./result.txt", "a")
                print(perf_str)
            
            reduce_lr_scheduler.step(ret['recall'][0])  # Update learning rate based on validation recall
            
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=300)

            if should_stop:
                break

            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                self.save_model()
                if self.record_alphas:
                    self.best_alphas = [i for i in self.model.get_alphas()]
                print('save the weights in path: ', self.weights_save_path)

        if args.save_recom:
            results_save_path = r'./output/%s/rec_result.csv' % (args.dataset)
            self.save_recResult(results_save_path)

        if rec_loger != []:
            self.print_final_results(rec_loger, pre_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger, training_time_list)

    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    set_seed(42)

    

    def norm(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def save_recResult(self, outputPath):
        # used for reverve the recommendation lists
        recommendResult = {}
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

        # get all apps (users)
        users_to_test = list(data_generator.test_set.keys())
        n_test_users = len(users_to_test)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0

        # calculate the result by our own
        # get the latent factors
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_u1, self.norm_u2, self.norm_i1,
                                                      self.norm_i2,self.adj_cat,self.adj_cat_user)

        # get result in batch
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = users_to_test[start: end]
            item_batch = range(ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]
            # get the ratings
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))
            # move from GPU to CPU
            rate_batch = rate_batch.detach().cpu().numpy()
            # contact each user's ratings with his id
            user_rating_uid = zip(rate_batch, user_batch)
            # now for each user, calculate his ratings and recommendation
            for x in user_rating_uid:
                # user u's ratings for user u
                rating = x[0]
                # uid
                u = x[1]
                training_items = data_generator.train_items[u]
                user_pos_test = data_generator.test_set[u]
                all_items = set(range(ITEM_NUM))
                test_items = list(all_items - set(training_items))
                item_score = {}
                for i in test_items:
                    item_score[i] = rating[i]
                K_max = max(Ks)
                K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
                recommendResult[u] = K_max_item_score

        # output the result to csv file.
        ensureDir(outputPath)
        with open(outputPath, 'w') as f:
            print("----the recommend result has %s items." % (len(recommendResult)))
            for key in recommendResult.keys():  # due to that all users have been used for test and the subscripts start from 0.
                outString = ""
                for v in recommendResult[key]:
                    outString = outString + "," + str(v)
                f.write("%s%s\n" % (key, outString))

    def print_final_results(self, rec_loger, pre_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger,
                            training_time_list):
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        map = np.array(map_loger)
        mrr = np.array(mrr_loger)
        fone = np.array(fone_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s], map=[%s],mrr=[%s], f1=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in hit_loger[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcg_loger[idx]]),
                      '\t'.join(['%.5f' % r for r in map[idx]]),
                      '\t'.join(['%.5f' % r for r in mrr[idx]]),
                      '\t'.join(['%.5f' % r for r in fone[idx]]))
        result.append(final_perf + "\n")
        txt.write(final_perf + "\n")
        print(final_perf)

    # pos_items: the IDs of positively correlated items
    # neg_items： the IDs of negatively correlated item
    def hinge_loss(self, users, pos_items, neg_items, margin=1.0):
        # Apply dropout to the embeddings
        dropout = 0.2
        users = F.dropout(users, p=dropout, training=self.model.training)
        pos_items = F.dropout(pos_items, p=dropout, training=self.model.training)
        neg_items = F.dropout(neg_items, p=dropout, training=self.model.training)
        
        # Calculate scores
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        
        # Hinge Loss
        hinge_loss_value = torch.mean(torch.clamp(margin - pos_scores + neg_scores, min=0.0))
        
        # Regularization
        regularizer = (1. / 2) * (users.norm(2).pow(2) + pos_items.norm(2).pow(2) + neg_items.norm(2).pow(2))
        regularizer = regularizer / self.batch_size
        
        # Combine Hinge Loss with the regularization loss
        mf_loss = hinge_loss_value
        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        
        return mf_loss, emb_loss, reg_loss



    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        #return torch.sparse.FloatTensor(indices, values, shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=values.dtype, device=values.device)


    def get_sparse_tensor_value(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

if __name__ == '__main__':



    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items


    t0 = time()

    pretrain_data = None

    Engine = Model_Wrapper(data_config=config, pretrain_data=pretrain_data)
    if args.pretrain:
        print('pretrain path: ', Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            users_to_test = list(data_generator.test_set.keys())
            ret = Engine.test(users_to_test, drop_flag=True)
            cur_best_pre_0 = ret['recall'][0]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                           'ndcg=[%.5f, %.5f], map=[%.5f, %.5f], mrr=[%.5f, %.5f], f1=[%.5f, %.5f]' % \
                           (ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1],
                            ret['map'][0], ret['map'][-1],
                            ret['mrr'][0], ret['mrr'][-1],
                            ret['fone'][0], ret['fone'][-1])
            print(pretrain_ret)
        else:
            print('Cannot load pretrained model. Start training from stratch')
    else:
        print('without pretraining')
    Engine.train()
