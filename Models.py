import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

class MsRec(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, layer_num, dropout_list, scale_weights=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = layer_num

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.dropout_list = nn.ModuleList([nn.Dropout(p) for p in dropout_list])

        # Initialize weights for scaling the contributions of each scale
        if scale_weights is None:
            self.scale_weights = nn.Parameter(torch.ones(layer_num + 1))
        else:
            self.scale_weights = nn.Parameter(torch.tensor(scale_weights))

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_u1, adj_u2, adj_i1, adj_i2, adj_cat, adj_cat_user):
        hu = self.user_embedding.weight
        embedding_u = [hu]

        for i in range(self.n_layers):
            t = torch.sparse.mm(adj_u2, embedding_u[-1])
            t = torch.sparse.mm(adj_u1, t)
            t_cat = torch.sparse.mm(adj_cat_user, t)
            t_cat = self.dropout_list[i](t_cat)
            
            # Combine original and new scale embeddings with adaptive weights
            combined_u = self.scale_weights[i] * t_cat + (1 - self.scale_weights[i]) * t
            embedding_u.append(combined_u)

        u_emb = torch.stack(embedding_u, dim=1).mean(dim=1)

        hi = self.item_embedding.weight
        embedding_i = [hi]

        for i in range(self.n_layers):
            t = torch.sparse.mm(adj_i2, embedding_i[-1])
            t = torch.sparse.mm(adj_i1, t)
            t_cat = torch.sparse.mm(adj_cat, t)
            t_cat = self.dropout_list[i](t_cat)

            # Combine original and new scale embeddings with adaptive weights
            combined_i = self.scale_weights[i] * t_cat + (1 - self.scale_weights[i]) * t
            embedding_i.append(combined_i)

        i_emb = torch.stack(embedding_i, dim=1).mean(dim=1)

        return u_emb, i_emb
