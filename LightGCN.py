
import tensorflow as tf
import time
import os
import sys
import numpy as np
import pandas as pd
from metrics.map import map_at_k
from metrics.ndcg import ndcg_at_k
from metrics.precision import precision_at_k
from metrics.recall import recall_at_k
from utilities import get_top_k_scored_items


tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x


class LightGCN(object):
    """LightGCN model
    :Citation:
        He, Xiangnan, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang.
        "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." arXiv
        preprint arXiv:2002.02126, 2020.
    """

    def __init__(self, hparams, data, defaultMode , seed=None):
        """Initializing the model. Create parameters, placeholders, embeddings and loss function.
        Args:
            hparams (HParams): A HParams object, hold the entire set of hyperparameters.
            data (object): A recommenders.models.deeprec.DataModel.ImplicitCF object, load and process data.
            seed (int): Seed.
        """

        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        self.defaultMode = defaultMode
        self.alpha = 0.5
        self.data = data
        self.epochs = hparams["epochs"]
        self.learning_rate = hparams["learning_rate"]
        self.emb_dim = hparams["embed_size"]
        self.batch_size = hparams["batch_size"]
        self.n_layers = hparams["n_layers"]
        self.decay = hparams["decay"]
        self.eval_epoch = hparams["eval_epoch"]
        self.top_k = hparams["top_k"]
        self.metrics = hparams["metrics"]

        self.n_users, self.n_items = self.data.n_users, self.data.n_items
        
        self.normalized_adjacency_matrix = data.create_norm_adj_mat()

        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        self.weights = self._init_weights()
        self.average_user_embeddings, self.average_item_embeddings = self.buildEmbeddings()

        # https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do
        self.u_g_embeddings = tf.nn.embedding_lookup(
            params=self.average_user_embeddings, ids=self.users
        )
        self.positive_item_avg_embeddings = tf.nn.embedding_lookup(
            params=self.average_item_embeddings, ids=self.pos_items
        )
        self.negative_item_avg_embeddings = tf.nn.embedding_lookup(
            params=self.average_item_embeddings, ids=self.neg_items
        )
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(
            params=self.weights["user_embedding"], ids=self.users
        )
        self.positive_item_avg_embeddings_pre = tf.nn.embedding_lookup(
            params=self.weights["item_embedding"], ids=self.pos_items
        )
        self.negative_item_avg_embeddings_pre = tf.nn.embedding_lookup(
            params=self.weights["item_embedding"], ids=self.neg_items
        )

        self.batch_ratings = tf.matmul(
            self.u_g_embeddings,
            self.positive_item_avg_embeddings,
            transpose_a=False,
            transpose_b=True,
        )

        self.mf_loss, self.emb_loss = self._create_bpr_loss(
            self.u_g_embeddings, self.positive_item_avg_embeddings, self.negative_item_avg_embeddings
        )
        self.loss = self.mf_loss + self.emb_loss

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss
        )

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _init_weights(self):
        """Initialize user and item embeddings.
        Returns:
            dict: With keys `user_embedding` and `item_embedding`, embeddings of all users and items.
        """
        ## https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize
        embeddings = dict()
        print("Initializing user embeddings and item embeddings...")
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )  # what is fan avg?

        embeddings["user_embedding"] ,embeddings["item_embedding"] = tf.Variable(
            initializer([self.n_users, self.emb_dim]), name="user_embedding"
        ), tf.Variable(
            initializer([self.n_items, self.emb_dim]), name="item_embedding"
        )
        
        return embeddings
    

    def buildEmbeddings(self):
        """Calculate the average embeddings of users and items after every layer of the model.
        Returns:
            tf.Tensor, tf.Tensor: Average user embeddings. Average item embeddings.
        """
        A_hat = self.sparseMatrix_Tensor(self.normalized_adjacency_matrix)

        ego_embeddings = tf.concat(
            [self.weights["user_embedding"], self.weights["item_embedding"]], axis=0
        )
        all_embeddings = [ego_embeddings]

        if(self.defaultMode):
            print("Default mode - LightGCN is running")
            for k in range(0, self.n_layers):
                ego_embeddings = tf.sparse.sparse_dense_matmul(A_hat, ego_embeddings)
                all_embeddings += [ego_embeddings]
                print(all_embeddings)
            all_embeddings = tf.stack(all_embeddings, 1)
            all_embeddings = tf.reduce_mean(
                input_tensor=all_embeddings, axis=1, keepdims=False
            )
        else:
            print("LightGCN++ is running")

            for k in range(0, self.n_layers):
                ego_embeddings = tf.sparse.sparse_dense_matmul(A_hat, ego_embeddings)
                ego_embeddings = tf.math.scalar_mul((self.alpha**(self.n_layers-1 -k)),ego_embeddings)
                all_embeddings += [ego_embeddings]
                print(all_embeddings)
            all_embeddings = tf.stack(all_embeddings, 1)
            all_embeddings = tf.reduce_mean(
                input_tensor=all_embeddings, axis=1, keepdims=False
            )

       
        u_g_embeddings, i_g_embeddings = tf.split(
            all_embeddings, [self.n_users, self.n_items], 0
        )
        return u_g_embeddings, i_g_embeddings

    def _create_bpr_loss(self, users, pos_items, neg_items):
        """Calculate BPR loss.
        Args:
            users (tf.Tensor): User embeddings to calculate loss.
            pos_items (tf.Tensor): Positive item embeddings to calculate loss.
            neg_items (tf.Tensor): Negative item embeddings to calculate loss.
        Returns:
            tf.Tensor, tf.Tensor: Matrix factorization loss. Embedding regularization loss.
        """
        positive_ratings = tf.reduce_sum(input_tensor=tf.multiply(users, pos_items), axis=1)
        negative_ratings = tf.reduce_sum(input_tensor=tf.multiply(users, neg_items), axis=1)

        regularizer = (
            tf.nn.l2_loss(self.u_g_embeddings_pre)
            + tf.nn.l2_loss(self.positive_item_avg_embeddings_pre)
            + tf.nn.l2_loss(self.negative_item_avg_embeddings_pre)
        )
        regularizer = regularizer / self.batch_size
        ## why softplus activation? (relu implemented as an approximation)
        relu_score = tf.nn.softplus(-(positive_ratings - negative_ratings))
        mf_loss = tf.reduce_mean(
            input_tensor=relu_score
        )
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss

    def sparseMatrix_Tensor(self, X):
        """Convert a scipy sparse matrix to tf.SparseTensor.
        Returns:
            tf.SparseTensor: SparseTensor after conversion.
        """
        ## coo - scipy matrix in the coordinate format. coo_matrix((data, (i, j)), [shape=(M, N)])
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def fit(self):
        """Fit the model on self.data.train. If eval_epoch is not -1, evaluate the model on `self.data.test`
        every `eval_epoch` epoch to observe the training status.
        """
        for epoch in range(1, self.epochs + 1):
            train_start = time.time()
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
            n_batch = self.data.train.shape[0] // self.batch_size + 1
            for idx in range(n_batch):
                users, pos_items, neg_items = self.data.train_loader(self.batch_size)
                _, batch_loss, batch_mf_loss, batch_emb_loss = self.sess.run(
                    [self.optimizer, self.loss, self.mf_loss, self.emb_loss],
                    feed_dict={
                        self.users: users,
                        self.pos_items: pos_items,
                        self.neg_items: neg_items,
                    },
                )
                loss += batch_loss / n_batch
                mf_loss += batch_mf_loss / n_batch
                emb_loss += batch_emb_loss / n_batch

            if np.isnan(loss):
                print("ERROR: loss is nan.")
                sys.exit()
            train_end = time.time()
            train_time = train_end - train_start
            
            if self.eval_epoch == -1 or epoch % self.eval_epoch != 0:
                print(
                    "Epoch %d (train)%.1fs: train loss = %.5f = (matrix factorization)%.5f + (embedding)%.5f"
                    % (epoch, train_time, loss, mf_loss, emb_loss)
                )
            else:
                eval_start = time.time()
                ret = self.run_eval()
                eval_end = time.time()
                eval_time = eval_end - eval_start

                print(
                    "Epoch %d (train)%.1fs + (eval)%.1fs: train loss = %.5f = (matrix factorization)%.5f + (embedding)%.5f, %s"
                    % (
                        epoch,
                        train_time,
                        eval_time,
                        loss,
                        mf_loss,
                        emb_loss,
                        ", ".join(
                            metric + " = %.5f" % (r)
                            for metric, r in zip(self.metrics, ret)
                        ),
                    )
                )

    def run_eval(self):
        """Run evaluation on self.data.test.
        Returns:
            dict: Results of all metrics in `self.metrics`.
        """
        topk_scores = self.recommendKItemsForallUsers(
            self.data.test, top_k=self.top_k, use_id=True
        )
        ret = []
        for metric in self.metrics:
            if metric == "map":
                ret.append(
                    map_at_k(
                        self.data.test, topk_scores, k=self.top_k
                    )
                )
            elif metric == "ndcg":
                ret.append(
                    ndcg_at_k(
                        self.data.test, topk_scores, k=self.top_k
                    )
                )
            elif metric == "precision":
                ret.append(
                    precision_at_k(
                        self.data.test, topk_scores, k=self.top_k
                    )
                )
            elif metric == "recall":
                ret.append(
                    recall_at_k(
                        self.data.test, topk_scores, k=self.top_k
                    )
                )
           
        return ret
     


    def score(self, user_ids, remove_seen=True):
        """Score all items for test users.
        Args:
            user_ids (np.array): Users to test.
            remove_seen (bool): Flag to remove items seen in training from recommendation.
        Returns:
            numpy.ndarray: Value of interest of all items for the users.
        """
        ## cold start problem
        if any(np.isnan(user_ids)):
            raise ValueError(
                "LightGCN cannot score users that are not in the training set"
            )
        u_batch_size = self.batch_size
        n_user_batchs = len(user_ids) // u_batch_size + 1
        test_scores = []
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = user_ids[start:end]
            item_batch = range(self.data.n_items)
            rate_batch = self.sess.run(
                self.batch_ratings, {self.users: user_batch, self.pos_items: item_batch}
            )
            test_scores.append(np.array(rate_batch))
        test_scores = np.concatenate(test_scores, axis=0)
        if remove_seen:
            test_scores += self.data.rating_matrix.tocsr()[user_ids, :] * -np.inf
        return test_scores

    def recommendKItemsForallUsers(
        self, test, top_k=10, sort_top_k=True, remove_seen=True, use_id=False
    ):
        """Recommend top K items for all users in the test set.
        Args:
            test (pandas.DataFrame): Test data.
            top_k (int): Number of top items to recommend.
            sort_top_k (bool): Flag to sort top k results.
            remove_seen (bool): Flag to remove items seen in training from recommendation.
        Returns:
            pandas.DataFrame: Top k recommendation items for each user.
        """
        data = self.data
        if not use_id:
            user_ids = np.array([data.user2id[x] for x in test[data.col_user].unique()])
        else:
            user_ids = np.array(test[data.col_user].unique())

        test_scores = self.score(user_ids, remove_seen=remove_seen)

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                data.col_user: np.repeat(
                    test[data.col_user].drop_duplicates().values, top_items.shape[1]
                ),
                data.col_item: top_items.flatten()
                if use_id
                else [data.id2item[item] for item in top_items.flatten()],
                data.col_prediction: top_scores.flatten(),
            }
        )

        return df.replace(-np.inf, np.nan).dropna()

    def output_embeddings(self, idmapper, n, target, user_file):
        embeddings = list(target.eval(session=self.sess))
        with open(user_file, "w") as wt:
            for i in range(n):
                wt.write(
                    "{0}\t{1}\n".format(
                        idmapper[i], " ".join([str(a) for a in embeddings[i]])
                    )
                )

    def infer_embedding(self, user_file, item_file):
        """Export user and item embeddings to csv files.
        Args:
            user_file (str): Path of file to save user embeddings.
            item_file (str): Path of file to save item embeddings.
        """
        # create output directories if they do not exist
        dirs, _ = os.path.split(user_file)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        dirs, _ = os.path.split(item_file)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        data = self.data

        self.output_embeddings(
            data.id2user, self.n_users, self.average_user_embeddings, user_file
        )
        self.output_embeddings(
            data.id2item, self.n_items, self.average_item_embeddings, item_file
        )