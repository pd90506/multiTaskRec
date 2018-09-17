import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.MSELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        ratings = ratings.view(-1, 1)
        #print(ratings_pred, ratings)
        loss = self.crit(ratings_pred, ratings)
        loss.backward()
        self.opt.step()
        #if self.config['use_cuda'] is True:
        #    loss = loss.data.cpu().numpy()[0]
        #else:
        #    loss = loss.data.numpy()[0]
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
            rating = rating.float()

            if self.config['use_cuda'] is True:
                user = user.cuda()
                item = item.cuda()
                rating = rating.cuda()
            loss = self.train_single_batch(user, item, rating)
            if batch_id % 1000 == 0:
                print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)
        print('traub_an_epoch')

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        test_users, test_items = Variable(evaluate_data[0]), Variable(evaluate_data[1])
        negative_users, negative_items = Variable(evaluate_data[2]), Variable(evaluate_data[3])
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
        test_scores = self.model(test_users, test_items)
        negative_scores = self.model(negative_users, negative_items)
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)