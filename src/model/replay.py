import random
import numpy as np


def replay_node_selection(args, influence_node_score, topk):
    if args.replay_strategy == 'random':
        return random_sampling(len(influence_node_score), topk)
    elif args.replay_strategy == 'inforeplay':
        return np.argpartition(np.asarray(influence_node_score), topk)[:topk]
    else:
        args.logger.info("repaly node selection mode illegal!")

def random_sampling(data_size, num_samples):
    return np.random.choice(data_size, num_samples)
