import sys, random, torch, logging
import numpy as np
import os.path as osp
from utils import common_tools as ct


def init(args):
    '''
    Step 1.1 : Initialize configuration parameters
    '''
    def _update(src, tmp):
        # Iterate over each key-value pair in the tmp dictionary, and if the key is not "gpuid", add it to the src dictionary
        for key in tmp:
            if key!= "gpuid":
                src[key] = tmp[key]
    
    conf_path = osp.join(args.conf)  # Concatenate the configuration file path into a complete path
    info = ct.load_json_file(conf_path)  # Loading a configuration file in JSON format
    _update(vars(args), info)  # Update the configuration information to the args parameter
    vars(args)["path"] = osp.join(args.model_path, args.logname+"-"+str(args.seed))  # Create a model save path and store it in the args parameter
    ct.mkdirs(args.path)  # Create the corresponding directory
    del info  # Delete the configuration information dictionary


def seed_anything(seed=42):
    '''
    Step 1.2: Initialize random seed
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_log(args):
    '''
    Step 1.3: Initialize the logging object
    '''
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)  
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger

