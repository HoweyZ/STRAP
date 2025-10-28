import tqdm
import numpy as np


def z_score(data):
    """
    Calculate the standardized value of the data, that is, subtract the mean from the data 
    and divide it by the standard deviation to ensure that the data follows a standard normal distribution in a statistical sense
    """
    return (data - np.mean(data)) / np.std(data)

def generate_dataset(data, idx, x_len=12, y_len=12):
    """"
    Generates a dataset of input x and output y from the input data at the given index idx
    """
    res = data[idx]  # Get data by index
    node_size = data.shape[1]  # Get the number of nodes
    t = len(idx)-1
    idic = 0
    x_index, y_index = [], []  # Initialize the x and y index lists
    
    # Traverse the index to generate the index of x and y
    for i in tqdm.tqdm(range(t, 0, -1)):
        if i-x_len-y_len>=0:
            x_index.extend(list(range(i-x_len-y_len, i-y_len)))
            y_index.extend(list(range(i-y_len, i)))

    x_index = np.asarray(x_index)  # Convert to numpy array
    y_index = np.asarray(y_index)
    x = res[x_index].reshape((-1, x_len, node_size))  # Reshape the data
    y = res[y_index].reshape((-1, y_len, node_size))
    
    return x, y

def generate_samples(days, savepath, data, graph, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False):
    """
    Generate training, validation and test datasets and save them as .npz files
    """
    edge_index = np.array(list(graph.edges)).T  # Get the edge index of the graph and transpose it
    del graph
    
    if savepath.split('/')[1] =='PEMS':
        data = data[0:days*288, :]  # Extract data based on days
            
            
            
    t, n = data.shape[0], data.shape[1]  # Get the time step and number of nodes of the data
    
    # Split the training, validation, and test set indices according to the ratio
    train_idx = [i for i in range(int(t*train_rate))]
    #train_idx = [i for i in range(int(t*0.2))] # for few-shot setting
    val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
    test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]
    
    train_x, train_y = generate_dataset(data, train_idx)
    val_x, val_y = generate_dataset(data, val_idx)
    test_x, test_y = generate_dataset(data, test_idx)
    
    # If you need to mix validation and test sets
    if val_test_mix:
        val_test_x = np.concatenate((val_x, test_x), 0)  # Combine validation and test sets x data
        val_test_y = np.concatenate((val_y, test_y), 0)  # Combine validation and test set y data
        val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])  # Generate Index
        np.random.shuffle(val_test_idx)  # Shuffle the index order
        val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]  # Re-partition validation and test set data
        test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

    # Normalize the data to z-scores
    """
    Important Note:
    It would be more reasonable to use only the mean and standard deviation of the training data to normalize the validation and test sets.
    However, for consistency reasons, we follow TrafficStream's approach, which is currently acceptable.
    """
    train_x = z_score(train_x)
    val_x = z_score(val_x)
    test_x = z_score(test_x)
    
    # Save data to file
    np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
    
    # Build the returned data dictionary
    data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}
    return data