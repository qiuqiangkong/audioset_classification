import numpy as np

            
class VanillaDataGenerator(object):

    def __init__(self, x, y, batch_size, shuffle=True, seed=1234):
        """
        Args:
          x: ndarray
          y: 2darray
          batch_size: int
          shuffle: bool
          seed: int
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rs = np.random.RandomState(seed)
        
    def generate(self, max_iteration=None):
        
        batch_size = self.batch_size
        
        samples_num = len(self.x)
        indexes = np.arange(samples_num)
        
        if self.shuffle:
            self.rs.shuffle(indexes)
        
        iteration = 0
        pointer = 0
        
        while True:
            
            if iteration == max_iteration:
                break
            
            # Get batch indexes
            batch_idxes = indexes[pointer : pointer + batch_size]
            pointer += batch_size
            
            # Reset pointer
            if pointer >= samples_num:
                pointer = 0
                
                if self.shuffle:
                    self.rs.shuffle(indexes)
            
            iteration += 1
            
            yield self.x[batch_idxes], self.y[batch_idxes]
            
            
class BalancedDataGenerator(object):
    """Balanced data generator. Each mini-batch is balanced with approximately 
    the same number of samples from each class. 
    """
    
    def __init__(self, x, y, batch_size, shuffle=True, seed=1234, verbose=0):
        """
        Args:
          x: ndarray
          y: 2darray
          batch_size: int
          shuffle: bool
          seed: int
          verbose: int
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rs = np.random.RandomState(seed)
        self.verbose = verbose
        
        assert self.y.ndim == 2, "y must have dimension of 2!"
            
    def get_classes_set(self, samples_num_of_classes):
        
        classes_num = len(samples_num_of_classes)
        classes_set = []
        
        for k in range(classes_num):
            classes_set += [k]
            
        return classes_set
        
    def generate(self, max_iteration=None):
        
        y = self.y
        batch_size = self.batch_size

        (samples_num, classes_num) = y.shape
        
        samples_num_of_classes = np.sum(y, axis=0)
        
        # E.g. [0, 1, 1, 2, ..., K, K]
        classes_set = self.get_classes_set(samples_num_of_classes)
        
        if self.verbose:
            print("samples_num_of_classes: {}".format(samples_num_of_classes))
            print("classes_set: {}".format(classes_set))
        
        # E.g. [[0, 1, 2], [3, 4, 5, 6], [7, 8], ...]
        indexes_of_classes = []
        
        for k in range(classes_num):
            indexes_of_classes.append(np.where(y[:, k] == 1)[0])
            
        # Shuffle indexes
        if self.shuffle:
            for k in range(classes_num):
                self.rs.shuffle(indexes_of_classes[k])
        
        queue = []
        iteration = 0
        pointers_of_classes = [0] * classes_num

        while True:
            
            if iteration == max_iteration:
                break
            
            # Get a batch containing classes from a queue
            while len(queue) < batch_size:
                self.rs.shuffle(classes_set)
                queue += classes_set
                
            batch_classes = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            samples_num_of_classes_in_batch = [batch_classes.count(k) for k in range(classes_num)]
            batch_idxes = []
            
            # Get index of data from each class
            for k in range(classes_num):
                
                bgn_pointer = pointers_of_classes[k]
                fin_pointer = pointers_of_classes[k] + samples_num_of_classes_in_batch[k]
                
                per_class_batch_idxes = indexes_of_classes[k][bgn_pointer : fin_pointer]
                batch_idxes.append(per_class_batch_idxes)

                pointers_of_classes[k] += samples_num_of_classes_in_batch[k]
                
                if pointers_of_classes[k] >= samples_num_of_classes[k]:
                    pointers_of_classes[k] = 0
                    
                    if self.shuffle:
                        self.rs.shuffle(indexes_of_classes[k])
                
            batch_idxes = np.concatenate(batch_idxes, axis=0)
            
            iteration += 1
            
            yield self.x[batch_idxes], self.y[batch_idxes]
            
            
if __name__ == '__main__':
    
    x = np.ones((1000, 784))
    y = np.ones((1000, 10))
    
    gen = BalancedDataGenerator(x, y, batch_size=128, shuffle=True, seed=1234)
    
    for (batch_x, batch_y) in gen.generate(max_iteration=3):
        print(batch_x.shape)