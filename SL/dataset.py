from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from numpy.testing import assert_array_almost_equal

np.random.seed(15)

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

class mnistNoisy(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, nosiy_rate=0.0, asym=False, imb_type='exp', imb_factor=0.0):
        super(mnistNoisy, self).__init__(root, download=download, transform=transform,
                                           target_transform=target_transform)    
        img_num_list = self.get_img_num_per_cls(10, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        if asym:
            source_class = [7, 2, 3, 5, 6]
            target_class = [1, 7, 8, 6, 5]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(nosiy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = torch.tensor(other_class(n_classes=10, current_class=self.targets[i]))
            print(len(noisy_idx))
        
        print("Print noisy label generation statistics:")
        for i in range(10):
            n_noisy = np.sum(np.array(self.targets) == i)
            print("Noisy class %s, has %s samples." % (i, n_noisy))
        return
        
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        #print(self.data)
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        # self.data = new_data
        self.data = torch.tensor(new_data)
        self.targets = new_targets

class DatasetGenerator():
    def __init__(self, batchSize=128, eval_batch_size=256, dataPath='/datasets',
                 seed=123, numOfWorkers=0, asym=False, cutout_length=16, noise_rate=0.4, imb_type='exp', imb_factor=0.0):
        self.seed = seed
        np.random.seed(seed)
        self.batchSize = batchSize
        self.eval_batch_size = eval_batch_size
        self.dataPath = dataPath
        self.numOfWorkers = numOfWorkers
        self.cutout_length = cutout_length
        self.noise_rate = noise_rate
        self.asym = asym
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        test_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor()])

        train_dataset = mnistNoisy(root=self.dataPath, train=True,
                                        transform=train_transform, download=True,
                                        asym=self.asym, nosiy_rate=self.noise_rate, imb_type=self.imb_type, imb_factor=self.imb_factor)

        test_dataset = datasets.MNIST(root=self.dataPath, train=False,
                                        transform=test_transform, download=True)

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.batchSize,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.numOfWorkers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.numOfWorkers)

        print("Num of train %d" % (len(train_dataset)))
        print("Num of test %d" % (len(test_dataset)))

        return data_loaders
