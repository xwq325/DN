import os
import random


def make_list(path, val_set):
    train_list = []
    val_list = []
    lists = os.listdir(path)
    for i in range(len(lists)):
        if val_set > 1:
            if i < val_set:
                val_list.append(os.path.join(path, lists[i]))
            else:
                train_list.append(os.path.join(path, lists[i]))
        else:
            if random.random() < val_set:
                val_list.append(os.path.join(path, lists[i]))
            else:
                train_list.append(os.path.join(path, lists[i]))
    write_file('train.txt', train_list)
    write_file('val.txt', val_list)


def write_file(path, lists):
    file = open(path, 'w')
    file.write('\n'.join(lists))
    file.close()


if __name__ == "__main__":
    make_list('Flickr2K', 100)
