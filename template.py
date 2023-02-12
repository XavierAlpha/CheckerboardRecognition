import os
from chess import Recongnition_Num
from log import logger

""" Training model using template"""

def train_template(path: list):
    logger.debug("train_template() Starting...")
    recogn = Recongnition_Num([])
    recogn.take_template_images() # default path
    for p in path: # self-defined path
        recogn.take_template_images(p) # move "self-defined path/0 , 1, 2 ..., 9" 's *png to "tmp/dataset"
    
    if not os.path.exists(os.path.join(recogn.dataset_path, 'datasets.csv')):
        recogn.gen_dataset()
    else:
        logger.warning("Datasets.csv exists! Ignore gen_dataset().")
    
    x_train, x_test, y_train, y_test = recogn.gen_train_test()

    if not os.path.exists(recogn.model_path):
        recogn.train_network(x_train, x_test, y_train, y_test)
    else:
        logger.warning("Network exists! Ignore train_network().")

    logger.debug("train_template() finished.")
