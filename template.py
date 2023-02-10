import os
from chess import Recongnition_Num
from log import logger

""" Training model using template"""

def train_template():
    logger.debug("train_template() Starting...")
    recogn = Recongnition_Num([])
    if not os.path.exists(recogn.dataset_path):
        recogn.preprocess_template_images() # move "template/0 , 1, 2 ..., 9" 's *png to "tmp/dataset"
    else:
       logger.warning("Datasets exists! Ignore preprocess_template_images().")
    
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
