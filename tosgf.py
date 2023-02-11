import os
import chess
from chess import Detect_Chess, Preprocess, Recongnition_Num
from log import logger
import template

""" EASY API """
class _Path_pool:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, pool: list) -> None:
        self.path = pool
    def take(self):
        for i in range(0, len(self.path)):
            yield self.path[i]


class ToSgf(_Path_pool):
    def __init__(self, img_paths) -> None:
        super().__init__(img_paths)

    @staticmethod
    def show_img(image, sec=5, name="image"):
        chess.show_img(image, sec, name)
    
    @staticmethod
    def saveto(filename:str, image, path: str):
        chess.saveto(filename, image, path)

    def run(self):
        logger.info("Start run()")

        template.train_template()
        
        for img_path in self.take():
            logger.info("Process image from {0}".format(img_path))
            pre_img = Preprocess(img_path)
            pre_board = pre_img.show_board()
            vertex, area = pre_img.find_board(pre_board)
            board = pre_img.board(vertex)
            pre_img = Detect_Chess(board)
            pre_img.cut()
            logger.info("Cut {0:d} chess circles of board.".format(len(pre_img.chess)))
            pre_img.gen_color()
            pre_img.detect_number()
        
            recogn = Recongnition_Num(pre_img.chess)
            if not os.path.exists(recogn.dataset_path):        
                recogn.preprocess_template_images() # move "template/0 , 1, 2 ..., 9" 's *png to "tmp/dataset"
            else:
                logger.warning("Datasets exists! Ignore preprocess_template_images().")
            
            recogn.gen_dataset()
            
            x_train, x_test, y_train, y_test = recogn.gen_train_test()
        
            if not os.path.exists(recogn.model_path):
                recogn.train_network(x_train, x_test, y_train, y_test)
            else:
                logger.warning("Network exists! Ignore train_network().")
            
            logger.info("Loading model from {0}".format(recogn.model_path))
            recogn.load_model()

            recogn.gen_chess_num()
            logger.info("End processing image from {0}".format(img_path))

            # @TODO: now all chess information was complete. generate SGF format
        logger.info("End run()")
        
