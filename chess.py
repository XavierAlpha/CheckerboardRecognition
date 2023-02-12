# Author: XavierAlpha
import uuid
import os
import glob
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

from log import logger

""" Generate Chess Information"""


class Preprocess:
    """
    one instance preprocess one image index by 'path'
    maybe using multithreading 
    """
    def __init__(self, path) -> None:
        self.path = path
        logger.debug("Read image from {0}".format(path))
        self.src = cv2.imread(path, cv2.WINDOW_AUTOSIZE)
    
    def show_board(self):
        logger.debug("show_board() starting...")
        bgr = self.src.copy()
        bgr_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([10, 0, 0])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(bgr_hsv, lower, upper)
        dilate_bgr = cv2.dilate(mask, (3, 3), iterations=1)
        #dilate_bgr = cv2.dilate(erode_bgr, None, iterations=2)

        bgr = cv2.bitwise_and(bgr, bgr, mask=dilate_bgr) # show board roughly
        bgr2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr2 = cv2.GaussianBlur(bgr2, (3, 3), 0)
        _, dst = cv2.threshold(bgr2, 100, 255, cv2.THRESH_BINARY)
        logger.debug("Return threashold image.")
        logger.debug("show_board() finished.")
        return dst
    
    def find_board(self, image):
        logger.debug("find_board() starting...")
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rect, area = None, 0 # board arc location and its area
        logger.debug("Finding the biggest square in contours.")
        for item in contours:
            hull = cv2.convexHull(item)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):             
                ps = np.reshape(approx, (4,2))
                ps = ps[np.lexsort(((ps[:, 1]), (ps[:, 0])))]
                lt, lb, rt, rb = ps[0], ps[1], ps[2], ps[3]
                s = cv2.contourArea(approx) # 计算四边形面积
                if s > area:
                   area = s
                   rect = (lt, lb, rt, rb)
        vertex = []
        for coordinate in rect:
            vertex.append(list(coordinate))
        logger.debug("find_board() finished. Return coordinate of board.")
        return (vertex, area)
    
    def board(self, vertex):
        logger.debug("board() starting...")
        lt = vertex[0]
        lb = vertex[1]
        rt = vertex[2]
        rb = vertex[3] # not used
        logger.debug("board() finished. Return new board image.")
        return self.src.copy()[lt[1]:lb[1], lt[0]:rt[0]]


class Recongnition_base:
    def __init__(self, board: cv2.Mat, size: int=19) -> None:
        self.board = board
        self.size = size
        self.chess_pd = None
    
    def coordinate(self, loc_x, loc_y):
        # location (x,y) belong to which coordinate?
        chess_pd = self.chess_pd
        g = [(x, y) for x in range(0, 20) if loc_x // chess_pd == x for y in range(0, 20) if loc_y // chess_pd == y ]
        if len(g) < 1:
            raise IndexError("LOCATION DATA ERRORS! OUT OF BOUNDS!")
        else: 
            return g[0]

    def location(self, coor_x, coor_y):
        # return the center of chess
        chess_pd = self.chess_pd
        return ((coor_x + 0.5) * chess_pd, (coor_y + 0.5) * chess_pd)
    
    def chess_diameter(self, diameter=None, Margin=1):
        """
        If known accurate diameter of each chess, use it directly; Or caculate roughly;
        Margin/2 mean the value of the distance between the outermost chess and the nearest boarder of board. If known accurate diameter, ignore it.
        """
        logger.debug("chess diameter() starts...")
        width, length = self.board.shape[0], self.board.shape[1] # accurately width==length
        rough = (min(width, length) - 2 * Margin) // self.size # chess pieces diameter
        if diameter and diameter <= rough + 1 and diameter >= rough - 1:
            logger.debug("Use input diameter.")
            self.chess_pd = diameter
        else:
            logger.debug("Use inner caculating result")
            self.chess_pd = rough
        logger.debug("chess diameter() finished...")
    
    def go_matrix(self):
        """
        image -> matrix[19x19]
        its number represents black or white
        """
        chess_pd = self.chess_pd
        len_of_square = chess_pd * self.size

        # Draw line for test
        board = self.board.copy()
        start_x_loc = [(i * chess_pd, 0) for i in range(self.size + 1)]
        end_x_loc = [(i * chess_pd, len_of_square) for i in range(self.size + 1)]
        pair_line = list(zip(start_x_loc, end_x_loc))
        start_y_loc = [(0, j * chess_pd) for j in range(self.size + 1) ]
        end_y_loc = [(len_of_square, j * chess_pd) for j in range(self.size + 1)]
        pair_line2 = list(zip(start_y_loc, end_y_loc))
        for (p1, p2) in pair_line:
            cv2.line(board, p1, p2, (255,0,0), 2)
        for (p1, p2) in pair_line2:
            cv2.line(board, p1, p2, (255,0,0), 2)
        show_img(board, 3, "Board With Line: Test chess size datas")

    def circle_detect(self):
        """
        Detect chess circles in board.
        @return numpy.ndarray [[centers_x, centers_y, radius], ...]
        """
        gray_board = cv2.cvtColor(self.board, cv2.COLOR_BGR2GRAY)
        blur_board = cv2.medianBlur(gray_board, 7)
        circles = cv2.HoughCircles(blur_board, cv2.HOUGH_GRADIENT, 1, self.chess_pd // 2, param1=40, param2=10, minRadius=self.chess_pd // 2 - 1, maxRadius= self.chess_pd // 2 + 1)
        return np.uint16(np.around(circles[0]))
    
    def draw_circle(self, circles: list):
        circles = np.uint16(np.around(circles))
        board = self.board.copy()
        for i in circles:
            cv2.circle(board, (i[0],i[1]), i[2], (0, 255, 0), 1)
            cv2.circle(board, (i[0],i[1]), 2, (0, 0, 255), 3)
        show_img(board)


class Chess():
    """
    Store each chess's coordinate and its image
    """
    def __init__(self, coor_x, coor_y, image) -> None:
        self.__coor_x = coor_x
        self.__coor_y = coor_y
        self.__chess = image
        self.__seq_img = []
        self.__color = None
        self.__number = None
    
    @property
    def x(self):
        return self.__coor_x
    @property
    def y(self):
        return self.__coor_y

    @property
    def img(self):
        return self.__chess
    @img.setter
    def img(self, img):
        self.__chess = img

    @property
    def color(self):
        return self.__color
    @color.setter
    def color(self, color):
        self.__color = color
    
    @property
    def seq(self):
        """ Sequence number image was split in list. """
        return self.__seq_img
    @seq.setter
    def seq(self, seq):
        self.__seq_img = seq
    
    @property
    def number(self):
        return self.__number
    @number.setter
    def number(self, num):
        self.__number = num


class Detect_Chess(Recongnition_base):
    def __init__(self, board: cv2.Mat, size: int = 19) -> None:
        logger.debug("Initialing Detect_Chess instance...")
        super().__init__(board, size)
        self.__chess = []
    
    @property
    def chess(self) -> list:
        return self.__chess

    def cut(self):
        """
        Cutting board image to peices of circles.
        Store to chess property.
        """
        logger.debug("cut() starting...")
        circles = self.circle_detect()
        for circle in circles:
            src_cp = self.board.copy()
            mask = np.zeros_like(src_cp, dtype=np.uint8)
            cv2.circle(mask, (circle[0], circle[1]), circle[2], (255,255,255), -1)
            ROI = cv2.bitwise_and(src_cp, mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            x, y, w, h = cv2.boundingRect(mask)
            res = ROI[y:y+h, x:x+w]
            mask = mask[y:y+h, x:x+w]
            res[mask==0] = (255,255,255)

            self.chess.append(Chess(circle[0], circle[1], res))
        logger.info("Append chess to list.")
        logger.debug("cut() finished.")
    
    def gen_color(self):
        """
        Refresh color information and write to chess.color.
        And change its background color adapted to image processing.
        store to color property.
        """
        logger.debug("gen_color() starting...")
        for chess in self.chess:
            img = cv2.cvtColor(chess.img.copy(), cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
            counter_black = 0
            # O(n^2), too slow, any better way?
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] == 0:
                        counter_black = counter_black + 1
            chess.color = 0 if counter_black / img.size > 0.5 else 255
            
            # after detecting color, it's neccessary to change background, copy from "cut()" partly.
            if chess.color == 0: # black
                """
                colors in chess are like this: from outer to inner
                square -> chess -> number in chess
                and its color
                white->black->black
                """
                mask = np.zeros_like(chess.img, dtype=np.uint8)
                cv2.circle(mask, (chess.img.shape[0] // 2, chess.img.shape[1] // 2), chess.img.shape[0] // 2, (255,255,255), -1) # a white circle
                chess.img = cv2.bitwise_xor(chess.img, mask) # although it looks like white now, but color show the orginal real color
            # white no need to change
        logger.debug("gen_color() finished.")

    def detect_number(self):
        """
        Spilt numbers in a single chess image;
        Store to its "seq" property
        """
        logger.debug("detect_number() starting...")
        for chess in self.chess:
            # preprocess image
            img = cv2.cvtColor(chess.img.copy(), cv2.COLOR_RGB2GRAY)
            img = cv2.GaussianBlur(img, (3,3), 0)
            _, img = cv2.threshold(img, 95, 255, cv2.THRESH_BINARY_INV)
            img = cv2.dilate(img, (3,3), iterations=1)

            # NOTE: findContours find white contours from black background, so it should use "_INV" when threshold.
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Element: [(area, (x,y,w,h)) , ...]
            __area_rect = [(cv2.boundingRect(item)[2] * cv2.boundingRect(item)[3], cv2.boundingRect(item)) for item in contours]
            __area_rect = sorted(__area_rect, key= lambda i: i[0])

            result = [] # store information of each chess: [(area, (x,y,w,h)), ...]

            # screening the real numbers, may contains some impurities
            if len(__area_rect) > 3: # the number of chess is up to three, so, remove smaller ones
                __area_rect = __area_rect[-3:]

            if len(__area_rect) == 3: # at least one is not zero
                s1 = __area_rect[0][0]
                s2 = __area_rect[1][0]
                s3 = __area_rect[2][0]
                if s1 == 0: 
                    __area_rect.pop(0) # and come to len()==2 condition
                else:
                    result.append(__area_rect[2])
                    if (s3 - s2) / s3 > 0.75: # only one satisfying
                        __area_rect.pop(0)
                        __area_rect.pop(1)

                    elif (s2 - s1) / s2 < 0.75: # check the smallest one is satisfying or not
                        result.append(__area_rect[1])
                        result.append(__area_rect[0]) # Index 0 Is statisfying
                    else:
                        result.append(__area_rect[1]) # Index 0 is not statisfying

            if len(__area_rect) == 2: # at least one is not zero
                s1 = __area_rect[0][0]
                s2 = __area_rect[1][0]
                if (s1 == 0): 
                    __area_rect.pop(0) # and come to len()==1 condition
                else:
                    result.append(__area_rect[1])
                    if (s2 - s1) / s2 < 0.75: # meaning the smaller one is satisfying
                        result.append(__area_rect[0])
            
            if len(__area_rect) == 1: # at leatest one is satisfying
                result.append(__area_rect[0])
            
            # store the result to chess's property
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            seq = []
            #full_nums = img.copy() # don't change the chess.img when drawing rectangle
            result = sorted(result, key=lambda x: x[1][0]) # sorted by location 'x' for number order
            for item in result:
                x, y, w, h = item[1]
                #cv2.rectangle(full_nums, (x, y), (x + w, y + h), (0,0,255), 1) # draw lines in "full_nums" image for showing the whole numbers
                seq.append(img.copy()[y:y+h, x:x+w])
            #show_img(full_nums, 1)
            chess.seq = seq # write to chess's property
        logger.debug("detect_number() finished.")
        logger.debug("Set the 'seq' property. ")


class Recongnition_Num():
    """
    Recongnition the number of chess.
    Store to number property.
    """
    def __init__(self, chess: list) -> None:
        self.chess = chess
        self._datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp'+os.sep+'dataset')
        self._modelpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp'+os.sep+'model')
        self._model = []

    @property
    def model(self):
        return self._model

    @property
    def dataset_path(self):
        return self._datapath
    
    @property
    def model_path(self):
        return self._modelpath

    def take_template_images(self, path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template')) -> str:
        """
        Swallow the template images of number 0,1,2...,9, rename and move to tmp/dataset.
        Also if path not default, the format of directory is like this:
        ```
        -'path'
          - '0'
             *.png
          - '1'
             *.png
          ...
          - '9'
             *.png
        ```
        """
        logger.debug("preprocess_template_images() starting...")
        if not os.path.isdir(path):
            raise FileExistsError("{0} Not Exists or Not A Directory".format(path))
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        
        abs = os.path.abspath(path)
        dirs = glob.glob(os.path.join(abs, '[0-9]'))
        dirs = sorted(dirs, key = lambda fn: os.path.basename(fn)) # sorted by filename
        if len(dirs) != 10:
            raise FileExistsError("Directory from 0 - 9 not satisfied")
        
        last_index = 0
        count = 0
        if not os.path.exists(os.path.join(self.dataset_path, 'tmp')):
            with open(os.path.join(self.dataset_path, 'tmp'), 'w', encoding='utf-8') as f:
                f.write(str(0))
        with open(os.path.join(self.dataset_path, 'tmp'), 'r', encoding='utf-8') as f:
            try:
                last_index = int(f.read())
            except IOError as e:
                logger.error(e)
            except TypeError as e:
                logger.error(e)
            except ValueError as e:
                logger.error(e)
            except Exception as e:
                logger.error(e)
            else:
                logger.debug("Read template image count from tmp succeed.")
        
        for i in range(len(dirs)):
            for p in glob.iglob(os.path.join(dirs[i], '*.png')):
                count += 1 
                os.rename(p, os.path.join(self.dataset_path, '{0:d}_{1:d}_{2}.png'.format(i, last_index + count, str(uuid.uuid4()))))
        
        if count > 0:
            with open(os.path.join(self.dataset_path, 'tmp'), 'w', encoding='utf-8') as f:
                try:
                    f.write(str(last_index + count))
                except IOError as e:
                    logger.error(e)
                except Exception as e:
                    logger.error(e)
                else:
                    logger.debug("Write count to tmp succeed.")
        
            logger.warn("Original images were moved to {0}".format(self.dataset_path))
            logger.debug("Counting {0:d} template images.".format(last_index + count  + 1))
        else:
            logger.debug("No new template images read")
        logger.debug("preprocess_template_images() finished.")
    
    def gen_dataset(self):
        """
        save dataset to datasets.csv.
        """
        logger.debug("gen_dataset() starting...")
        dataset = pd.DataFrame(columns=["pixel{0:d}".format(i) for i in range(28*28)])
        for p in glob.iglob(os.path.join(self.dataset_path, '*.png')):
            (dir, filename) = os.path.split(p)
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28,28), cv2.INTER_LINEAR)
            index = filename.split('_')[0]
            
            d_tmp = pd.DataFrame(np.array(img).reshape((1, 28*28)), index=list(index), columns=dataset.columns)
            dataset = pd.concat([dataset, d_tmp], axis=0, ignore_index=False) # add one image to dataset index 

        dataset.index.name = 'label'
        # save to .csv
        logger.debug("Save the datasets.csv to {0}".format(self.dataset_path))
        pd.DataFrame.to_csv(dataset, os.path.join(self.dataset_path, "datasets.csv"))
        logger.debug("gen_dataset() finished.")

    def gen_train_test(self):
        """
        generate the train and test datas from dataset.csv.
        """
        logger.debug("gen_train_test() starting...")
        dataset = pd.read_csv(os.path.join(self.dataset_path, "datasets.csv"))
        y_train = dataset["label"]
        x_train = dataset.drop(labels = ["label"],axis = 1)
        x_train = x_train / 255.0
        x_train = x_train.values.reshape(-1,28,28,1)
        y_train = to_categorical(y_train, num_classes = 10)

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=23)
        logger.debug("Generate x, y train and test datas.")
        logger.debug("gen_train_test() finished.")
        return (x_train, x_test, y_train, y_test)
    
    def train_network(self, x_train, x_test, y_train, y_test, model_number=3):
        """
        train network.
        save the model to 'model_path'/mode/.
        """
        logger.debug("train_network() starting...")
        datagen = ImageDataGenerator(
        rotation_range=20,  
        zoom_range = 0.20,  
                width_shift_range=0.2, 
                height_shift_range=0.2)
        
        # BUILD CONVOLUTIONAL NEURAL NETWORKS
        nets = model_number #
        model = [0] *nets
        for j in range(nets):
            model[j] = Sequential()
        
            model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
            model[j].add(BatchNormalization())
            model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Dropout(0.4))
        
            model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Dropout(0.4))
        
            model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Flatten())
            model[j].add(Dropout(0.4))
            model[j].add(Dense(10, activation='softmax'))
        
            # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
            model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        
        # DECREASE LEARNING RATE EACH EPOCH
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
        # TRAIN NETWORKS
        history = [0] * nets
        epochs = 65 # 45
        for j in range(nets):
            x_train2, x_val2, y_train2, y_val2 = x_train, x_test, y_train, y_test
            history[j] = model[j].fit(datagen.flow(x_train2,y_train2, batch_size=64),
                epochs = epochs, steps_per_epoch = x_train2.shape[0]//64,  
                validation_data = (x_val2,y_val2), callbacks=[annealer], verbose=0)
            print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
                j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
            
            # save to 'model_path'/mode_{j}.h5
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            logger.debug("Save the model to {}.".format(self.model_path))
            model[j].save(os.path.join(self.model_path, "model_{0:d}.h5".format(j)))
        logger.debug("train_network() finished.")

    def load_model(self, model_number=3):
        for i in range(model_number):
            logger.debug("Loading model {0}...".format(i))
            self._model.append(tf.keras.models.load_model(os.path.join(self.model_path, "model_{0:d}.h5".format(i))))
        logger.debug("Loading all {} models!".format(model_number))

    def predict(self, img):
        # preprocess input image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28,28), cv2.INTER_LINEAR)
        p_img = img.reshape(-1,28,28,1).astype("float32") / 255

        vals = []
        for model in self.model:
            p = model.predict(p_img)
            number = np.argmax(p[0])
            if p[0][number] <= 0.9: # fails if anyone not satisfied
                return -1
            vals.append(number)
        most_common_num = Counter(vals).most_common(1)
        return most_common_num[0][0]
    
    def gen_chess_num(self):
        logger.debug("gen_chess_num() starting...")
        count = 0
        for c in self.chess:
            number = []
            for img in c.seq:
                num = self.predict(img)
                if (num != -1):
                    number.append(str(num))
            try:
                c.number = int(''.join(number))
            except Exception as e:
                logger.error(e)
                logger.error("ERROR! There are empty images of number in chess.seq, check 'seq' directory!")
                logger.error("Which means the number on chess is not detected and cut into pieces correctly!")
            count += 1
        logger.debug("Totally {0} numbers generated.".format(count))
        logger.debug("Finish setting every chess's number property")
        logger.debug("gen_chess_num() finished.")

def show_img(image, sec=5, name="image"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    k = cv2.waitKey(sec * 1000)
    cv2.destroyAllWindows()

def saveto(filename:str, image, path: str):
    """
    save image to path (default current work directory), named "filename"
    """
    path = os.path.abspath(path)
    if os.path.exists(path):
        cv2.imwrite(os.path.join(path, filename), image)
    else:
        raise NotADirectoryError

def mkdir(path):
    path = os.getcwd() + path
    if not os.path.exists(path):
        os.mkdir(path)
