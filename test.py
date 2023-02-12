from tosgf import ToSgf, Detect_Chess, Preprocess, Recongnition_Num
import os
import multiprocessing
from log import logger
import glob

""" Test to complete chess's information"""

testpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')

def saveto_test_chess(chess, i):
    if not os.path.exists(os.path.join(testpath, 'chess{0:d}'.format(i))):
        os.mkdir(os.path.join(testpath, 'chess{0:d}'.format(i)))
    for s in chess.chess:
        #tosgf.show_img(s.img, 1)
        ToSgf.saveto("chess_"+str(s.x)+"_"+str(s.y)+".png", s.img, os.path.join(testpath, 'chess{0:d}'.format(i)))

def saveto_chess_seq(chess, k):
    if not os.path.exists(os.path.join(testpath, 'seq{0:d}'.format(k))):
        os.mkdir(os.path.join(testpath, 'seq{0:d}'.format(k)))
    for i in range(len(chess.chess)):
        img_path = os.path.join(testpath, 'seq{0:d}'.format(k)+os.sep+"{0:d}".format(i))
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        for j in range(len(chess.chess[i].seq)):
            ToSgf.saveto("{0:d}.png".format(j), chess.chess[i].seq[j], img_path)

def saveto_chess_num(recogn, i):
        num_path = os.path.join(testpath, "numbers{0:d}.txt".format(i))
        number = [str(chess.number) for chess in recogn.chess]
        number.sort(key=lambda i: int(i))
        number = ' '.join(number)
        with open(num_path, 'a', encoding='utf-8') as f:
            f.writelines(number)

def test_prog(path, i):
        pre_img = Preprocess(path)
        tmp = pre_img.show_board()
        ToSgf.show_img(tmp, 5, "Threashold %d" % i)
        vertex, area = pre_img.find_board(tmp)
        board = pre_img.board(vertex)
        ToSgf.show_img(board, 5, "Board %d" % i)
    
        chess = Detect_Chess(board)
        chess.chess_diameter()
        chess.go_matrix() # draw
    
        chess.cut()
        chess.gen_color()
        saveto_test_chess(chess, i)
    
        chess.detect_number()
        saveto_chess_seq(chess, i)
    
        recogn = Recongnition_Num(chess.chess)
        recogn.load_model()
        recogn.gen_chess_num()
        saveto_chess_num(recogn, i)

def test_chess_information():
    """ 
    Source files in test: test1.png, test2.png and test3.png;
    The chess images were saved to test/chess{i}, and after cutting to pieces of number, it was saved to seq{i};
    The result of each test{i}.png was saved to numbers{i}.txt, which shows sequence number of each chess.
    """
    process = []
    path = glob.glob(os.path.join(testpath, "*.png"))
    for i in range(0, len(path)//2 * 2, 2):
        task_group = path[i:i+2]
        for j in range(2):
            pro = multiprocessing.Process(target=test_prog, args=(task_group[j], i+j))
            logger.info("Process {0:d} executes!".format(i+j))
            pro.start()
            process.append(pro)
        
        process[-2].join()
        process[-1].join()

    if len(path) % 2 == 1:
        pro = multiprocessing.Process(target=test_prog, args=(path[-1], len(path)-1))
        pro.start()
        pro.join()

if __name__ == "__main__":
    test_chess_information()
    