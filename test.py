from tosgf import ToSgf, Detect_Chess, Preprocess, Recongnition_Num
import os
import multiprocessing
import time
import asyncio
import concurrent

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
        with open(num_path, 'a', encoding='utf-8') as f:
            for i in range(len(recogn.chess)):
                if (i == 100):
                    f.writelines(list(os.linesep))
                f.writelines(list(str(recogn.chess[i].number) + ' '))


def test_prog(path, i):
        pre_img = Preprocess(path)
        tmp = pre_img.show_board()
        ToSgf.show_img(tmp, 5, "Threashold %d" % i)
        vertex, area = pre_img.find_board(tmp)
        board = pre_img.board(vertex)
        ToSgf.show_img(board, 5, "Board %d" % i)
    
        chess = Detect_Chess(board)
        chess.go_matrix() # draw
    
        chess.cut()
        chess.gen_color()
        saveto_test_chess(chess, i)
    
        chess.detect_number()
        saveto_chess_seq(chess, i)
    
        recogn = Recongnition_Num(chess.chess)
        recogn.gen_dataset()
        recogn.load_model()
        recogn.gen_chess_num()
        saveto_chess_num(recogn, i)

def main():
    thread = []
    path = [os.path.join(testpath, 'test1.png'), 
            os.path.join(testpath, 'test2.png'), 
            os.path.join(testpath, 'test3.png')]
    for i in range(len(path)):
        thd = multiprocessing.Process(target=test_prog, args=(path[i], i))
        thread.append(thd)
        time.sleep(i)
        thd.start()
    
    for thd in thread:
        thd.join()

if __name__ == "__main__":
    main()