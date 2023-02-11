import tosgf
from tosgf import ToSgf, Detect_Chess, Preprocess, Recongnition_Num
import os

testpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')

def saveto_test_chess():
    if not os.path.exists(os.path.join(testpath, 'chess')):
        os.mkdir(os.path.join(testpath, 'chess'))
    for s in chess.chess:
        #tosgf.show_img(s.img, 1)
        ToSgf.saveto("chess_"+str(s.x)+"_"+str(s.y)+".png", s.img, os.path.join(testpath, 'chess'))

def saveto_chess_seq():
    if not os.path.exists(os.path.join(testpath, 'seq')):
        os.mkdir(os.path.join(testpath, 'seq'))
    for i in range(len(chess.chess)):
        img_path = os.path.join(testpath, "seq"+os.sep+"{0:d}".format(i))
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        for j in range(len(chess.chess[i].seq)):
            ToSgf.saveto("{0:d}.png".format(j), chess.chess[i].seq[j], img_path)

def saveto_chess_num():
        with open(num_path, 'wa', encoding='uft-8') as f:
            for i in range(len(chess.chess)):
                num_path = os.path.join(testpath, "numbers.txt")
                f.writelines(list(chess.chess[i].number))
    
path = ToSgf([os.path.join(testpath, 'test.png')])
for p in path.take():
    pre_img = Preprocess(p)
    tmp = pre_img.show_board()
    ToSgf.show_img(tmp, 5, "Threashold")
    vertex, area = pre_img.find_board(tmp)
    board = pre_img.board(vertex)
    ToSgf.show_img(board, 5, "Board")

    chess = Detect_Chess(board)
    chess.go_matrix() # draw

    chess.cut()
    chess.gen_color()
    saveto_test_chess()

    chess.detect_number()
    saveto_chess_seq()

    recogn = Recongnition_Num(chess.chess)

    recogn.gen_dataset()
    
    x_train, x_test, y_train, y_test = recogn.gen_train_test()

    recogn.load_model()
    recogn.gen_chess_num()
    saveto_chess_num()

