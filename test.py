import tosgf
from tosgf import ToSgf, Detect_Chess, Preprocess, Recongnition_Num
import unittest

class IntegerArithmeticTestCase(unittest.TestCase):
    pass


path = ToSgf(['./opencvs/src/template.png'])

for p in path.take():
    ins = Preprocess(p)
    tmp = ins.show_board()
    #tosgf.show_img(tmp, 5)
    vertex, area = ins.find_board(tmp)
    board = ins.board(vertex)
    #tosgf.show_img(board, 5)


    ins2 = Detect_Chess(board)
    #ins2.go_matrix() # draw

    ins2.cut()
    # for s in ins2.chess:
    #     #tosgf.show_img(s.img, 1)
    #     tosgf.saveto("chess_"+str(s.x)+"_"+str(s.y)+".png", s.img, "./opencvs/src/")

    ins2.gen_color()
    # for s in ins2.chess:
    #     #tosgf.show_img(s.img, 1)
    #     tosgf.saveto("chess_"+str(s.x)+"_"+str(s.y)+".png", s.img, "./opencvs/tmp/")


    ins2.detect_number()
    def gen_chess_numbers():
        for i in range(len(ins2.chess)):
            img_path = "\\opencvs\\chess\\{0:d}\\".format(i)  # chess_" + str(chess.x) + "_" + str(chess.y) + "\\"
            tosgf.mkdir(img_path)
            for j in range(len(ins2.chess[i].seq)):
                tosgf.saveto("{0:d}.png".format(j), ins2.chess[i].seq[j], img_path)
    #gen_chess_numbers()

    ins3 = Recongnition_Num(ins2.chess)
    # ins3.preprocess_template_images() # move "template/0 , 1, 2 ..., 9" 's *png to "template/dataset"
    ins3.gen_dataset()
    x_train, x_test, y_train, y_test = ins3.gen_train_test()
    #ins3.train_network(x_train, x_test, y_train, y_test)
    
    ins3.load_model()
    
    ins3.gen_chess_num() # succeed! all 279 number was recognized correctly
    # for c in ins3.chess:
    #     print('-----------' + str(c.number))

