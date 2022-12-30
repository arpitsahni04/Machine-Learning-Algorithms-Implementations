import lr
import math

if __name__ == '__main__':

    # ans = round((-1/4)*(math.log(lr.sigmoid(2)) + math.log(lr.sigmoid(3)) +
    #                     math.log(1-lr.sigmoid(1.5)) + math.log(1-lr.sigmoid(1)) ),4)
    
    # print(ans)
    ans = (-1/4)*(lr.sigmoid(1.5)*1)
    # ans = (-1/4)*(lr.sigmoid(2-1)*1 + lr.sigmoid(3-1)*1)
    # ans = (-1/4)*(lr.sigmoid(1)*1 + lr.sigmoid(3-1)*1)
    ans = round(ans,4)
    print(ans)
    # print(1.5 +0.2044)
    # print(2+0.403)
    # print(1+0.403)