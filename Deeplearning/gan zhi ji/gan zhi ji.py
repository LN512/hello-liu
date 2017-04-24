trainint_set = [[(3,3),1],[(4,3),1],[(1,1),-1]]
w = [0,0]
b = 0


def update(item):
    global w,b
    w[0] +=1 * item[1] * item[0][0]
    w[1] +=1 * item[1] * item[0][1]
    b += 1 * item[1]
    print("w=",w,"b=",b)

def judge(item):
    res = 0
    for i in range(len(item[0])):
        res += item[0][i]*w[i]
        res += b
        res *= item[1]
        return res
def check():
    flag = False
    for item in trainint_set:
        if judge(item)<=0:
            flag = True
            update(item)
    return flag
if __name__=='__main__':
    flag = False
    for i in range(1000):
        if not check():
            flag = True
            break
    if flag:
        print("在1000次内全部分对")
    else:
        print("错误")