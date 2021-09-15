import math
def splitList(l):
    """
    :type l: List
    :rtype: Int
    """
    # minimum val of sublist3
    l.reverse()
    meanval = math.ceil(sum(l)/3)
    ans = 0
    tmp3, tmp2, tmp1 = 0, 0, 0
    for i in range(len(l)):
        # 找到可行的 sublist3，使得 sum 值大于 meanval
        while(i < l-2):
            tmp3 = tmp3 + l[i]
            # 在找到满足的 sublist3 后，计算可行的 sublist1，2
            if tmp3 >= meanval:
                for m in range(i,len(l)):
                #  计算满足 sublist2 > sublist1 的可行解
                   while(m < l-i):
                    tmp2 += l(m)
                    tmp1 += l(l-m)
                    if tmp2 > tmp1:
                        ans += 1
                        tmp2, tmp1 = 0, 0


if __name__ == '__main__':
    l = [1, 2, 2, 3, 3, 5]
    ans = splitList(l)
    print(ans)
