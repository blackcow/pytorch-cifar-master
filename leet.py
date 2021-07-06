第一题：

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
#str = input()
#print(str)
class Solution(object):
    def findMedium(l):
        length = len(l)
        l.sort()
        # 如果为奇数，输出中间的值
        if length % 2 != 0:
            print(l[length//2])
        # 如果为偶数，中心两位均值
        else:
            print((l[length//2-1] + l[length//2])/2)

l = [1, 3, 5, 2, 8, 7]
Solution.findMedium(l)

第二题：
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# str = input()
# print(str)
class Solution:
    def maxStr(str_in):
        # 初始化
        length = len(str_in)
        count = [0 for i in range(26)]
        char_a = ord('a')

        # 统计出现次数
        for i in range(length):
            count[ord(str_in[i]) - char_a] += 1

        last = str_in[0]
        num = 1
        res = 1
        for m in range(1, length):
            # 不同
            if last != str_in[m]:
                tmp_idx = m
                while (tmp_idx + 1 < length) and (last == str_in[tmp_idx + 1]):
                    num += 1
                    tmp_idx += 1

                if count[ord(last) - char_a] > num:
                    num += 1
                num, res = 1, max(num, res)
                last = str_in[m]
            # 相同则累加
            else:
                num += 1
        if (num > 1) and (count[ord(last) - char_a] > num):
            num += 1

        # 获取 max 长度后，对 str 遍历访问
        max_length = max(num, res)
        str2ls = list(str_in)
        for i in count:
            if i != max_length:
                str2ls = str2ls[i:]
            else:
                str2ls = str2ls[:max_length]
                out = ''.join(str2ls)
                print(out)
                return (out)


text = 'abbbbcccddddddddeee'
Solution.maxStr(text)

第三题：
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
#str = input()
#print(str)
class Solution:
    def findMaxArray(l):
        # 初始化
        tmp = l[0]
        max_val = tmp
        length = len(l)

        for i in range(1, length):
            # 计算当前序列和，记录当前最大值
            if tmp + l[i] > l[i]:
                max_val = max(max_val, tmp + l[i])
                tmp = tmp + l[i]
            # 否则到此为最长序列，并记录此时最大值
            else:
                max_val = max(max_val, tmp, tmp+l[i], l[i])
                tmp = l[i]
        print(max_val)
        return max_val


l = [1, -2, 4, 5, -1, 1]
Solution.findMaxArray(l)