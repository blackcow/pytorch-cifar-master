import math

# 跟顺序有关
def pearson(vector1, vector2):
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    #分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

if __name__ == '__main__':

    # #0-8
    # v1 = [0.935,0.902,0.859,0.707]
    # v2 = [0.978,0.973,0.973,0.972] #1
    #
    # # v1 = sorted(v1)
    # # v2 = sorted(v2)
    #
    # val = pearson(v1, v2)
    # print(val)
    s= 'abcda'

    occ = set()
    ans = 0
    idx_right = -1
    start = 0
    length = len(s)

    for start in range(length):

        if start != 0:
            occ.remove(s[start - 1])

        while idx_right < length and s[idx_right + 1] not in occ:
            occ.add(s[idx_right])
            idx_right += 1
        ans = max(ans, idx_right - start - 1)
    print(ans)