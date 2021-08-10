import math
def twoSum(nums, target):
    hashtable = dict()
    for i, num in enumerate(nums):
        # 如果不存在，加入到 hashtable 中
        hashtable[num] = i
        if target - num in hashtable:
            return [i, hashtable[target - num]]
    return []

def twoSum2(nums, target):
    hashtable = dict()
    for i, num in enumerate(nums):
        if target - num in hashtable:
            return [hashtable[target - num], i]
        hashtable[nums[i]] = i
    return []


def threeSum(nums):
    # 排序+处理，后续避免重复解
    ans = []
    nums.sort()
    n = len(nums)
    # 特殊情况，len < 3
    if n < 3:
        return []
    for i in range(n):
        if (nums[i]) > 0:
            return ans
        if (i > 0 and nums[i] == nums[i - 1]):
            continue
        L = i + 1
        R = n - 1
        while (L < R):
            if (nums[i] + nums[L] + nums[R]) == 0:
                ans.append([nums[i], nums[L], nums[R]])
                while (nums[L] == nums[L + 1] and L < R):
                    L = L + 1
                while (nums[R] == nums[R - 1] and L < R):
                    R = R - 1
                L = L + 1
                R = R - 1
            elif (nums[i] + nums[L] + nums[R]) > 0:
                R = R - 1
            else:
                L = L + 1
    return ans

def threeSum2(nums):
    n = len(nums)
    res = []
    if (not nums or n < 3):
        return []
    nums.sort()
    res = []
    for i in range(n):
        if (nums[i] > 0):
            return res
        if (i > 0 and nums[i] == nums[i - 1]):
            continue
        L = i + 1
        R = n - 1
        while (L < R):
            if (nums[i] + nums[L] + nums[R] == 0):
                res.append([nums[i], nums[L], nums[R]])
                while (L < R and nums[L] == nums[L + 1]):
                    L = L + 1
                while (L < R and nums[R] == nums[R - 1]):
                    R = R - 1
                L = L + 1
                R = R - 1
            elif (nums[i] + nums[L] + nums[R] > 0):
                R = R - 1
            else:
                L = L + 1
    return res

if __name__ == '__main__':
    # nums = [3, 2, 4]
    # target = 6
    nums = [0,0,0]
    # nums = [0,0,0,0]
    # nums = [-1,0,1,2,-1,-4]

    # a = twoSum2(nums, target)
    ans = threeSum2(nums)
    print(ans)
