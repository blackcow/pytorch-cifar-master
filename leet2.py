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
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    # 排序+处理，后续避免重复解
    ans = []
    nums.sort()
    # 特殊情况，len < 3
    if len(nums) < 3:
        return []

    for i, num in enumerate(nums):
        # L\R 的起始位置
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        L = i + 1
        R = len(nums) - 1
        if num > 0 or L == R:
            return ans
        while L != R:
            sum_all = nums[i] + nums[L] + nums[R]
            if sum_all == 0:
                ans.append([nums[i], nums[L], nums[R]])
                L = L + 1
                # break
            elif sum_all > 0:
                R = R - 1
                while nums[R] == nums[R - 1] and L != R:
                    R = R - 1
            elif sum_all < 0:
                L = L + 1
                while nums[L] == nums[L - 1] and L != R:
                    L = L + 1
    return ans

if __name__ == '__main__':
    # nums = [3, 2, 4]
    # target = 6
    # nums = [0,0,0]
    nums = [0,0,0,0]
    # nums = [-1,0,1,2,-1,-4]

    # a = twoSum2(nums, target)
    ans = threeSum(nums)
    print(ans)
