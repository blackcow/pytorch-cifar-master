def minSpeed(piles, H):
    def eatTime(K):
        ans = 0
        for banana in piles:
            # banana = max(piles)
            if banana % K > 0:
                ans += banana // K + 1
            else:
                ans += banana // K
        # print(ans)
        if ans <= H:  # True
            return True
        else:  # False
            return False

    low_speed = 1
    high_speed = max(piles)
    ans_speed = 1e10
    for i in range(high_speed):
        if eatTime(i + 1) and ans_speed > i + 1:
            ans_speed = i + 1
            break
    return ans_speed

# print(minSpeed(piles=[3, 6, 7, 11], H=8))
# print(minSpeed(piles=[30, 11, 23, 4, 20], H=5))
print(minSpeed(piles=[30, 11, 23, 4, 20], H=6))

# print(eatTime(29, piles=[30, 11, 23, 4, 20], H=5))
# print(eatTime(30, piles=[30, 11, 23, 4, 20], H=5))
