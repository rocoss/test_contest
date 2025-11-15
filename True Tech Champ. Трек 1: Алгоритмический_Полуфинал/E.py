import sys

MOD = 10**9 + 7

def main():
    input_data = sys.stdin.read().split()
    index = 0
    t = int(input_data[index])
    index += 1
    for _ in range(t):
        n = int(input_data[index])
        index += 1
        k_bin = input_data[index]
        index += 1
        # k_bin is the binary string, left MSB, right LSB
        k_bin = k_bin.zfill(n)
        # Now k_bin[0] MSB bit n-1, k_bin[n-1] LSB bit 0
        S = []
        for j in range(n):
            if k_bin[j] == '1':
                i = n - 1 - j  # bit position from low
                S.append(i)
        s = len(S)
        # Now S list of positions, sort it
        S.sort()
        n_u = n - s
        pow2_n_u = pow(2, n_u, MOD)
        # Build popcount_prefix: pop[i] = number of S < i (for i=0 to n)
        popcount_prefix = [0] * (n + 1)
        s_idx = 0
        for i in range(1, n + 1):
            popcount_prefix[i] = popcount_prefix[i - 1]
            if s_idx < s and S[s_idx] == i - 1:
                popcount_prefix[i] += 1
                s_idx += 1
        # Inside S
        if s >= 2:
            c_s_2 = s * (s - 1) // 2
            total_ans = pow(2, c_s_2, MOD)
        else:
            total_ans = 1
        # For each v
        for v in S:
            # n_v_less = |U < v| = v - |S < v| = v - popcount_prefix[v]
            # Note: popcount_prefix[v] = |S cap [0, v-1]|
            num_s_less = popcount_prefix[v]
            n_v_less = v - num_s_less
            # choices = 2^{n_u} - 2^{n_v_less}
            pow2_less = pow(2, n_v_less, MOD)
            choices_v = (pow2_n_u - pow2_less + MOD) % MOD
            total_ans = (total_ans * choices_v) % MOD
        print(total_ans)

if __name__ == "__main__":
    main()
