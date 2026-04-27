import argparse

def replace_with_N_single(sequence, length):
    seq_len = len(sequence)
    outputs = ['label,sequence\n']
    for i in range(seq_len - length + 1):
        replaced = '1,' + sequence[:i] + 'N' * length + sequence[i + length:]
        outputs.append(f"{replaced}\n")
    return outputs

def replace_with_N_double(sequence, length):
    seq_len = len(sequence)
    outputs = ['label,sequence\n']
    for i in range(seq_len - length + 1 - length ):
        replaced = sequence[:i] + 'N' * length + sequence[i + length:]
        for j in range(i + length  , seq_len - length + 1 ):
            replaced2 = '1,' + replaced[:j] + 'N' * length + replaced[j + length:]
            outputs.append(f"{replaced2}\n")
    return outputs

def replace_with_N_triple(sequence, length):
    seq_len = len(sequence)
    outputs = ['label,sequence\n']
    for i in range(seq_len - length + 1 - length - length ):
        replaced = sequence[:i] + 'N' * length + sequence[i + length:]
        for j in range(i + length, seq_len - length + 1  - length):
            replaced2 = replaced[:j] + 'N' * length + replaced[j + length:]
            for k in range(j + length, seq_len - length + 1 ):
                replaced3 = '1,' + replaced2[:k] + 'N' * length + replaced2[k + length:]
                outputs.append(f"{replaced3}\n")
    return outputs

def replace_with_N_dual(sequence, length, min_region_length):
    seq_len = len(sequence)
    outputs = ['label,sequence\n']

    part1_len = min_region_length
    part2_len = length - min_region_length
    max_start1 = seq_len - part1_len
    max_start2 = seq_len - part2_len

    for start1 in range(0, max_start1 + 1):
        for start2 in range(0, max_start2 + 1):
            # Not to overlap two regions
            end1 = start1 + part1_len
            end2 = start2 + part2_len
            if end1 <= start2 or end2 <= start1:
                replaced = list(sequence)
                replaced[start1:end1] = 'N' * part1_len
                replaced[start2:end2] = 'N' * part2_len
                outputs.append(f"1,{''.join(replaced)}\n")
    return outputs

def generate_replaced(sequence, length, min_region_length, generation_type):
    if generation_type == 'single':
        return replace_with_N_single(sequence, length)
    elif generation_type == 'double':
        return replace_with_N_double(sequence, length)
    elif generation_type == 'triple':
        return replace_with_N_triple(sequence, length)
    elif generation_type == 'dual':
        return replace_with_N_dual(sequence, length, min_region_length)
    else:
        return None


def get_args():
    parser = argparse.ArgumentParser(description="Generation of N-masked ASO sequences")
    parser.add_argument('-s','--sequence', type=str, required=True, help="Input sequence")
    parser.add_argument('-l','--length',   type=int, required=True, help="Length of N region")
    parser.add_argument('-ml','--min_region_length', type=int, default=5, help="Minimum allowed length for each region in a dual-targeting ASO")
    parser.add_argument('-t','--generation_type',    type=str, required=True, help="Generation method: single, double, triple, or dual")
    parser.add_argument('-o','--output', type=str, default='output.csv', help="Output file name")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    sequence = args.sequence
    length = args.length
    min_region_length = args.min_region_length
    generation_type = args.generation_type
    output = args.output
    results = generate_replaced(sequence, length, min_region_length, generation_type)
    with open(output, mode='w') as f:
        f.writelines(results)

if __name__ == '__main__':
    main()


