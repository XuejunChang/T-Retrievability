import math
import argparse

def compute_gini(map_values):
    values = sorted(map_values.values())
    n = len(values)
    if n == 0:
        return 0.0

    total = sum(values)
    if total == 0:
        return 0.0

    gini_numerator = sum((2 * (i + 1) - n - 1) * val for i, val in enumerate(values))
    return gini_numerator / (n * total)


def build_log_reciprocal_rank_map(filename):
    rr_map = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # skip invalid lines
            docid = parts[2]
            try:
                rank = int(parts[3]) + 1
                if rank <= 0:
                    continue  # avoid invalid rank
                rr = 1.0 / math.log(1 + rank)
                rr_map[docid] = rr_map.get(docid, 0.0) + rr
            except (ValueError, ZeroDivisionError, OverflowError):
                continue  # skip malformed lines
    return rr_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Gini coefficient.")
    parser.add_argument("filename", help="Path to the input space-separated file")
    args = parser.parse_args()

    rr_map = build_log_reciprocal_rank_map(args.filename)
    gini = compute_gini(rr_map)
    print(f"Gini coefficient: {gini:.6f}")

