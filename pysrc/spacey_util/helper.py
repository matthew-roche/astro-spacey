def answer_indexes_from_context(start, end, text_list):
    boundaries = []
    cursor = 0
    for i, piece in enumerate(text_list):
        start_idx = cursor
        end_idx = cursor + len(piece)
        boundaries.append((start_idx, end_idx))
        cursor = end_idx + 1  # +1 for join space

    # Find which pieces overlap with [start, end)
    hit_indices = []
    for i, (s, e) in enumerate(boundaries):
        if not (end <= s or start >= e):  # overlap check
            hit_indices.append(i)
    
    return hit_indices