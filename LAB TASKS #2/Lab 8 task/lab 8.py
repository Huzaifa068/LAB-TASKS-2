def minmax(depth, is_maximizing, scores):
    if depth == 0: 
        return scores[0]
    
    if is_maximizing:
        return max(minmax(depth - 1, False, scores[:len(scores)//2]),
                   minmax(depth - 1, False, scores[len(scores)//2:]))
    else:
        return min(minmax(depth - 1, True, scores[:len(scores)//2]),
                   minmax(depth - 1, True, scores[len(scores)//2:]))

scores = [3, 5, 6, 9, 1, 2, 0, -1]  
depth = 3 

optimal_score = minmax(depth, True, scores)
print("Optimal Score:", optimal_score) 
