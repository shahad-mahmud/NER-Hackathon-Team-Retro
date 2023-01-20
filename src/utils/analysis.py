def multi_tag_words(distribution: dict):
    result = {}
    
    for k, v in distribution.items():
        if len(v) > 1:
            result[k] = v
    
    return result