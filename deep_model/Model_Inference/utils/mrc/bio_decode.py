from typing import Tuple, List

class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})

def bio_decode(char_label_list: List[Tuple[str, str]]) -> List[Tag]:
    """
    decode inputs to tags
    Args:
        char_label_list: list of tuple (word, bmes-tag)
    Returns:
        tags
    Examples:
        >>> x = [("Hi", "O"), ("Beijing", "B-LOC")]
        >>> bio_decode(x)
        [{'term': 'Beijing', 'tag': 'LOC', 'begin': 1, 'end': 1}]
    """
    idx = 0
    length = len(char_label_list)
    # print(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]
        
        # merge chars
        if current_label == "O":
            idx += 1
            continue
        
        # if current_label == "O":
        #     end = idx+1
        #     while end < length and char_label_list[end][1] == "O":
        #         end += 1
        #     entity = "".join(char_label_list[i][0] for i in range(idx, end))
        #     tags.append(Tag(entity, label[0], idx, end-1))
        #     idx = end
        
        elif current_label == "B":
            end = idx + 1
            while end < length and char_label_list[end][1][0] == "I":
                end += 1
            entity = "".join(char_label_list[i][0] for i in range(idx, end))
            tags.append(Tag(entity, label[2:], idx, end-1))
            idx = end
        else:
            # print(current_label,idx)
            raise Exception("Invalid Inputs")
    return tags
