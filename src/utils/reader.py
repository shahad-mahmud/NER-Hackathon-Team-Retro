def read_ner_data(file_path: str):
    file = open(file_path, 'r')
    lines, tags = [[]], [[]]

    for line in file:
        line = line.strip()
        info = line.split()

        if not line:
            assert len(lines[-1]) == len(tags[-1])
            
            if not lines[-1]:
                continue

            lines.append([])
            tags.append([])
            continue

        word, tag = info[0], info[-1]

        lines[-1].append(word)
        tags[-1].append(tag)

    return lines, tags
