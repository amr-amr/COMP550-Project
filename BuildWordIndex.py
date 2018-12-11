def build_word_index(list_of_token_lists):
    word_index = {}
    word_index["<PAD>"] = 0
    word_index["<OOV>"] = 1

    i = 2
    for token_list in list_of_token_lists:
        for token in token_list:
            if token not in word_index:
                word_index[token] = i
                i += 1

    return word_index