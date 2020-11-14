def split_into_chunks(tokens, label_ids, words_map, spliting_strategy, max_token_lenght):
    if spliting_strategy == 'max_token_lenght':
        chunks = []
        # TODO maybe don't split on token from one words to different chunks?
        for i in range(0, len(tokens), max_token_lenght):
            chunks.append((
                tokens[i:i+max_token_lenght],
                label_ids[i:i+max_token_lenght],
                words_map[i:i+max_token_lenght])
            )
    elif spliting_strategy == 'overlapping':
        chunks = []
        # TODO maybe add padding here?
        for i in range(0, len(tokens), max(max_token_lenght//3, 1)):
            chunks.append((
                tokens[i:i+max_token_lenght],
                label_ids[i:i+max_token_lenght],
                words_map[i:i+max_token_lenght])
            )
    else:
        assert len(tokens) <= max_token_lenght
        chunks = [(tokens, label_ids, words_map)]
    return chunks