def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str])->tuple[dict[int,bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, 'rb') as f:
        text_bytes=f.read()
    text_str=text_bytes.decode('utf-8')

    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words=re.findall(PAT, text_str)

    from collections import Counter
    word_freq=Counter(words)
    
    word_tokens={}
    for word, freq in word_freq.items():
        word_bytes=word.encode('utf-8')
        token_seq=tuple(word_bytes) # "abs" -> (64,37,37)
        word_tokens[token_seq]=freq 

    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])
    nxt_tid=256
    for x in special_tokens:
        vocab[nxt_tid]=x.encode('utf-8')
        nxt_tid+=1

    def cnt_pairs(word_tokens):
        pair_cnt=Counter()
        for token_seq, freq in word_tokens.items():
            for i in range(len(token_seq)-1):
                pair=(token_seq[i], token_seq[i+1])
                pair_cnt[pair]+=freq
        return pair_cnt
    
    def mfr_pair(pair_cnt, word_tokens):
        if not pair_cnt:
            return None
        max_freq = max(pair_cnt.values())
        most_freq_pairs = [pair for pair, freq in pair_cnt.items() if freq == max_freq]
        return max(most_freq_pairs, key=lambda p: (vocab[p[0]], vocab[p[1]])) #This was the error that gpt resolved
    
    def merge(word_tokens, pair, new_token_id):
        new_word_tokens={}
        tk1, tk2=pair
        for token_seq, freq in word_tokens.items():
            new_seq=[]
            i=0
            while i<len(token_seq):
                if(i<len(token_seq)-1 and token_seq[i]==tk1 and token_seq[i+1]==tk2):
                    new_seq.append(new_token_id)
                    i+=2
                else:
                    new_seq.append(token_seq[i])
                    i+=1

            new_word_tokens[tuple(new_seq)]=freq
        return new_word_tokens
    
    merges=[]
    while len(vocab)<vocab_size :
        pair_count=cnt_pairs(word_tokens)
        mfp=mfr_pair(pair_count, word_tokens)
        if mfp is None:
            break
        tk1_bytes=vocab[mfp[0]]
        tk2_bytes=vocab[mfp[1]]
        new_token_id=nxt_tid
        nxt_tid+=1
        vocab[new_token_id]=tk1_bytes+tk2_bytes
        merges.append((tk1_bytes, tk2_bytes))
        word_tokens=merge(word_tokens, mfp, new_token_id)

    return vocab, merges
    

