import ast
import regex as re
from typing import Iterator, Iterable, Optional

class Tokenizer:

    def __init__(self, vocab: dict[int,bytes], merges: list[tuple[bytes,bytes]], special_tokens: Optional[list[str]] = None):
        self.vocab=vocab.copy() # tk_id->byte
        self.vocab_rev={v: k for k,v in vocab.items()} #bytes->tk_id
        self.merges=merges
        self.special_tokens=special_tokens or []

        if special_tokens:
            nxt_id=max(vocab.keys())+1 if vocab else 0
            for special_tk in special_tokens:
                tk_bytes=special_tk.encode('utf-8')
                if tk_bytes not in self.vocab_rev:
                    self.vocab[nxt_id]=tk_bytes
                    self.vocab_rev[tk_bytes]=nxt_id
                    nxt_id+=1

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
            self.special_tk_pat = re.compile('(' + '|'.join(escaped_tokens) + ')')
        else:
            self.special_tk_pat = None

        self.pretokenize_pattern=re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )

    @classmethod
    def from_files(cls, vocab_fpath:str, merges_fpath: str, special_tokens: Optional[list[str]] = None):
        vocab={}
        with open(vocab_fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line =line.strip()
                if not line:
                    continue
                parts=line.split(' ',1)
                tk_id=int(parts[0])
                byte_repr=parts[1]
                tk_bytes=ast.literal_eval(byte_repr)
                vocab[tk_id]=tk_bytes
        
        merges=[]
        with open(merges_fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue

                parts=line.split(' ',1)
                byte_seq1=ast.literal_eval(parts[0])
                byte_seq2=ast.literal_eval(parts[1])
                merges.append((byte_seq1,byte_seq2))

        return cls(vocab,merges,special_tokens)

    def _pretokenize(self, text:str)->list[str]:
        return self.pretokenize_pattern.findall(text)
    
    def _apply_merges(self, pretoken_bytes:bytes)->list[bytes]:
        tokens=[bytes([b]) for b in pretoken_bytes]  

        for mg1,mg2 in self.merges:
            i=0
            while i<len(tokens)-1:
                if tokens[i]==mg1 and tokens[i+1]==mg2 :
                    merged=mg1+mg2
                    tokens=tokens[:i]+[merged]+tokens[i+2:]  
                else:
                    i+=1
        
        return tokens
    
    def _encode_chunk(self, text:str)->list[int]:
        if not text:
            return []
        res=[]
        pre_tokens=self._pretokenize(text)
        for pre_tk in pre_tokens:
            pretk_bytes=pre_tk.encode('utf-8')
            vocab_elements=self._apply_merges(pretk_bytes)
            for ve in vocab_elements:
                if ve in self.vocab_rev:
                    tk_id=self.vocab_rev[ve]
                    res.append(tk_id)
                else:
                    raise ValueError(f"Vocab element {ve} not found")
        return res
    
    def encode(self, text:str)->list[int]:
        if not text:
            return []
        
        res=[]
        if self.special_tk_pat:
            parts=self.special_tk_pat.split(text)
        else:
            parts=[text]

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tk_bytes=part.encode('utf-8')
                tk_id=self.vocab_rev[tk_bytes]
                res.append(tk_id)
            else:
                res.extend(self._encode_chunk(part))
        return res
    
    def encode_iterable(self, iterable:Iterable[str])->Iterator[int]:
        buffer=""
        for chunk in iterable:
            buffer+=chunk
            split_pos=self._find_safe_split(buffer)
            if split_pos>0:
                safe_txt=buffer[:split_pos]
                buffer=buffer[split_pos:]
                tk_ids=self.encode(safe_txt)
                for tk_id in tk_ids:
                    yield tk_id
        
        if buffer:
            tk_ids=self.encode(buffer)
            for tki in tk_ids:
                yield tki

    def _find_safe_split(self, text:str, min_keep:int=100)->int:
        if len(text)<min_keep:
            return 0
        search_st=len(text)-min_keep

        for i in range(len(text)-1,search_st-1,-1):
            if text[i]=='\n':
                return i+1

        for i in range(len(text)-min_keep,-1,-1):
            if text[i].isspace():
                return i+1
        return max(0, len(text)-min_keep)
    
    def decode(self, ids:list[int])->str:
        if not ids:
            return ""
        byte_seq=[]
        for tk_id in ids:
            if tk_id in self.vocab:
                byte_seq.append(self.vocab[tk_id])
            else:
                pass
        all_bytes=b''.join(byte_seq)
        text=all_bytes.decode('utf-8',errors='replace')

        return text


