from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm
import os


def download_dataset():
    # Skip if corpus already exists
    if os.path.exists("corpus.txt"):
        print("corpus.txt already exists. Skipping dataset download.")
        return
    print("Downloading dataset...")
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train"
    )
    print("Writing corpus.txt...")
    with open("corpus.txt", "w", encoding="utf-8") as f:
        for item in tqdm(dataset):
            text = item["text"].strip()
            if text:
                f.write(text + "\n")
    print("Dataset saved to corpus.txt")


def train_tokenizer():
    # Skip if tokenizer already exists
    if os.path.exists("tiny_llm.model"):
        print("Tokenizer already exists. Skipping tokenizer training.")
        return
    print("Training tokenizer...")
    spm.SentencePieceTrainer.train(
        input="corpus.txt",
        model_prefix="tiny_llm",
        vocab_size=8000,
        model_type="bpe",
        character_coverage=1.0,

        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print("Tokenizer training complete")

def tokenize():
    if os.path.exists("tokens.txt"):
        print("tokens.txt already exists. Skipping tokenization.")
        return
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load("tiny_llm.model")
    print("Loading dataset...")
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train"
    )

    print("Tokenizing dataset...")

    total_tokens = 0

    with open("tokens.txt", "w") as f:
        for item in tqdm(dataset):
            tokens = sp.encode(item["text"])
            token_line = " ".join(map(str, tokens))
            f.write(token_line + "\n")
            total_tokens += len(tokens)

    print("Tokenization complete")
    print("Total tokens:", total_tokens)

if __name__ == "__main__":
    download_dataset()
    train_tokenizer()
    tokenize()