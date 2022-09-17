import torch
from clip import clip

from ordinalclip.models.ordinalclip import load_clip_to_cpu


def test_clip_model():
    clip_model = load_clip_to_cpu(text_encoder_name="RN50", image_encoder_name="RN50", root=".cache/clip")

    text = "I am a robot."
    all_tokens = clip.tokenize(text)  # (num_sentence, context_legnth=77)
    tokens = all_tokens[0]  # (context_legnth=77)
    # sot: 49406; eot: 49407; null: 0
    # num embedings: 49408

    print(text)
    print(all_tokens[0, :10])

    num_words = len(text.split(" "))
    num_tokens = torch.argmax(tokens) - 1
    print(f"num words: {num_words}, num_tokens: {num_tokens}")

    embedding_dim = clip_model.token_embedding.embedding_dim
    token_embeds = clip_model.token_embedding(tokens)  # (context_length=77, embedding_dim)
    context_embeds = token_embeds[1 : 1 + num_tokens, :]
