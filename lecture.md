# LLM Tokenization

Hi everyone, today we are going to look at Tokenization in Large Language Models (LLMs). Sadly, tokenization is a relatively complex and gnarly component of the state of the art LLMs, but it is necessary to understand in some detail because a lot of the shortcomings of LLMs that may be attributed to the neural network or otherwise appear mysterious actually trace back to tokenization.

### Previously: character-level tokenization

So what is tokenization? Well it turns out that in our previous video, [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY), we already covered tokenization but it was only a very simple, naive, character-level version of it. When you go to the [Google colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) for that video, you'll see that we started with our training data ([Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)), which is just a large string in Python:

```
First Citizen: Before we proceed any further, hear me speak.

All: Speak, speak.

First Citizen: You are all resolved rather to die than to famish?

All: Resolved. resolved.

First Citizen: First, you know Caius Marcius is chief enemy to the people.

All: We know't, we know't.
```

But how do we feed strings into a language model? Well, we saw that we did this by first constructing a vocabulary of all the possible characters we found in the entire training set:

```python
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# 65
```

And then creating a lookup table for converting between individual characters and integers according to the vocabulary above. This lookup table was just a Python dictionary:

```python
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers, output a string
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))

# [46, 47, 47, 1, 58, 46, 43, 56, 43]
# hii there
```

Once we've converted a string into a sequence of integers, we saw that each integer was used as an index into a 2-dimensional embedding of trainable parameters. Because we have a vocabulary size of `vocab_size=65`, this embedding table will also have 65 rows:

```python
class BigramLanguageModel(nn.Module):

def __init__(self, vocab_size):
	super().__init__()
	self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

def forward(self, idx, targets=None):
	tok_emb = self.token_embedding_table(idx) # (B,T,C)
```

Here, the integer "plucks out" a row of this embedding table and this row is the vector that represents this token. This vector then feeds into the Transformer as the input at the corresponding time step.

### "Character chunks" for tokenization using the BPE algorithm

This is all well and good for the naive setting of a character-level language model. But in practice, in state of the art language models, people use a lot more complicated schemes for constructing these token vocabularies. In particular, these schemes work not on a character level, but on character chunk level. And the way these chunk vocabularies are constructed is by using algorithms such as the **Byte Pair Encoding** (BPE) algorithm, which we are going to cover in detail below.

Turning to the historical development of this approach for a moment, the paper that popularized the use of the byte-level BPE algorithm for language model tokenization is the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) from OpenAI in 2019, "Language Models are Unsupervised Multitask Learners". Scroll down to Section 2.2 on "Input Representation" where they describe and motivate this algorithm. At the end of this section you'll see them say:

> *The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.*

Recall that in the attention layer of a Transformer, every token is attending to a finite list of tokens previously in the sequence. The paper here says that the GPT-2 model has a context length of 1024 tokens, up from 512 in GPT-1. In other words, tokens are the fundamental "atoms" at the input to the LLM. And tokenization is the process for taking raw strings in Python and converting them to a list of tokens, and vice versa. As another popular example to demonstrate the pervasiveness of this abstraction, if you go to the [Llama 2](https://arxiv.org/abs/2307.09288) paper as well and you search for "token", you're going to get 63 hits. So for example, the paper claims that they trained on 2 trillion tokens, etc.

### Brief taste of the complexities of tokenization

Before we dive into details of the implementation, let's briefly motivate the need to understand the tokenization process in some detail. Tokenization is at the heart of a lot of weirdness in LLMs and I would advise that you do not brush it off. A lot of the issues that may look like issues with the neural network architecture actually trace back to tokenization. Here are just a few examples:

- Why can't LLM spell words? **Tokenization**.
- Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
- Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
- Why is LLM bad at simple arithmetic? **Tokenization**.
- Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
- Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? **Tokenization**.
- What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
- Why did the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
- Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
- Why is LLM not actually end-to-end language modeling? **Tokenization**.
- What is the real root of suffering? **Tokenization**.

We will loop back around to these at the end of the video.

### Visual preview of tokenization

Next, let's load this [tokenization webapp](https://tiktokenizer.vercel.app). What is nice about this webapp is that tokenization is running live in your web browser, allowing you to easily input some text string at the input, and see the tokenization on the right. On the top, you can see that we are currently using the `gpt2` tokenizer, and we see that the string that we pasted in with this example is currently tokenizing into 300 tokens. Here they are shown explicitly in colors:

![tiktokenizer](assets/tiktokenizer.png)

So for example, the string "Tokenization" encoded into the tokens 30642 followed by the token 1634. The token " is" (note that these is three characters, including the space in the front, this is important!) is index 318. Be careful with whitespace because it is absolutely present in the string and must be tokenized along with all the other characters, but is usually omitted in visualization for clarity. You can toggle on and off its visualization at the bottom of the app. In the same way, the token " at" is 379, " the" is 262, etc.

Next, we have a simple example of some arithmetic. Here, we see that numbers may be inconsistently decomposed by the tokenizer. For example, the number 127 is a single token of three characters, but the number 677 because two tokens: the token " 6" (again, note the space in the front!) and the token "77". We rely on the large language model to make sense of this arbitrariness. It has to learn inside its parameters and during training that these two tokens (" 6" and "77" actually combine to create the number 677). In the same way, we see that if the LLM wanted to predict that the result of this sum is the number 804, it would have to output that in two time steps: first it has to emit the token " 8", and then the token "04". Note that all of these splits look completely arbitrary. In the example right below, we see that 1275 is "12" followed by "75", 6773 is actually two tokens " 6", "773", and 8041 is " 8", "041".

(to be continued...)
(TODO: may continue this unless we figure out how to generate it automatically from the video :))

## Exploring the Quirks of Tokenization (08:14)

Tokenization can sometimes be counterintuitive. Take the word "egg" for example. When it stands alone, it becomes two tokens, but when preceded by a space, as in "an egg," it's suddenly a single token. This illustrates how tokenization is case-sensitive and context-dependent. The same word "egg," whether it's at the beginning of a sentence, at the end, in lowercase, uppercase, or mixed case, will be tokenized differently. The language model must learn from the vast amounts of internet text that these variations represent the same concept and group them accordingly in its neural network parameters.

## Language and Tokenization: The Non-English Challenge (09:28)

Tokenization isn't just a challenge for English. Non-English languages, like Korean, often fare worse with language models like ChatGPT. This isn't just because there's more English training data; it's also because the tokenizer itself is trained with more English data. As a result, non-English sentences often require more tokens than their English counterparts, leading to longer token sequences. This "bloats" the sequence length, causing issues with the transformer's attention mechanism, which has a maximum context length. Non-English text, therefore, appears stretched out to the transformer, with more fragmented chunks and boundaries.

## Tokenization and Coding: The Python Example (11:22)

Let's look at Python code and its tokenization. Spaces, which are abundant in Python due to indentation, are each tokenized separately. This is highly inefficient and leads to a bloated sequence length, which is why GPT-2 struggles with Python. It's not a problem with the language model's understanding of coding; it's the tokenization of spaces that eats up too much of the sequence and runs out the context length.

## Exploring Different Tokenizers and Their Impact (12:28)

As we continue our exploration of tokenization, let's consider the impact of different tokenizers on the token count. For instance, the GPT-2 tokenizer generates 300 tokens for a given string. However, when we switch to the GPT-4 tokenizer, the token count significantly drops to 185 for the same string. This reduction is due to the larger vocabulary size in GPT-4, which has roughly doubled from GPT-2's 50K to 100K tokens.

This increase in vocabulary size means that the same text is now represented with fewer tokens, leading to a denser input for the transformer. Since each token in the transformer attends to a finite number of preceding tokens, a denser input allows the model to consider a broader context when predicting the next token. However, there's a trade-off, as a larger vocabulary also means a larger embedding table and a more extensive softmax layer at the output. We'll delve into these details later, but it's important to note that there's an optimal number of tokens that balances density and efficiency.

## Improved Whitespace Handling in GPT-4 (13:57)

A noteworthy improvement in the GPT-4 tokenizer is its handling of whitespace, especially in the context of Python code. In GPT-4, multiple spaces are grouped into a single token, making the representation of Python code more efficient. This design choice by OpenAI allows the model to attend to more code, which contributes to the enhanced coding capabilities of GPT-4 compared to GPT-2. It's not just the language model's architecture or optimization details that have improved; the tokenizer's design plays a significant role in these advancements.

## The Journey from Strings to Tokens (14:56)

Now, let's dive into the process of writing code to tokenize strings for language models. The goal is to convert strings into integers from a fixed vocabulary and then use these integers to look up vectors in a table, which are then fed into the transformer as input. This process becomes complex when we consider the need to support various languages and special characters found on the internet, such as emojis.

## Understanding Unicode and Python Strings (15:42)

In Python, strings are immutable sequences of Unicode code points. But what exactly are Unicode code points? They are defined by the Unicode Consortium as part of the Unicode standard, which currently includes about 150,000 characters across 161 scripts. The standard is continuously updated, with the latest version being 15.1 as of September 2023.

To access the Unicode code point of a character in Python, we use the `ord` function. For example, `ord('H')` gives us the code point 104. This function only works with single characters, not entire strings, and it allows us to look up the integer representation of each character in a string.

## The Challenge of Using Unicode Directly for Tokenization (17:43)

One might wonder why we can't just use these Unicode code points directly for tokenization. The issue is that the vocabulary would be enormous, with 150,000 different code points. Moreover, the Unicode standard is not static; it evolves, which means it's not a stable representation for our purposes. Therefore, we need a more robust solution.

## Encodings: Translating Unicode to Binary Data (18:15)

The Unicode Consortium defines three types of encodings: UTF-8, UTF-16, and UTF-32. These encodings translate Unicode text into binary data or byte streams. UTF-8 is the most common and translates each code point into a byte stream that can be between one to four bytes long, making it a variable-length encoding. UTF-32, on the other hand, is fixed-length but has other downsides.

The full spectrum of pros and cons of these encodings is beyond the scope of this video, but it's worth noting that UTF-8 is preferred for several reasons, including its backward compatibility with ASCII encoding.

## UTF-8 Encoding and Its Implications for Tokenization (20:02)

When we encode a string into UTF-8, we get a byte stream that represents the string according to this encoding. However, if we were to use UTF-8 naively, we'd end up with a vocabulary of only 256 possible tokens, which is very small. This would result in our text being stretched out over long sequences of bytes, leading to tiny embedding tables but very long sequences, which is not ideal given the finite context lengths that transformers can handle.

## UTF-8 and Vocabulary Size (21:21)

When dealing with UTF-8 encoding, we're essentially working with byte streams, which suggests a vocabulary size of only 256 possible tokens. However, this is a very small vocabulary size, and if used naively, it would result in our text being represented by very long sequences of bytes. This would lead to a tiny embedding table and predictions at the final layer, but with the downside of extremely long sequences. Given the finite context lengths that transformers can handle, this would be highly inefficient and prevent the model from attending to sufficiently long text for the next token prediction task. Therefore, we don't want to use raw bytes from UTF-8 encoding.

## The Byte Pair Encoding Algorithm (22:32)

So, what's the solution? We turn to the byte pair encoding (BPE) algorithm, which allows us to compress these byte sequences to a variable extent. While feeding raw byte sequences into language models would be ideal, and there's even a paper discussing the potential of doing so, the reality is that the attention mechanism would become prohibitively expensive due to the length of the sequences. The paper proposes a hierarchical structuring of the transformer to accommodate raw bytes, but this approach hasn't been widely adopted or proven at scale yet. For now, we must rely on BPE to compress our sequences.

## Understanding Byte Pair Encoding (BPE) (23:50)

The byte pair encoding algorithm, despite its importance in the world of language models, is not overly complicated. The Wikipedia page offers a good starting point for understanding the basic concept. Essentially, BPE is a compression algorithm that iteratively merges the most frequent pairs of tokens in a sequence. 

Let's consider a simple example with a vocabulary consisting of only four elements: a, b, c, and d. If we have a sequence that's too long, we can compress it by finding the most common pair of tokens and replacing them with a new token that we add to our vocabulary. For instance, if 'aa' is the most common pair, we create a new token 'Z' to represent it. This process reduces the sequence length while increasing the vocabulary size, but the new token 'Z' stands for the concatenation of 'aa'.

We can repeat this process, identifying the next most frequent pair, say 'ab', and replacing it with another new token 'Y'. This further compresses the sequence and adds another element to our vocabulary. We continue this process until we achieve a desired level of compression.

## Implementing BPE in Practice (26:22)

In practice, we start with byte sequences and a vocabulary size of 256. We then apply the BPE algorithm to find the most common byte pairs and iteratively create new tokens, which we add to our vocabulary. This process not only compresses our training dataset but also provides us with an algorithm to encode any arbitrary sequence using this vocabulary and decode it back into strings.

Now, let's move on to the implementation. I've taken the first paragraph from a blog post and encoded it into UTF-8, converting the bytes to integers for easier manipulation in Python. The original paragraph has 533 code points, but when encoded in UTF-8, it expands to 616 bytes or tokens. This is because some characters are represented by multiple bytes in UTF-8.

The first step in the BPE algorithm is to find the most common pair of bytes. I encourage those following along to try writing this function themselves. My implementation uses a dictionary to keep track of counts and a Pythonic way to iterate over consecutive elements of the list. After identifying the most common pair, which in this case is 'e' followed by a space, we can replace each occurrence with a new token, starting with the ID of 256, as the current tokens range from 0 to 255.

## Iterative Token Merging and Vocabulary Expansion (30:36)

As we delve deeper into the tokenization process, we come across the need to iteratively merge the most common pairs of tokens to create a new token. This is a crucial step in the Byte Pair Encoding (BPE) algorithm. We start by identifying the most common pair in our sequence and decide to mint a new token with an id of 256, since our current tokens range from 0 to 255.

## Implementing the Merging Function (30:41)

The next step is to implement the function that will allow us to replace every occurrence of the identified pair with our new token. This is a critical part of the BPE algorithm, as it simplifies the sequence by reducing redundancy. We iterate over the entire list, and whenever we encounter the pair 101,32, we swap it out for the new token id 256.

## Python Tricks for Token Merging (31:15)

In Python, we can elegantly obtain the highest-ranking pair by using the `max` function on our dictionary of stats. This returns the maximum key based on the values, which represent the frequency of each pair. We then write a function to merge the identified pair throughout the list of ids, replacing it with the new index.

## Handling Edge Cases in Merging (32:24)

When implementing the merging function, we must be careful to handle edge cases, such as avoiding out-of-bounds errors when we reach the end of the list. This requires a conditional check to ensure we don't attempt to access an index that doesn't exist.

## Results of the First Merge (33:34)

After running our merging function, we observe a reduction in the length of our list, which indicates that the merging process is working as intended. We can also verify that the new token id 256 appears in the list, and the original pair 101,32 no longer exists.

## Iterative Merging and Vocabulary Tuning (34:10)

The process of merging common pairs and expanding the vocabulary is iterative. We continue to find the most common pair and replace it with a new token id, incrementing each time. This process is repeated until we reach a desired vocabulary size, which is a hyperparameter that can be tuned based on the specific requirements of the language model.

## Preparing for More Iterations (34:52)

Before we continue with more iterations of merging, we take a step to ensure that our statistics for byte pairs are more representative. By using a longer text, we can get more sensible results, as the statistics will be based on a more extensive and varied dataset.

## The Merging Loop in Action (35:35)

Finally, we put all the pieces together and create a loop that will perform the merging process iteratively. This loop will continue to merge the most common pairs, expanding the vocabulary and simplifying the sequence until we reach our desired vocabulary size, which, as an example, is around 100,000 tokens for state-of-the-art language models like GPT-4.

## Building the Vocabulary for Tokenization (35:42)

In the process of developing a tokenizer, one of the first steps is to decide on the final vocabulary size. This is a critical hyperparameter that can affect the performance of your tokenizer. For our example, we're aiming for a vocabulary size of 276, which means we'll perform exactly 20 merges, starting from the 256 raw byte tokens we begin with.

## Constructing a Binary Forest of Merges (36:52)

The construction of our tokenizer's vocabulary is akin to building a binary forest, where we start with the leaves, which are the individual bytes, and begin merging them. Unlike a tree with a single root, we're creating a forest by merging pairs of tokens, gradually building up a more complex structure.

## The Merging Process and Its Output (37:18)

During the merging process, we find the most commonly occurring pair of tokens and create a new token for them. This new token replaces all instances of that pair in our data. It's important to note that the original tokens can still appear individually; they only become the new token when they occur consecutively. Interestingly, these new tokens are also eligible for future merges, which is how we build up our binary forest.

## Compression Achieved Through Merging (38:39)

After performing 20 merges, we can observe the compression ratio achieved. Starting with 24,000 bytes and ending with 19,000 tokens, we've managed to compress the data, which is a significant aspect of the tokenizer's efficiency.

## Tokenizer Training and Its Separation from LLM (39:17)

It's crucial to understand that the tokenizer is a completely separate entity from the large language model (LLM) itself. The tokenizer has its own training set and undergoes a separate preprocessing stage. We train the tokenizer using the byte pair encoding algorithm, and once trained, it serves as a translation layer between raw text and token sequences, capable of both encoding and decoding.

## Understanding Tokenizer Training and Vocabulary (40:13)

Once you have trained your tokenizer, you have a vocabulary and a set of merges, which allows you to perform both encoding and decoding. The tokenizer serves as a translation layer between raw text, which is a sequence of Unicode code points, and token sequences. It can convert raw text into a token sequence and vice versa, translating a token sequence back into raw text.

## Encoding and Decoding with a Trained Tokenizer (40:40)

Now that we have trained the tokenizer and have these merges, we can look at how to perform the encoding and decoding steps. If you provide text, the tokenizer can give you the corresponding tokens, and if you provide tokens, it can give you the text. This translation between the two realms is crucial, and the language model will be trained as a subsequent step.

Typically, in a state-of-the-art application, you might take all of your training data for the language model, run it through the tokenizer to translate everything into a massive token sequence, and then discard the raw text. You're left with just the tokens, which are stored on disk and are what the large language model reads when training on them. This can be done as a single massive pre-processing stage.

## The Impact of Training Sets on Tokenization (41:33)

The training set for the tokenizer usually differs from that of the large language model. For instance, when training the tokenizer, you might want to include a variety of languages and code because the composition of your tokenizer training set will determine the number of merges and, consequently, the density of different types of data in the token space. If you have a significant amount of data in a particular language, like Japanese, in your tokenizer training set, more tokens from that language will get merged, resulting in shorter sequences for that language. This is beneficial for the large language model, which has a finite context length it can work with in the token space.

## Encoding and Decoding: The Technical Details (42:38)

We're now going to delve into the technical aspects of encoding and decoding with a trained tokenizer. We have our merges, but how do we actually perform encoding and decoding?

Let's start with decoding, which involves converting a token sequence back into a Python string object, the raw text. This function takes a list of integers (tokens) and returns a Python string. There are many ways to implement this function, and it can be a fun exercise to try it yourself.

One approach is to create a pre-processing variable called `vocab`, which is a dictionary in Python mapping the token ID to the bytes object for that token. We start with the raw bytes for tokens from 0 to 255 and then populate the vocab list in the order of all the merges by concatenating the bytes representations of the children tokens.

Given the IDs, we then retrieve the tokens by iterating over all the IDs, using `vocab` to look up their bytes, and concatenating these bytes together to create our tokens. These tokens are raw bytes, so we must decode them using UTF-8 back into Python strings. We call `.decode` on the bytes object to get a string in Python, which we can then return as text.

## UTF-8 Encoding and Decoding Issues (45:15)

In the world of text processing, encoding and decoding can sometimes lead to unexpected errors. Let's explore why certain sequences of IDs could result in an error when decoded. For instance, decoding the single element 97 returns the letter 'a', which is straightforward. However, decoding 128 throws a Unicode decode error, indicating an invalid start byte. This is because the binary representation of 128 doesn't conform to the UTF-8 encoding schema.

UTF-8 has a specific format for multi-byte characters, and if a byte sequence doesn't fit this format, it cannot be decoded as valid UTF-8. The solution to this problem is to use the `errors='replace'` option in Python's decode function, which replaces invalid byte sequences with a special replacement character. This is a common practice and is also used in OpenAI's code. Whenever you see this replacement character in your output, it indicates that something went wrong and the output was not a valid sequence of tokens.

## Implementing Token Encoding (48:22)

We've discussed how to decode tokens into strings, but now let's flip the script. We're going to implement the reverse process: encoding a string into tokens. This is a crucial function for preparing text data to be processed by a language model. The function signature we're interested in will take a string as input and output a list of integers representing the tokens.

If you're up for a challenge, try implementing this yourself. Otherwise, I'm going to walk you through my solution. There are many ways to approach this, but I'll share one method that I've found effective.

## Encoding Text into UTF-8 Bytes (48:55)

The first step in our encoding process is to convert our text into UTF-8 to get the raw bytes. We then convert these bytes into a list of integers, which will serve as our starting tokens. These are the raw bytes of our sequence.

## Merging Bytes According to the Dictionary (49:15)

Now, we need to consider our merges dictionary, which dictates how some bytes may be combined. Remember, the merges were built from top to bottom, so we prefer to do the early merges before the later ones. This is because later merges may rely on the results of earlier ones.

## Finding Merge Candidates (49:53)

We expect to perform several merges, so we'll use a while loop to continuously find pairs of bytes that can be merged. To reuse some of our existing functionality, we'll call the `getStats` function, which counts how many times each pair occurs in our sequence of tokens.

However, we're not interested in the frequency of pairs this time—just the pairs themselves. We'll use the keys of the dictionary returned by `getStats` as our set of possible merge candidates.

## Identifying Pairs for Merging (50:43)

Our goal is to find the pair with the lowest index in the merges dictionary, as we want to perform early merges first. We'll use Python's `min` function over an iterator to find the eligible merging candidate pair in our tokens. If a pair doesn't exist in the merges dictionary, it's not eligible for merging and is assigned an infinite value to ensure it's not chosen as a candidate.

## Handling Non-Mergeable Pairs (53:01)

We must be cautious, as our function could fail if there's nothing left to merge. If all pairs return an infinite value, it means no more pairs can be merged, and we should break out of the loop.

## Performing the Merges (54:09)

Once we've identified a mergeable pair, we look up its index in the merges dictionary and replace occurrences of the pair in our tokens with this index. We continue this process until no more merges can be performed, at which point we return the final list of tokens.

## Conclusion and Testing (54:51)

That wraps up our implementation of the token encoding function. Let's run it and see if it works as expected. For example, the ASCII value 32 represents a space, and it appears that our function has correctly handled this and other characters. Great, it looks like our encoding process is successful!

## Addressing Special Cases in Tokenization (55:10)

As we wrap up this section, it's important to address a special case in the implementation of tokenization. If we encounter a single character or an empty string, we run into an issue because the statistics dictionary (`stats`) would be empty, causing an error with the `min` function. To handle this, we can add a condition that checks if the length of tokens is at least two. If it's less than two, meaning we have a single token or no tokens, there's nothing to merge, so we simply return without doing anything. This adjustment fixes the case and ensures our tokenizer can handle these edge cases.

## Testing the Robustness of Our Tokenizer (55:44)

Now, let's move on to testing our tokenizer to ensure its robustness. A good test is to check if encoding a string and then decoding it gives us back the same string. This should generally be true, but it's important to note that the reverse isn't always the case. Not all strings will maintain their identity when encoded and then decoded. However, we can verify that for the training text we used to train the tokenizer, encoding and decoding do indeed give us back the same text. This gives us confidence that our implementation is correct.

## Exploring Tokenization on Unseen Text (56:41)

To further test our tokenizer, we can try it on text that it has not seen before. For example, by grabbing some text from a webpage that the tokenizer hasn't been trained on, we can check if the encode-decode process still holds up. This is a crucial step to ensure that our tokenizer generalizes well and can handle new data effectively.

## Understanding the Byte Pair Encoding Algorithm (56:48)

We've covered the basics of the Byte Pair Encoding (BPE) algorithm and seen how we can take a training set, train a tokenizer, and create a dictionary of merges that effectively builds a binary forest on top of raw bytes. With this merges table, we can encode and decode between raw text and token sequences. This is the simplest setting of the tokenizer.

## State-of-the-Art Language Models and Their Tokenizers (56:48)

Now, let's shift our focus to state-of-the-art language models and the kinds of tokenizers they use. We'll see that the picture becomes much more complex very quickly. We'll go through the details of this complexification one at a time, starting with the GPT series.

## Exploring Tokenizer Limitations and Solutions (56:48)

In our journey to understand tokenization, we've encountered text that the tokenizer has not been able to decode. This is a crucial aspect to consider because it highlights the limitations of our current tokenization methods. To ensure that our tokenizer can handle unseen text, we need to test it thoroughly, which gives us confidence in its correct implementation.

The basics of the Byte Pair Encoding (BPE) algorithm involve training a tokenizer on a dataset and creating a dictionary of merges. This dictionary effectively creates a binary forest on top of raw bytes. Once we have the merges table, we can encode and decode between raw text and token sequences. This is the simplest setting of the tokenizer.

However, as we delve into state-of-the-art language models, we find that the tokenization process becomes significantly more complex. Let's examine the GPT series, particularly the GPT-2 paper, to understand how tokenization has evolved.

## GPT-2's Approach to Tokenization (56:48)

The GPT-2 paper discusses the tokenizer used for the model, which departs from the naive BPE algorithm. A key issue they address is the merging of common words with punctuation, such as "dog," "dog.", "dog!", etc. The naive BPE algorithm might merge these into single tokens, leading to a clustering of semantics with punctuation, which is suboptimal.

To resolve this, GPT-2 enforces manual rules to prevent certain types of characters from being merged. This is done by creating a regex pattern that specifies parts of the text that should never be merged. The `regex` Python package, an extension of the standard `re` module, is used to compile this pattern.

## Diving into GPT-2's Tokenization Code (59:32)

The GPT-2 tokenizer, found in the `encoder.py` file on GitHub, is responsible for both encoding and decoding text. The regex pattern created by the GPT-2 team is complex, but it serves as the core mechanism for enforcing non-merging rules. By examining this pattern, we can understand how the tokenizer separates text into chunks that are processed independently.

## Understanding Regex Patterns in Tokenization (01:00:07)

The regex pattern used by GPT-2 is designed to match different types of characters and prevent unwanted merges. For example, `\p{L}` matches any letter from any language, while `\p{N}` matches any numeric character. Apostrophes and other punctuation are also considered to ensure that common words and punctuation are not merged into single tokens.

This approach allows the tokenizer to split text into chunks that are processed separately. Each chunk is independently converted into a token sequence, and the results are concatenated. This method ensures that certain combinations, such as a letter followed by a space, are never merged, maintaining the integrity of the tokenization process.

## Understanding Regex Patterns in Tokenization (01:00:45)

In the world of tokenization, regex patterns play a crucial role in how we process and encode strings into tokens for language models like GPT-2. The `re.findall` function is used to match a regex pattern against a string, organizing the matches into a list. This pattern matching is done from left to right, and the pattern itself is composed of several alternatives separated by vertical bars, which represent the OR operation in regex.

For example, the pattern `r"\s?[\p{L}]+"` matches an optional space followed by one or more letters. This is how the string "hello world" would be tokenized into two elements: "hello" and " world". The pattern is designed to prevent certain merges from happening, such as merging a letter with a space, by splitting the text into chunks that are processed independently.

## Tokenization: Separating Letters, Numbers, and Punctuation (01:01:11)

The regex pattern used for tokenization is designed to separate letters, numbers, and punctuation. For instance, `\p{N}` matches any numeric character, ensuring that numbers are tokenized separately from letters. Apostrophes are also handled specifically, with different patterns matching common types of apostrophes. However, this can lead to inconsistencies, especially with Unicode apostrophes, which may not be matched by the pattern and thus become separate tokens.

## Case Sensitivity and Tokenization Inconsistencies (01:06:41)

The GPT-2 documentation suggests that the regex pattern should have included the `re.IGNORECASE` flag to allow for case-insensitive matching of contractions. Without this flag, uppercase and lowercase versions of words with apostrophes are tokenized differently, leading to inconsistencies.

## Tokenizing Python Code and Special Cases (01:10:51)

When tokenizing Python code, the regex pattern results in a list with many elements, as it splits the text frequently whenever a category changes. This ensures that no mergers occur within these elements. Interestingly, spaces are often kept as independent elements and are not merged, which seems to be an additional rule enforced by OpenAI on top of chunking and BPE.

## OpenAI's Tokenization: Inference vs. Training Code (01:10:51)

It's important to note that OpenAI has only released the inference code for the GPT-2 tokenizer, not the training code. This means that while we can apply the known merges to new text, we don't have access to the process used to train the tokenizer itself.

## Introducing the Tick Token Library (01:10:51)

The Tick Token library is OpenAI's official library for tokenization. It provides inference code for tokenization, allowing users to obtain tokens for GPT-2 and GPT-4. One notable difference between the two is that in GPT-4, whitespace characters are merged, unlike in GPT-2.

## Changes in GPT-4 Tokenization Patterns (01:10:51)

In GPT-4, the regex pattern used for chunking text has changed from that of GPT-2. The new pattern includes case-insensitive matching, addressing one of the inconsistencies mentioned earlier. However, the full details of the pattern change are complex and require careful analysis with documentation and tools like ChatGPT to fully understand.

## Understanding OpenAI's Tokenization Rules (01:10:51)

OpenAI has implemented specific rules for tokenization that are not entirely transparent. The training code for the GPT-2 tokenizer was never released, so we only have access to the inference code, which applies predefined merges to new text. The exact methodology OpenAI used to train the tokenizer remains unknown, but it certainly involved more than just chunking and applying Byte Pair Encoding (BPE).

## Introducing the Tick Token Library (01:13:56)

Next, I'd like to introduce you to the Tick Token library from OpenAI, which is the official library for tokenization. This library is used for inference, not training, and it's how you would obtain tokens for GPT-2 or GPT-4. Notably, GPT-4's tokenizer merges whitespace differently than GPT-2's, reflecting a change in the regular expression used to chunk up text.

## GPT-4's Tokenization Changes (01:13:56)

GPT-4 has introduced several changes to its tokenization process. The regular expression pattern has been altered to include case-insensitive matching and to prevent merging of numbers with more than three digits. These changes, along with an increase in vocabulary size from roughly 50k to 100k, are some of the adjustments made in GPT-4's tokenizer.

## Exploring the GPT2Encoder.py File (01:15:05)

The GPT2Encoder.py file released by OpenAI is a key resource for understanding the tokenization process. This file, while somewhat messy, is algorithmically similar to what we've discussed previously. It includes the BPE function, which identifies and merges bigrams in the text, and it's essential for building a BPE tokenizer that can encode and decode text.

## Understanding BPE Merges and Tokenizer Representation (01:16:07)

In the realm of tokenization, BPE (Byte Pair Encoding) merges are a critical component. These merges, which are based on the data within the vocabulary, are equivalent to what OpenAI refers to as 'bpe merges'. Essentially, using just two variables—merges and vocab—you can represent a tokenizer capable of both encoding and decoding once trained. OpenAI's implementation includes an additional layer of byte encoding and decoding, which is used serially with the tokenizer. This extra layer is not deeply significant, so we'll skip over it, but it's important to note that the core of the tokenizer is the BPE function, which is algorithmically identical to what we've built in our own implementation.

## Special Tokens in Tokenization (01:18:57)

Special tokens play a significant role in tokenization. They are used to delimit different parts of the data or to create a special structure within the token streams. For example, OpenAI's GPT-2 uses a special token called 'end of text' to signal the end of a document in the training set. This token is inserted between documents, and the language model must learn to interpret it as a delimiter, indicating that what follows is unrelated to the previous document. The GPT-2 tokenizer has a vocabulary length of 50,257, with the last token being the special 'end of text' token.

## Handling Special Tokens and Extending Tokenizers (01:21:13)

Special tokens are handled differently from the typical BPE merges. They have special case instructions for handling within the code that outputs the tokens. For instance, the 'end of text' token is not processed through BPE merges but is swapped in directly when recognized. This special handling is not present in the basic encoder but can be found in libraries like 'tick token', which is implemented in Rust. Special tokens are not just used in base language modeling but are crucial during the fine-tuning stage and in applications like ChatGPT, where they help delimit conversations between an assistant and a user.

## From GPT-2 to GPT-4: Evolving Tokenization (01:22:35)

As we move from GPT-2 to GPT-4, we see changes in the pattern of tokenization and the addition of new special tokens. GPT-4 introduces tokens like 'thin', 'prefix', 'middle', and 'suffix', each with its specific role and purpose. Adding special tokens requires model surgery to the transformer, ensuring that the embedding matrix and the final projection layer are extended to accommodate the new tokens.

## Expanding Vocabulary and Special Tokens in GPT-4 (01:23:41)

In the evolution from GPT-2 to GPT-4, we've seen not just improvements in the model's architecture but also in its tokenization process. GPT-4 introduces new special tokens such as 'end of text', 'thin', 'prefix', 'middle', and 'suffix'. These tokens are not just arbitrary additions; they serve specific functions that enhance the model's understanding and generation of text. For instance, 'fin' stands for 'fill in the middle', a concept derived from a particular research paper. 

Adding these special tokens isn't a simple task. It requires what's often referred to as 'model surgery'. This involves extending the embedding matrix to accommodate the new tokens, initializing them with small random numbers, and ensuring that the final projection layer of the transformer is also extended. This process is crucial, especially when fine-tuning models for specific tasks, such as converting a base model into a chat model like ChatGPT.

## Building Your Own GPT-4 Tokenizer (01:25:56)

For those interested in building their own GPT-4 tokenizer, I've laid out a path to achieve this. The journey is broken down into four steps, culminating in a tokenizer that can replicate the behavior of the GPT-4 tokenizer. This includes encoding and decoding strings to recover the original text and implementing your own training function for token vocabularies. 

The 'minbpe' repository I've created serves as a guide and reference point for this endeavor. It's a work in progress, and I plan to continue developing it. The repository includes an exercise progression that you can follow, which is detailed in the 'exercise.md' file. This file breaks down the task into manageable steps, and you can always refer back to the 'minbpe' repository for guidance or when you feel stuck.

## Comparing Token Vocabularies and Training Sets (01:31:02)

When training your own tokenizer, the resulting vocabulary will likely resemble that of GPT-4, assuming the same algorithm is used. However, the differences in the training set will lead to variations. For example, GPT-4's training set likely included a significant amount of Python code, influencing its vocabulary to include more whitespace-related tokens. In contrast, a tokenizer trained on a Wikipedia page, such as the one I used for Taylor Swift, will have a different merge order and vocabulary composition.

## Exploring SentencePiece Tokenizer Configuration (01:31:11)

SentencePiece is a tokenizer that has been around for quite some time, and it's known for its versatility in handling a wide range of languages and scripts. However, with its long history comes a significant amount of "historical baggage," including a plethora of configuration options that can be overwhelming. These options are not all necessary for every use case, and many are irrelevant for those working with the byte pair encoding (BPE) algorithm.

For example, the "shrinking factor" option is not used in BPE and is thus irrelevant for our purposes. It's important to sift through these options and identify which ones are applicable to your specific task.

## Setting Up SentencePiece for LLM Training (01:32:28)

When setting up SentencePiece, the goal is to configure it as closely as possible to the way it was used for training large language models like Llama 2. By examining the model file released by Meta, one can inspect the options used and replicate them for their own tokenizer setup.

The input for SentencePiece is raw text, and the output includes both a model file and a vocabulary file. In this case, the BP algorithm is used with a desired vocabulary size of 400. There are numerous configurations for pre-processing and normalization, which were more prevalent in natural language processing tasks like machine translation and text classification before the rise of LLMs.

## The Preference for Raw Data in LLMs (01:33:07)

In the context of language models, the preference is often to avoid normalization and keep the text as raw as possible. This means turning off many of the pre-processing features that SentencePiece offers. The idea is to preserve the original form of the data to the greatest extent possible, which is a common preference among deep learning practitioners.

## Understanding SentencePiece Tokenization (01:33:37)

SentencePiece is a tokenization library that introduces the concept of sentences as individual training examples. It was developed early on with the idea that tokenizers are trained on a bunch of independent sentences. This approach includes parameters for the number of sentences to train on, maximum sentence length, and shuffling sentences. However, in the context of Large Language Models (LLMs), the distinction of sentences seems spurious and unnecessary. Sentences do exist in raw datasets, but defining what exactly constitutes a sentence can be challenging, especially across different languages. Therefore, it might be more practical to treat a file as a giant stream of bytes without imposing the concept of sentences.

## SentencePiece's Treatment of Rare Words and Special Tokens (01:34:40)

SentencePiece also has specific rules for dealing with rare word characters, or more accurately, code points. It includes merge rules that are somewhat equivalent to using regular expressions to split up categories, such as digits and whitespace. Additionally, SentencePiece allows for the indication of special tokens, hardcoding tokens like the unknown token (UNK), beginning of sentence (BOS), end of sentence (EOS), and a pad token. The UNK token is essential and must exist in the system.

## Training and Inspecting SentencePiece Vocabularies (01:35:33)

When training a SentencePiece model, it generates a model file and a vocabulary file. By inspecting the vocabulary, we can see the individual tokens that SentencePiece will create. The vocabulary starts with special tokens like UNK, BOS, and EOS, followed by byte tokens representing the 256 byte values. After the byte tokens come the merge tokens, which are the parent nodes in the merges. The individual code point tokens, which are encountered in the training set, are listed at the end. Extremely rare code points are ignored and not added to the vocabulary.

## Encoding and Decoding with SentencePiece (01:37:41)

Once a vocabulary is established, SentencePiece can encode text into token IDs and decode tokens back into text. An example is provided with the encoding and decoding of the phrase "Hello, 안녕하세요." The Korean characters, not being part of the training set, result in unknown tokens. However, with byte fallback enabled, SentencePiece falls back to bytes, encoding the characters with UTF-8 and using tokens to represent those bytes. The UTF-8 encoding is shifted by three due to the special tokens with earlier IDs.

## Understanding Byte Fallback in Tokenization (01:38:22)

In the world of tokenization, we sometimes encounter code points that do not have a token associated with them. These are what we call unknown tokens. But there's a mechanism called byte fallback that comes to the rescue. When byte fallback is set to true, the tokenizer, instead of giving up on these unknown tokens, falls back to bytes. It encodes the characters using UTF-8 and then uses tokens to represent those bytes.

This is what we're seeing here with the UTF-8 encoding. The encoding is shifted by three because of the special tokens that have IDs earlier on. This is a clever way to ensure that even if a character isn't recognized, it can still be represented in the tokenization process.

## The Impact of Disabling Byte Fallback (01:38:59)

Now, let's explore what happens when we disable byte fallback. If we set byte fallback to false and retrain our tokenizer, we'll notice that all the byte tokens disappear. This is because we're no longer reserving space in our vocabulary for bytes, which allows for more merges and a more efficient use of our vocabulary space.

However, without byte fallback, if we try to encode a string that contains characters not recognized by our tokenizer, we get a zero. This is because the tokenizer labels the entire string as unknown, denoted by "unk," and the unk token is token zero.

This poses a significant challenge for the language model. How is it supposed to handle various unrecognized elements that are rare and end up being mapped to the same unk token? This is not an ideal property for a tokenizer, as it can lead to confusion and inaccuracies in the language model's understanding and generation of text.

## The Importance of Byte Fallback in LLMs (01:39:43)

The decision to use byte fallback is crucial in the context of large language models (LLMs). Without it, any unrecognized character would be mapped to the unk token, leading to a potential loss of information and nuance in the text being processed. Byte fallback allows for a more granular representation of text, even when dealing with rare or unusual characters.

This is why the Llama model, for instance, correctly utilized byte fallback. It's a way to ensure that the language model has the best chance of accurately representing and understanding the full spectrum of text it encounters, including those pesky rare characters that might otherwise be lost in translation.

