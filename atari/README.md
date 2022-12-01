# Atari

## Deep TAMER
This implements a version of [Deep TAMER](../literature/Project/Deep_TAMER_2018.pdf) presented in a paper titled *Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces* by Garrett Warnell, Nicholas Waytowich, Vernon Lawhern, and Peter Stone in 2018. Some key differences are listed below:
- Rather than using an autoencoder to learn feature embeddings from the raw pixels of the Atari game, features provided from the OpenAI RAM observation type are used
- This implementation does not perform SGD on the reward model using random samples from a feedback replay buffer

## Files
| File              | Description                                           |
| ----------------- | ----------------------------------------------------- |
| atari_play.py     | play an atari game                                    |
| atari_env.py      | atari environment object                              |
| tamer.py          | train an agent to play an atari game using Deep TAMER |
| nn_init.py        | initializes the reward model for Deep TAMER           |
| models/           | models trained with Deep TAMER                        |
| tests/            | test python files                                     |
