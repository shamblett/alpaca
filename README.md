# alpaca
An alpaca.cpp implementation in Dart

This s port of the main alpaca.cpp project repository [here]( https://github.com/antimatter15/alpaca.cpp), 
as such you should read its [README](https://github.com/antimatter15/alpaca.cpp/blob/master/README.md) before 
reading further.

A few points note about this implementation

1. If you need to build your own libggml.so library see the README in the lib/src/ggml/implementation directory.
   This is a simple task, the ggml code only has dependencies on standard linux system libraries.


Note that the application is linux only until this [issue](https://github.com/shamblett/alpaca/issues/1) is resolved.


2. The model file referenced in the alpaca README should be renamed to ggml-model-q4_0.bin and copied to the
   model directory. The model file is also available [here](https://huggingface.co/Pi3141/alpaca-7B-ggml). You will need 
   git-lfs installed to download these files.

Running the application is a simple matter of typing 'dart bin/chat.dart'.

Note this [issue](https://github.com/shamblett/alpaca/issues/2), until this is resolved you will need plenty(64GB at least) of RAM and only
ask one question at a time, hopefully I'll come up with a fix for this soon.

Feel free to contact me about any of this, contact details [here](https://www.darticulate.com/), also I'm on Mastodon now at
@shamblett@darticulate.com. Please also feel free to raise issues, pull requests etc.




