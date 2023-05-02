# alpaca
An alpaca.cpp implementation in Dart

This s port of the main alpaca.cpp project repository [here]( https://github.com/antimatter15/alpaca.cpp), 
as such you should read its [README](https://github.com/antimatter15/alpaca.cpp/blob/master/README.md) before 
reading further.

A few points note about this implementation

1. If you need to build your own libggml.so library see the README in the lib/src/ggml/implementation directory.
   This is a simple task, the ggml code only has dependencies on standard linux system libraries.


Note that the application is linux only until this [issue](https://github.com/shamblett/alpaca/issues/1) is resolved.


2. The model file referenced in the alpaca README should be renamed to ggml-model-q4_0.bin and copied to a
   directory named 'model' that you must create in the top level of the package. The model file is also available [here](https://huggingface.co/Pi3141/alpaca-7B-ggml). You will need 
   git-lfs installed to download these files.

Running the application is a simple matter of typing 'dart bin/chat.dart' from the top level of the package.

RAM usage is currently higher than its C++ counterpart, currently it settles at around 14GB and creeps forward per quastion.
Alpaca chat settles at around 12GB and stays at that level throughout the session. Clearly some memory management
updates are still needed, these are being pursued on this [issue](https://github.com/shamblett/alpaca/issues/3).

Feel free to contact me about any of this, contact details [here](https://www.darticulate.com/), also I'm on Mastodon now at
@shamblett@darticulate.com. Please also feel free to raise issues, pull requests etc.




