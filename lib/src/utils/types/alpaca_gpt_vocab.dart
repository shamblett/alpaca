/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

typedef Token = String;
typedef Id = int;

class AlpacaGptVocab {
  Id id = 0;
  Token token = '';

  Map<Token, Id> tokenToId = {};
  Map<Id, Token> idToToken = {};
}
