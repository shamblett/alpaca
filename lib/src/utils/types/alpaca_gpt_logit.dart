/*
 * Package : alpaca
 * Author : S. Hamblett <steve.hamblett@linux.com>
 * Date   : 03/04/2023
 * Copyright :  S.Hamblett
 */

part of alpaca;

class AlpacaGptLogit extends Comparable {
  Id id = 0;
  double val = 0.0;

  @override
  String toString() => 'Id : $id, Val : $val \n';

  bool operator <(AlpacaGptLogit other) => other.val < val;

  bool operator >(AlpacaGptLogit other) => other.val > val;

  @override
  int compareTo(other) => other.val > val ? -1 : 1;
}
