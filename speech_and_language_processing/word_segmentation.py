#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@version:0.1
author: Yang Liu
@time: 2019/10/16
@file: word_segmentation.py
@function:
@modify:
"""

def max_match(sentence, dictionary):
    """A simple algorithm for segmenting Chinese sentence.

    Args:
        sentence (str): input string to be segmented.
        dictionary (list): list of valid words.

    Returns:
        list: segmented sentence.
    """
    if sentence == '':
        return []
    for i in reversed(range(1, len(sentence)+1)):
        firstword = sentence[:i]
        remainder = sentence[i:]
        if firstword in dictionary:
            return [firstword] + max_match(remainder, dictionary)

    # no word was found, so make a one-character word
    firstword = sentence[0]
    remainder = sentence[1:]
    return [firstword] + max_match(remainder, dictionary)

# ======================================
# TODO: statistical sequence models trained via supervised machine learning on hand-segmented training sets
# ======================================

if __name__ == "__main__":
    # =======================================
    # Test max_match
    # =======================================
    zh_words = ['他', '特别', '喜欢', '北', '北京', '北京烤鸭', '烤', '鸭']
    zh_sentence = '他特别喜欢北京烤鸭'
    print("Input:  {}".format(zh_sentence))
    print("Output: {}".format(max_match(zh_sentence, zh_words)))
    # output: ['他', '特别', '喜欢', '北京烤鸭']

    ## Short-coming: problem unknown words (words that not in the dictionary)
    test_sentence = '他喜欢中国的街边小吃'
    print("Input:  {}".format(test_sentence))
    print("Output: {}".format(max_match(test_sentence, zh_words)))
    # output: ['他', '喜欢', '中', '国', '的', '街', '边', '小', '吃']

    # max_match does not perform well on English sentence
    en_words = ['we', 'can', 'canon', 'only', 'on', 'see', 'ash', 'ort',
                'short', 'distance', 'stan', 'ahead', 'ah', 'head', 'ad']
    en_sentence = 'wecanonlyseeashortdistanceahead'
    print("Input:  {}".format(en_sentence))
    print("Output: {}".format(max_match(en_sentence, en_words)))
    # output: ['we', 'canon', 'l', 'y', 'see', 'ash', 'ort', 'distance', 'ahead']
    # =======================================
    # =======================================



