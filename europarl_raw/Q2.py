import random
random.seed(666)

with open('train.en', 'r', encoding='utf8') as f:
    train_en = f.readlines()

with open('test.en', 'r', encoding='utf8') as f:
    test_en = f.readlines()

with open('train.de', 'r', encoding='utf8') as f:
    train_de = f.readlines()

with open('test.de', 'r', encoding='utf8') as f:
    test_de = f.readlines()


def compute_data(data):
    data_dict = {}
    for item in data:
        word_list = item.replace('\n', '').split(' ')
        for word in word_list:
            try:
                data_dict[word] += 1
            except:
                data_dict[word] = 1
    return data_dict


def count_sent_length(data):
    length = 0
    num = len(data)
    for item in data:
        length += len(item.replace('\n', '').split(' '))
    return length/num


en_dict = compute_data(train_en)
de_dict = compute_data(train_de)

print('\n\n')
print('How many word tokens are in the English data? In the German data? Give both the total count and the number of word types in each language.\n')
print('total count of English training data: {}.  Number of types: {}'.format(
    sum(en_dict.values()), len(en_dict)))
print('total count of German training data: {}.  Number of types: {}'.format(
    sum(de_dict.values()), len(de_dict)))
print('\n\n')
print('How many word tokens will be replaced by <UNK> in English? In German? Sub- sequently, what will the total vocabulary size be?\n')
en_infre_words = [k for k, v in en_dict.items() if v == 1]
de_infre_words = [k for k, v in de_dict.items() if v == 1]
print('In English, {} words will be replaced by <UNK>. Total vocabulary size would be {} .'.format(len(en_infre_words),
                                                                                                   len(en_dict)-len(en_infre_words)+1))
print('In German, {} words will be replaced by <UNK>. Total vocabulary size would be {} .'.format(len(de_infre_words),
                                                                                                  len(de_dict)-len(de_infre_words)+1))

print('\n\n')
print('Is there a specific type of word which will be commonly replaced? Give an example of...')
print('English: ')
print(random.sample(en_infre_words, 10))
with open('infrenquen_English_word.txt', 'w', encoding="utf8") as f:
    for word in sorted(en_infre_words):
        f.write(word+'\n')
print('German: ')
print(random.sample(de_infre_words, 10))
print('\n\n')

with open('infrenquen_German_word.txt', 'w', encoding="utf8") as f:
    for word in sorted(de_infre_words):
        f.write(word+'\n')
print('How many vocabulary tokens are the same between both languages? How could we exploit this similarity in our model? \n')

en_wordset = set(en_dict.keys())
de_wordset = set(de_dict.keys())
print('{} words are the same between two language.'.format(
    len(en_wordset.intersection(de_wordset))))

l_en = count_sent_length(train_en)
l_de = count_sent_length(train_de)
print('average length of English sentences is {} .'.format(l_en))
print('average length of German sentences is {} .'.format(l_de))

# Check what effect will replacement have on test set
from functools import reduce


def count_words_in(line_enumerator, infreq_words):
    all_test_words = reduce(
        set.union, [set(line.replace('\n', '').split(' ')) for line in line_enumerator])
    infreq_words = set(infreq_words)
    infreq_in = {w for w in infreq_words if w in all_test_words}
    infreq_out = infreq_words-infreq_in
    return infreq_in, infreq_out


en_in, en_out = count_words_in(test_en, en_infre_words)
print(len(en_infre_words), len(en_in), len(en_out))

de_in, de_out = count_words_in(test_en, de_infre_words)
print(len(de_infre_words), len(de_in), len(de_out))
