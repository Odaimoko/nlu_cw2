import random
random.seed(666)

with open('train.en','r') as f:
    train_en = f.readlines()

with open('train.de','r') as f:
    train_de = f.readlines()



def compute_data(data):
    data_dict = {}
    for item in data:
        word_list = item.replace('\n','').split(' ')
        for word in word_list:
            try:
                data_dict[word]+=1
            except:
                data_dict[word] = 1
    return data_dict

en_dict = compute_data(train_en)
de_dict = compute_data(train_de)

print('\n\n')
print('How many word tokens are in the English data? In the German data? Give both the total count and the number of word types in each language.\n')
print('total count of English training data: {}.  Number of types: {}'.format(sum(en_dict.values()),len(en_dict)))
print('total count of German training data: {}.  Number of types: {}'.format(sum(de_dict.values()),len(de_dict)))
print('\n\n')
print('How many word tokens will be replaced by <UNK> in English? In German? Sub- sequently, what will the total vocabulary size be?\n')
en_infre_words = [k for k,v in en_dict.items() if v==1]
de_infre_words = [k for k,v in de_dict.items() if v==1]
print('In English, {} words will be replaced by <UNK>. Total vocabulary size would be {} .'.format(len(en_infre_words),
                                                                                                   len(en_dict)-len(en_infre_words)+1))
print('In German, {} words will be replaced by <UNK>. Total vocabulary size would be {} .'.format(len(de_infre_words),
                                                                                                   len(de_dict)-len(de_infre_words)+1))

print('\n\n')
print('Is there a specific type of word which will be commonly replaced? Give an example of...')
print('English: ')
print(random.sample(en_infre_words,10))
# with open('infrenquen_English_word.txt','w') as f:
#     for word in en_infre_words:
#         f.write(word+'\n')
print('German: ')
print(random.sample(de_infre_words,10))
print('\n\n')
print('How many vocabulary tokens are the same between both languages? How could we exploit this similarity in our model? \n')

en_wordset = set(en_dict.keys())
de_wordset = set(de_dict.keys())
print('{} words are the same between two language.'.format(len(en_wordset.intersection(de_wordset))))