from nltk import WordNetLemmatizer
from numpy import zeros
import sklearn
import re

def filter_words(words):
    filtered_words = []
    stop_words = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.union(['amp', 'apos']);
    for word_vector in words:
        word_vector = [word for word in word_vector if word not in stop_words];
        filtered_words.append(word_vector)

    wnl = WordNetLemmatizer();
    words = []
    for word_vector in filtered_words:
        word_vector = [wnl.lemmatize(word) for word in word_vector];
        words.append(word_vector)


    return words

def get_most_frequent_words(words):
    words_to_count = {}

    for word_vector in words:
        for word in word_vector:
            if word not in words_to_count:
                words_to_count[word] = 1;
            else:
                words_to_count[word] += 1;

    values = [val for val in words_to_count.values()];
    values.sort(reverse=True)
    threshold =  values[1000];

    final_words = [word for word in words_to_count.keys() if words_to_count[word] >= threshold];

    return final_words;


def rep_id(word, ids):
    if word in ids:
        return 'part_id';
    else:
        return word;

def replace_participant_id(words, ids):
    res = []
    for word_vector in words:
        res.append(list(map(lambda x: rep_id(x, ids),  word_vector)))
    return res

def get_agetype(word):
    try:
        wordint = int(re.search(r'\d+', word).group());
        if wordint < 18 and wordint > 7:
            return 'age_minor'
        else:
            return 'age_adult'
    except AttributeError:
        return word;

def replace_age(words):
    res = []
    for word_vector in words:
        res.append(list(map(lambda x: get_agetype(x), word_vector)))
    return res


def bag_of_words(words, words_to_count = None):
    if words_to_count == None:
        words_to_count = get_most_frequent_words(words)

    new_train_data = []
    for word_vector in words:
        current_train_data = zeros(len(words_to_count))
        for word in word_vector:
            if word in words_to_count:
                word_index = words_to_count.index(word)
                current_train_data[word_index] += 1;
        new_train_data.append(current_train_data)

    return (words_to_count, new_train_data);

def thin_ids(words, ids):
    length = len(words)
    i=0
    while i<length:
        if len(words[i]) < 4:
            words.pop(i);
            ids.pop(i);
            i = i-1
            length = length - 1
        i = i+1
    return (words, ids);


def conversations_binary(conversation_ids, predatory_conversation_ids):
    result =[]
    for i in range(len(conversation_ids)):
        if conversation_ids[i] in predatory_conversation_ids:
            result.append(1);
        else:
            result.append(0);
    return result;