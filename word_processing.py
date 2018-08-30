from nltk import WordNetLemmatizer
from numpy import zeros

import file_processing
import gensim
import datetime
import sklearn
import re
import random


def filter_words(words):
    filtered_words = []
    stop_words = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.union(['amp', 'apos', 'http', 'quot', 'www', 'href', 'html', 'com'])

    for word_vector in words:
        word_vector = [word for word in word_vector if word not in stop_words]
        filtered_words.append(word_vector)

    wnl = WordNetLemmatizer()
    words = []
    for word_vector in filtered_words:
        word_vector = [wnl.lemmatize(word) for word in word_vector]
        words.append(word_vector)

    return words


def get_most_frequent_words(words):
    words_to_count = {}

    for word_vector in words:
        for word in word_vector:
            if word not in words_to_count:
                words_to_count[word] = 1
            else:
                words_to_count[word] += 1

    values = [val for val in words_to_count.values()]
    values.sort(reverse=True)
    threshold = values[1000]

    final_words = [word for word in words_to_count.keys() if words_to_count[word] >= threshold]

    return final_words


def rep_id(word, ids):
    if word in ids:
        return 'part_id'
    else:
        return word


def replace_participant_id(words, ids):
    res = []
    for word_vector in words:
        res.append(list(map(lambda x: rep_id(x, ids),  word_vector)))
    return res


def replace_participant_id_conversations(words, id_file):
    ids = file_processing.split_ids( None ,id_file)
    return replace_participant_id(words, ids)


def get_agetype(word):
    try:
        wordint = int(re.search(r'\d+', word).group())
        if 7 < wordint < 18:
            return 'age_minor'
        else:
            return 'age_adult'
    except AttributeError:
        return word


def replace_age(words):
    res = []
    for word_vector in words:
        res.append(list(map(lambda x: get_agetype(x), word_vector)))
    return res

def misc_clean_up(words):
    filtered_words = []

    for word_vector in words:
        word_vector = [re.sub(r'[^a-zA-Z]', '', word) for word in word_vector]
        filtered_words.append(word_vector)

    words = filtered_words;
    filtered_words = []

    for word_vector in words:
        word_vector = [word for word in word_vector if len(word) > 1 or word is 'm' or word is 'f'or word is 'u']
        filtered_words.append(word_vector)

    return filtered_words


def bag_of_words(words, words_to_count=None):
    if words_to_count is None:
        words_to_count = get_most_frequent_words(words)

    new_train_data = []
    for word_vector in words:
        current_train_data = [zeros(len(words_to_count))]
        for word in word_vector:
            if word in words_to_count:
                word_index = words_to_count.index(word)
                current_train_data[word_index] += 1
        new_train_data.append(current_train_data)

    return words_to_count, new_train_data


def bag_of_words_for_topics(topics, topics_no):
    new_topics = []
    for topic_vector in topics:
        current_topics = [0.0] * topics_no
        for (topic, prob) in topic_vector:
            current_topics[topic] = prob
        new_topics.append(current_topics.copy())

    return new_topics

def bag_of_words_double(data, ldamodel1, ldamodel2):
    dictionary1 = ldamodel1.id2word
    corpus1 = [dictionary1.doc2bow(text) for text in data]
    gens_data1 = []
    for doc in corpus1:
        current_topics = ldamodel1[doc]
        gens_data1.append(current_topics)

    dictionary2 = ldamodel2.id2word
    corpus2 = [dictionary2.doc2bow(text) for text in data]
    gens_data2 = []
    for doc in corpus2:
        current_topics = ldamodel2[doc]
        gens_data2.append(current_topics)

    all_topics = []

    for i in range(len(gens_data1)):
        topic_vector1 = ldamodel1[corpus1[i]]
        current_topics1 = [0.0] * ldamodel1.num_topics
        for (topic, prob) in topic_vector1:
            current_topics1[topic] = prob
        all_topics.append(current_topics1.copy())

        topic_vector2 = ldamodel2[corpus2[i]]
        current_topics2 = [0.0] * ldamodel2.num_topics
        for (topic, prob) in topic_vector2:
            current_topics2[topic] = prob
        all_topics[i].extend(current_topics2)

    return all_topics


def thin_ids(words, ids):
    length = len(words)
    i = 0
    while i < length:
        if len(words[i]) < 4:
            words.pop(i)
            ids.pop(i)
            i = i-1
            length = length - 1
        i = i+1
    return words, ids


def conversations_binary(conversation_ids, predatory_conversation_ids):
    result = []
    for i in range(len(conversation_ids)):
        if conversation_ids[i] in predatory_conversation_ids:
            result.append(1)
        else:
            result.append(0)
    return result


def lda_fewer_nonpred(num_nonpred, data, ids):
    current_data = []
    nonpred_prob = int((len(ids) - sum(ids)) / num_nonpred)

    for i in range(len(data)):
        data_vector = data[i]
        current_id = ids[i]
        if current_id is 1:
            current_data.append(data_vector)
        else:
            app = random.randrange(1, nonpred_prob)
            if app is 1:
                current_data.append(data_vector)

    dictionary = gensim.corpora.Dictionary(current_data)
    corpus = [dictionary.doc2bow(text) for text in current_data]

    print(datetime.datetime.now().hour, datetime.datetime.now().minute)

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=40, id2word=dictionary, passes=5)

    return ldamodel


def lda_train_sep(data, ids):
    pred_data = []
    other_data = []

    for i in range(len(data)):
        current_id = ids[i]
        data_vector = data[i]
        if current_id is 1:
            pred_data.append(data_vector)
        else:
            other_data.append(data_vector)

    dictionary = gensim.corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in pred_data]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=40, id2word=dictionary, passes=5)


    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)


    corpus = [dictionary.doc2bow(text) for text in other_data]
    ldamodel.update(corpus)

    return ldamodel

def double_lda(data, ids):
    pred_data = []
    other_data = []

    for i in range(len(data)):
        current_id = ids[i]
        data_vector = data[i]
        if current_id is 1:
            pred_data.append(data_vector)
        else:
            other_data.append(data_vector)

    dictionary = gensim.corpora.Dictionary(data)
    corpus1 = [dictionary.doc2bow(text) for text in pred_data]

    ldamodel1 = gensim.models.ldamodel.LdaModel(corpus1, num_topics=35, id2word=dictionary, passes=5)

    corpus2 = [dictionary.doc2bow(text) for text in other_data]
    ldamodel2 = gensim.models.ldamodel.LdaModel(corpus2, num_topics=35, id2word=dictionary, passes=5)

    return ldamodel1, ldamodel2
