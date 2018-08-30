import sklearn
import datetime
import gensim
from sklearn.feature_extraction.text import TfidfTransformer

import file_processing
import word_processing

default_train_file = '../pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
default_test_file = '../pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'

default_train_conversations = '../pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-diff.txt'
default_test_conversations = '../pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem2.txt'

default_train_user_ids = '../pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
default_test_user_ids = '../pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem1.txt'

default_processing_function = file_processing.split_by_conversation
default_id_function = file_processing.split_ids

NUM_TOPICS = 5


def train(train_file, train_conversations, classifier):
    (train_data, train_ids) = file_processing.split_by_conversation(train_file, default_train_user_ids)

    train_data = word_processing.filter_words(train_data)
    train_data = word_processing.replace_age(train_data)
    print('replaced age')
    (train_data, train_ids) = word_processing.thin_ids(train_data, train_ids)
    #train_data = word_processing.replace_participant_id_conversations(train_data, train_file)
    train_data = word_processing.misc_clean_up(train_data)
    print('filtered data')
    print('len ids ' + str(len(train_ids)))
 #   train_data = word_processing.replace_participant_id(train_data, ids);
    print('replaced ids')

#    predatory_ids = default_id_function(train_file, train_conversations)
#    train_ids = word_processing.conversations_binary(ids, predatory_ids)

    print('processed training ids')


    print('processed training ids')
    print('len train ids' + str(len(train_ids)))
    print(train_ids)
    print('number of predatory ids in train ids ' + str(sum(train_ids)))
    print (train_data[0])


    # Gensim Latent Dirichlet Allocation
    (ldamodel1, ldamodel2) = word_processing.double_lda(train_data, train_ids)

    print("lda pred")
    topics = ldamodel1.print_topics(num_words=10)
    pt = open("predator_topics_separate.txt", "w+")
    for topic in topics:
        pt.write(str(topic) + "\n")
    pt.close()

    print("lda nonpred")
    topics = ldamodel2.print_topics(num_words=10)
    npt = open("non_predator_topics_separate.txt", "w+")
    for topic in topics:
        npt.write(str(topic) + "\n")
    npt.close()

#    dictionary = ldamodel.id2word
#    corpus = [dictionary.doc2bow(text) for text in train_data]

#    print(datetime.datetime.now().hour, datetime.datetime.now().minute)

#    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=20)
#    ldamodel.save('modeltrain.gensim')

#    print("lda done")
#    print(datetime.datetime.now().hour, datetime.datetime.now().minute)

#    topics = ldamodel.print_topics(num_words=5)
#    for topic in topics:
#        print(topic)

#   (most_used_words, train_data) = word_processing.bag_of_words(train_data);
#    tfidf = TfidfTransformer();
#    tfidf.fit_transform(train_data);

#    gens_data = []
#    for doc in corpus:
#        current_topics = ldamodel[doc]
#        gens_data.append(current_topics)

#    gens_data = word_processing.bag_of_words_for_topics(gens_data, NUM_TOPICS)

    gens_data = word_processing.bag_of_words_double(train_data, ldamodel1, ldamodel2)

    classifier.fit(gens_data, train_ids)

    print('classified data')

    return (ldamodel1, ldamodel2)


def test(test_file, test_conversations, classifier, ldamodel1, ldamodel2):
    (test_data, test_ids) = file_processing.split_by_conversation(test_file, default_test_user_ids)

    test_data = word_processing.filter_words(test_data)
    test_data = word_processing.replace_age(test_data)

    print("filtered test data")

    (test_data, test_ids) = word_processing.thin_ids(test_data, test_ids)
    #test_data = word_processing.replace_participant_id_conversations(test_data, test_file)
    test_data = word_processing.misc_clean_up(test_data)

    print("thinned test ids")

    print(test_data[0])

  #  predatory_ids = default_id_function(test_file, test_conversations)
 #   test_ids = word_processing.conversations_binary(ids, predatory_ids)
#    (_, test_data) = word_processing.bag_of_words(test_data, most_used_words);

#    tfidf = TfidfTransformer();
#    tfidf.fit_transform(test_data);

#    dictionary = ldamodel.id2word
#    corpus = [dictionary.doc2bow(text) for text in test_data]

#    gens_data = []
#    for doc in corpus:
#        current_topics = ldamodel[doc]
#        gens_data.append(current_topics)

#    gens_data = word_processing.bag_of_words_for_topics(gens_data, NUM_TOPICS)
    gens_data = word_processing.bag_of_words_double(test_data, ldamodel1, ldamodel2)

    pred_ids = classifier.predict(gens_data)

    print(sklearn.metrics.accuracy_score(test_ids, pred_ids))
    print(len(test_ids))

    false_positives = 0
    false_negatives = 0

    true_positives = 0
    true_negatives = 0

    for i in range(len(pred_ids)):
        if pred_ids[i] == test_ids[i]:
            if pred_ids[i] == 0:
                true_negatives += 1
            else:
                true_positives += 1
        else:
            if pred_ids[i] == 0:
                false_negatives += 1
            else:
                false_positives += 1

    res = open('res.txt', 'w')
    res.write('true positives / negatives: ' + str(true_positives) + ' / ' + str(true_negatives))
    res.write('\nfalse positives / negatives: ' + str(false_positives) + ' / ' + str(false_negatives))
    res.close()

    print('true positives / negatives: ' + str(true_positives) + ' / ' + str(true_negatives))
    print('\nfalse positives / negatives: ' + str(false_positives) + ' / ' + str(false_negatives))


def main(train_file=default_train_file, test_file=default_test_file,
         train_conversations=default_train_conversations, test_conversations=default_test_conversations,
         train_user_ids=default_train_user_ids, test_user_ids=default_test_user_ids):
    classifier = sklearn.svm.SVC(kernel='linear',  class_weight={0: 1, 1: 20})
    (ldamodel1, ldamodel2) = train(train_file, train_user_ids, classifier)
    test(test_file, test_user_ids, classifier, ldamodel1, ldamodel2)

if __name__ == '__main__':
    main()
