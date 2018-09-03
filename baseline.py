import sklearn

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


def train(train_file, train_conversations, stage1_classifier, stage2_classifier):
    (train_data, train_ids) = file_processing.split_by_conversation(train_file, default_train_user_ids)

    train_data = word_processing.filter_words(train_data)
    train_data = word_processing.replace_age(train_data)
    print('replaced age')

    (train_data, train_ids) = word_processing.thin_ids(train_data, train_ids)
    train_data = word_processing.misc_clean_up(train_data)
    print('filtered data')
    print('len ids ' + str(len(train_ids)))
    print(train_data[0])
    print('processed training ids')


    print('processed training ids')
    print('len train ids' + str(len(train_ids)))
    print(train_ids)
    print('number of predatory ids in train ids ' + str(sum(train_ids)))
    print (train_data[0])

    dict_train_data = []
    for data_line in train_data:
        dict_train_data.append(dict(Counter(data_line)))
    train_data = dict_train_data
    print(train_data[0])
    dicVec = DictVectorizer()
    train_data = dicVec.fit_transform(train_data)

    stage1_classifier.fit(train_data, train_ids)
    print("fitted stage 1 classifier")

    (lda_data, lda_ids) = file_processing.split_by_user_id(train_file)
    predatory_ids = file_processing.split_ids(None, default_train_user_ids)
    lda_ids = list(map(lambda x: 1 if x in predatory_ids else 0, lda_ids))
    print(lda_data[0])
    print(predatory_ids)
    print(lda_ids)
    print(sum(lda_ids))

    lda_data = word_processing.filter_words(lda_data)
    lda_data = word_processing.replace_age(lda_data)
    print('replaced age')

    (lda_data, lda_ids) = word_processing.thin_ids(lda_data, lda_ids)
    lda_data = word_processing.misc_clean_up(lda_data)

    # Gensim Latent Dirichlet Allocation
    (ldamodel1, ldamodel2) = word_processing.double_lda(lda_data, lda_ids)

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
    gens_data = word_processing.bag_of_words_double(lda_data, ldamodel1, ldamodel2)

    stage2_classifier.fit(gens_data, lda_ids)

    print('fitted stage 2 classifier')

    return (ldamodel1, ldamodel2)


def test(test_file, test_conversations, stage1_classifier, stage2_classifier, ldamodel1, ldamodel2):
    (test_data, test_ids) = file_processing.split_by_conversation(test_file, default_test_user_ids)

    test_data = word_processing.filter_words(test_data)
    test_data = word_processing.replace_age(test_data)

    print("filtered test data")

    (test_data, test_ids) = word_processing.thin_ids(test_data, test_ids)
    test_data = word_processing.misc_clean_up(test_data)

    print("thinned test ids")

    print(test_data[0])
    dict_test_data = []
    for data_line in test_data:
        dict_test_data.append(dict(Counter(data_line)))
    test_data = dict_test_data
    print(test_data[0])

    dicVec = DictVectorizer()
    train_data = dicVec.fit_transform(test_data)
    pred_ids = stage1_classifier.predict(train_data)
    print("Statistics for STAGE 1------------------------------------------------------------------------------------")
    word_processing.print_metrics(test_ids, pred_ids)

    (current_data, current_ids) = file_processing.split_by_user_id(test_file)
    lda_data = []
    lda_ids = []
    for i in range(len(pred_ids)):
        if pred_ids[i] is 1:
            lda_data.append(current_data[i])
            lda_ids.append(current_ids[i])
    print("Current ids")
    print (current_ids)

    print("\n\nLDA ids")
    print(lda_ids)

    gens_data = word_processing.bag_of_words_double(lda_data, ldamodel1, ldamodel2)
    pred_ids = stage2_classifier.predict(gens_data)
    print("Statistics for STAGE 2------------------------------------------------------------------------------------")
    word_processing.print_metrics(test_ids, pred_ids)


def main(train_file=default_train_file, test_file=default_test_file,
         train_conversations=default_train_conversations, test_conversations=default_test_conversations,
         train_user_ids=default_train_user_ids, test_user_ids=default_test_user_ids):
    stage1_classifier = sklearn.svm.SVC(kernel='linear',  class_weight={0: 1, 1: 5})
    stage2_classifier = sklearn.svm.SVC(kernel='linear',  class_weight={0: 1, 1: 5})
    (ldamodel1, ldamodel2) = train(train_file, train_user_ids, stage1_classifier, stage2_classifier)
    test(test_file, test_user_ids, stage1_classifier, stage2_classifier, ldamodel1, ldamodel2)

if __name__ == '__main__':
    main()
