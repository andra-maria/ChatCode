import sklearn
from sklearn.feature_extraction import DictVectorizer
import file_processing
import word_processing

default_train_file = '../pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
default_test_file = '../pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'

default_train_conversations = '../pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-diff.txt'
default_test_conversations = '../pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem2.txt'

default_train_user_ids = '../pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
default_test_user_ids = '../pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem1.txt'

demo_test_conversations = '../demo/demo_test_corpus.xml'
demo_test_preds = '../demo/demo_test_pred.txt'

default_processing_function = file_processing.split_by_conversation
default_id_function = file_processing.split_ids

NUM_TOPICS = 5


def train(train_file, train_conversations, stage1_classifier, stage2_classifier):
    (train_data, binary_ids, train_ids) = file_processing.split_by_conversation(default_train_file, default_train_user_ids)

    (train_data, train_ids, binary_ids) = word_processing.entire_cleanup(train_data, train_ids, binary_ids)
    train_data = word_processing.to_dictionary(train_data)
    dicVec = DictVectorizer()
    train_data = dicVec.fit_transform(train_data)
    stage1_classifier.fit(train_data, binary_ids)
    print("fitted stage 1 classifier")

    (lda_data, lda_ids) = file_processing.split_by_user_id(default_train_file)
    predatory_ids = file_processing.split_ids(None, default_train_user_ids)
    binary_ids = list(map(lambda x: 1 if x in predatory_ids else 0, lda_ids))
    (binary_ids, lda_ids, lda_data) = word_processing.reduce_ids_for_demo(binary_ids, lda_ids, lda_data)
    (lda_data, lda_ids, binary_ids) = word_processing.entire_cleanup(lda_data, lda_ids, binary_ids)
    print("Predators in binary after cleanup " + str(sum(binary_ids)))
    print("Total users after cleanup " + str(len(binary_ids)))

    # Gensim Latent Dirichlet Allocation
    (ldamodel1, ldamodel2) = word_processing.double_lda(lda_data, binary_ids)

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

    stage2_classifier.fit(gens_data, binary_ids)
    print('fitted stage 2 classifier')

    return (ldamodel1, ldamodel2, dicVec)


def test(test_file, test_conversations, stage1_classifier, stage2_classifier, ldamodel1, ldamodel2, dictionary):
    (test_data, binary_ids, test_ids) = file_processing.split_by_conversation(demo_test_conversations, demo_test_preds)

    (test_data, test_ids, binary_ids) = word_processing.entire_cleanup(test_data, test_ids, binary_ids)
    test_data = word_processing.to_dictionary(test_data)
    test_data = dictionary.transform(test_data)
    pred_ids = stage1_classifier.predict(test_data)

    stage_1_results = pred_ids.copy()

    print("Statistics for STAGE 1------------------------------------------------------------------------------------")
    word_processing.print_metrics(binary_ids, pred_ids)
    pred_conversation_ids = []
    for i in range(len(stage_1_results)):
        if stage_1_results[i] is not 0:
            pred_conversation_ids.append(test_ids[i])

    (current_data, current_ids) = file_processing.split_by_user_pred_conv(test_file, pred_conversation_ids)
    binary_ids = []
    predator_ids = file_processing.split_ids(None, demo_test_preds)
    for i in range(len(current_ids)):
        if current_ids[i] in predator_ids:
            binary_ids.append(1)
        else:
            binary_ids.append(0)
    print("Number of predators in test data" + str (sum(binary_ids)))
    (current_data, predator_ids, binary_ids) = word_processing.entire_cleanup(current_data, current_ids, binary_ids)

    gens_data = word_processing.bag_of_words_double(current_data, ldamodel1, ldamodel2)
    pred_ids = stage2_classifier.predict(gens_data)
    print("Statistics for STAGE 2------------------------------------------------------------------------------------")
    word_processing.print_metrics(binary_ids, pred_ids)


def main(train_file=default_train_file, test_file=default_test_file,
         train_conversations=default_train_conversations, test_conversations=default_test_conversations,
         train_user_ids=default_train_user_ids, test_user_ids=default_test_user_ids):
    stage1_classifier = sklearn.svm.SVC(kernel='linear',  class_weight={0: 1, 1: 1})
    stage2_classifier = sklearn.svm.SVC(kernel='linear',  class_weight={0: 5, 1: 1})
    (ldamodel1, ldamodel2, dicVec) = train(train_file, train_user_ids, stage1_classifier, stage2_classifier)
    test(test_file, test_user_ids, stage1_classifier, stage2_classifier, ldamodel1, ldamodel2, dicVec)

if __name__ == '__main__':
    main()
