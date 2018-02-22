import sklearn

import file_processing
import word_processing

default_train_file = '../pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml';
default_test_file = '../pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml';

default_train_conversations = '../pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-diff.txt'
default_test_conversations = '../pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem2.txt'

default_train_user_ids = '../pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
default_test_user_ids = '../pan12-sexual-predator-identification-test-corpus-2012-05-21/pan12-sexual-predator-identification-groundtruth-problem1.txt'

default_processing_function = file_processing.split_by_user_id
default_id_function = file_processing.split_ids

def train(train_file, train_conversations, classifier):
    (train_data, ids) = default_processing_function(train_file);

    train_data = word_processing.filter_words(train_data);
    #train_data = word_processing.replace_participant_id(train_data, ids);
    print('replaced ids')
    train_data = word_processing.replace_age(train_data);
    print('replaced age')
    (train_data, ids) = word_processing.thin_ids(train_data, ids);
    print('filtered data')

    predatory_ids = default_id_function(train_conversations);
    train_ids = word_processing.conversations_binary(ids, predatory_ids);

    print('processed training ids')
    print(len(train_ids))
    print(len(predatory_ids))
    print(sum(train_ids))
    print (train_data[0])

    (most_used_words, train_data) = word_processing.bag_of_words(train_data);

    classifier.fit(train_data, train_ids);

    print('classified data')
    return most_used_words;


def test(test_file, test_conversations, classifier, most_used_words):
    (test_data, ids) = default_processing_function(test_file)

    test_data = word_processing.filter_words(test_data)
    #test_data = word_processing.replace_participant_id(test_data, ids)
    test_data = word_processing.replace_age(test_data)

    (test_data, ids) = word_processing.thin_ids(test_data, ids);
    print(test_data[0])

    predatory_ids = default_id_function(test_conversations);
    test_ids = word_processing.conversations_binary(ids, predatory_ids);
    (_, test_data) = word_processing.bag_of_words(test_data, most_used_words);

    pred_ids = classifier.predict(test_data);

    print(sklearn.metrics.accuracy_score(test_ids, pred_ids))
    print(len(test_ids))
    print(len(predatory_ids))

    false_positives = 0;
    false_negatives = 0;

    true_positives = 0;
    true_negatives = 0;

    for i in range(len(pred_ids)):
        if pred_ids[i] == test_ids[i]:
            if pred_ids[i] == 0:
                true_negatives += 1;
            else:
                true_positives += 1;
        else:
            if pred_ids[i] == 0:
                false_negatives += 1;
            else:
                false_positives += 1;

    print ('true positives / negatives: ' + str(true_positives) +  ' / ' + str(true_negatives))
    print ('false positives / negatives: ' + str (false_positives) + ' / ' + str(false_negatives))

def main(train_file = default_train_file, test_file = default_test_file, \
         train_conversations = default_train_conversations, test_conversations = default_test_conversations, \
         train_user_ids = default_train_user_ids, test_user_ids = default_test_user_ids):
    classifier = sklearn.svm.SVC(kernel= 'linear')
    most_used_words = train(train_file, train_user_ids, classifier);
    test(test_file, test_user_ids, classifier, most_used_words);

if __name__ == '__main__':
    main()
