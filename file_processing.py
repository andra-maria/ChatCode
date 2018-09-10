from re import split

conversation_tag = '<conversation id'
text_tag = '<text>'
end_text_tag = '</text>'
author_tag = '<author>'
enc = 'utf-8'
min_line_count = 20

def split_by_conversation(filename, pred_file):
    i = 0
    result = []
    current_convo = []
    conversation_ids = []
    binary_conversation_ids = []

    current_id = None
    line_count = 0

    pred_ids = split_ids(None, pred_file)
    is_predatory_conversation = 0

    with open(filename, "r", encoding=enc) as ins:
        for line in ins:

            if conversation_tag in line:
                if current_convo:
                    if line_count > min_line_count:
                        result.append(current_convo)
                        conversation_ids.append(current_id)
                        binary_conversation_ids.append(is_predatory_conversation)
                        i = i + 1

                    current_convo = []
                    current_id = line.split('"')[1].split('"')[0]
                    is_predatory_conversation = 0

            if author_tag in line:
                author_id = line.split('>')[1].split('<')[0]
                if author_id in pred_ids:
                    is_predatory_conversation = 1

            if text_tag in line:
                line = line.replace(text_tag, ' ')
                line = line.replace(end_text_tag, ' ')
                line = line.lower()
                line_count = line_count + 1
                current_convo.extend(filter(bool, (split(r'\W+', line))))

            if i is 100:
                break

    if current_convo and line_count > min_line_count:
        result.append(current_convo)
        conversation_ids.append(current_id)
        binary_conversation_ids.append(is_predatory_conversation)

    return result, binary_conversation_ids, conversation_ids

def split_by_user_id(filename):
    result = {}
    current_id = None
    i = 0
    with open(filename, "r", encoding=enc) as ins:
        for line in ins:

            if author_tag in line:
                current_id = line.split('>')[1].split('<')[0]

            if text_tag in line:
                if current_id not in result:
                    result[current_id] = []
                    i = i+1

                line = line.replace(text_tag, ' ')
                line = line.replace(end_text_tag, ' ')
                line = line.lower()

                result[current_id].extend(filter(bool, (split(r'\W+', line))))
            if i is 500:
                break;
    return list(result.values()), list(result.keys())


def split_by_user_pred_conv(filename, pred_conversations):
    result = {}
    current_id = None
    is_predatory_conversation = 0

    with open(filename, "r", encoding=enc) as ins:
        for line in ins:

            if conversation_tag in line:
                current_conversation_id = line.split('"')[1].split('"')[0]
                if current_conversation_id in pred_conversations:
                    is_predatory_conversation = 1
                else:
                    is_predatory_conversation = 0

            if author_tag in line:
                current_id = line.split('>')[1].split('<')[0]

            if text_tag in line:

                line = line.replace(text_tag, ' ')
                line = line.replace(end_text_tag, ' ')
                line = line.lower()
                if is_predatory_conversation is 1:
                    if current_id not in result:
                        result[current_id] = []
                    result[current_id].extend(filter(bool, (split(r'\W+', line))))

    return list(result.values()), list(result.keys())


def split_conversations(filename):
    result = []

    with open(filename, "r", encoding=enc) as ins:
        for line in ins:
            current_id = line.split()[0]
            if current_id not in result:
                result.append(current_id)

    return result


def split_ids(_, filename):
    result = []

    with open(filename, "r", encoding=enc) as ins:
        for line in ins:
            current_id = line.split('\n')[0]
            if current_id not in result:
                result.append(current_id)

    return result


def predatory_conversations(filename, pred_file):
    pred_ids = split_ids(None, pred_file)
    result = []
    is_pred = 0
    current_id = None

    with open(filename, "r", encoding=enc) as ins:
        for line in ins:

            if conversation_tag in line:
                if is_pred:
                    result.append(current_id)
                    is_pred = 0
                    current_id = line.split('"')[1].split('"')[0]

            if author_tag in line:
                current_author = line.split('>')[1].split('<')[0]
                if current_author in pred_ids:
                    is_pred = 1
    return result
