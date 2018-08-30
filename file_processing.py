from re import split

conversation_tag = '<conversation id'
text_tag = '<text>'
end_text_tag = '</text>'
author_tag = '<author>'
enc = 'utf-8'


def split_by_conversation(filename):
    result = []
    current_convo = []
    conversation_ids = []
    current_id = None
    line_count = 0
    add_count = 0
    skip_count = 0

    with open(filename, "r", encoding=enc) as ins:
        for line in ins:

            if conversation_tag in line:
                if current_convo:
                    result.append(current_convo)
                    conversation_ids.append(current_id)
                    current_convo = []
                    current_id = line.split('"')[1].split('"')[0]

            if text_tag in line:
                line = line.replace(text_tag, ' ')
                line = line.replace(end_text_tag, ' ')
                line = line.lower()
                line_count = line_count + 1
                if add_count < 10:
                    current_convo.extend(filter(bool, (split(r'\W+', line))))
                    add_count = add_count + 1
                else:
                    skip_count = skip_count + 1
                    if skip_count is 5:
                        add_count = 0
                        skip_count = 0

    if current_convo and line_count > 15:
        result.append(current_convo)
        conversation_ids.append(current_id)

    return result, conversation_ids


def split_by_user_filter_short_conversations(filename):
    current_convo = {}
    result = {}
    line_count = 0
    current_id = None

    with open(filename, "r", encoding=enc) as ins:
        for line in ins:

            if conversation_tag in line:
                if current_convo and line_count > 20:
                    for current_id in current_convo:
                        if current_id not in result:
                            result[current_id] = []
                        result[current_id].extend(current_convo[current_id][:])

                current_convo = {}
                line_count = 0

            if author_tag in line:
                current_id = line.split('>')[1].split('<')[0]
                if current_id not in current_convo:
                    current_convo[current_id] = []

            if text_tag in line:
                line = line.replace(text_tag, ' ')
                line = line.replace(end_text_tag, ' ')
                line = line.lower()
                line_count = line_count + 1
                current_convo[current_id].extend(filter(bool, (split(r'\W+', line))))

    if current_convo and line_count > 20:
        if current_id not in result:
            result[current_id] = []
        result[current_id].extend(current_convo[current_id][:])

    return list(result.values()), list(result.keys())


def split_by_user_id(filename):
    result = {}

    current_id = None
    with open(filename, "r", encoding=enc) as ins:
        for line in ins:

            if author_tag in line:
                current_id = line.split('>')[1].split('<')[0]

            if text_tag in line:
                if current_id not in result:
                    result[current_id] = []

                line = line.replace(text_tag, ' ')
                line = line.replace(end_text_tag, ' ')
                line = line.lower()

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
