from re import split

conversation_tag = '<conversation id'
text_tag = '<text>'
end_text_tag = '</text>'
author_tag = '<author>'

def split_by_conversation(filename):
    result = [];
    current_convo = [];
    conversation_ids = []
    current_id = None

    with open(filename, "r", encoding="utf8") as ins:
        for line in ins:

            if conversation_tag in line:
                if current_convo:
                    result.append(current_convo);
                    conversation_ids.append(current_id)
                    current_convo = [];
                    current_id = line.split('"')[1].split('"')[0]

            if text_tag in line:
                line = line.replace(text_tag, ' ');
                line = line.replace(end_text_tag, ' ');
                line = line.lower();

                current_convo.extend(filter(bool, (split(r'\W+', line))));

    if current_convo:
        result.append(current_convo);
        conversation_ids.append(current_id);

    return (result, conversation_ids);

def split_by_user_id(filename):
    result = {};

    current_id = None;
    with open(filename, "r", encoding="utf8") as ins:
        for line in ins:

            if author_tag in line:
                current_id = line.split('>')[1].split('<')[0];

            if text_tag in line:
                if current_id not in result:
                    result[current_id] = []

                line = line.replace(text_tag, ' ');
                line = line.replace(end_text_tag, ' ');
                line = line.lower();

                result[current_id].extend(filter(bool, (split(r'\W+', line))));

    return (list(result.values()), list(result.keys()))

def split_conversations(filename):
    result = [];

    with open(filename, "r", encoding="utf8") as ins:
        for line in ins:
            current_id = line.split()[0];
            if current_id not in result:
                result.append(current_id);

    return result;

def split_ids(filename):
    result = []

    with open(filename, "r", encoding="utf8") as ins:
        for line in ins:
            current_id = line.split('\n')[0]
            if current_id not in result:
                result.append(current_id)

    return result;