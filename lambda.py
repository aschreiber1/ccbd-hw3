import numpy as np
import boto3
import json
import botocore
import sys
from hashlib import md5
import os
from email.parser import BytesParser
from email import policy
from datetime import date

maketrans = str.maketrans
vocabulary_length = 9013
endpoint = os.environ['ENDPOINT']

def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]
    
def getEmailText(event):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    print('Bucket: {}, Key: {}'.format(bucket, key))
   
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    email = obj['Body'].read()
    
    parsed = BytesParser(policy=policy.SMTP).parsebytes(email)
    plain = parsed.get_body(preferencelist=('plain'))
    plain = ''.join(plain.get_content().splitlines(keepends=True))
    
    print(plain)
    
    subject = parsed.get('Subject')
    print(subject)
    
    return { 'body' : plain, 'subject' : subject }
    
def sendEmail(subject, body, classificaiton, score):
    client = boto3.client('ses')
    today = date.today()
    formatted = today.strftime("%b-%d-%Y")
    
    body = body[:240] #they only want first 240 chars
    
    body = """ We received your email sent on {} with the subject: \"{}\". 
            
        Here is a 240 character sample of the email body: 
        {}
            
        The email was categorized as {} with a {} % confidence.""".format(formatted, subject, body, classificaiton, score*100)
    
    response = client.send_email(
        Destination={
            'ToAddresses': [
                'ajs2409@columbia.edu'
            ],
        },
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': body,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'SES/Lambda Based SPAM Predictor',
            },
        },
        Source='test@ccbd-hw3.xyz',
    )

def lambda_handler(event, context):
    print(event)
    
    email_info = getEmailText(event)
    
    test_messages = [email_info['body']]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

    config = botocore.config.Config(read_timeout=80)
    runtime= boto3.client('runtime.sagemaker', config=config)

    payload = json.dumps(encoded_test_messages.tolist())
    response = runtime.invoke_endpoint(EndpointName=endpoint,
                                    ContentType='application/json',
                                    Body=payload)
    parsed = response['Body'].read().decode()
    result = json.loads(parsed)
    print(result)

    probability = result['predicted_probability'][0][0]

    label = "SPAM"
    if result['predicted_label'][0][0] == 0.0:
        label = "HAM"
        probability = 1 - probability
    
    sendEmail(email_info['subject'], email_info['body'], label, probability)

   # print(event)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
