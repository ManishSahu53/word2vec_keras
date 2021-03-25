import string
import re
from collections import Counter


def tokenize(text):
    text = preprocess_str(text)
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return text.split()


def preprocess_str(s):
    # if input string or number, convert to unicode
    s_lowered = s.lower()
    s = s_lowered.split()

    # "80x90" becomes "80 x 90"
    s = [re.sub('^([0-9]+)(x)([0-9]+$)', r'\g<1> x \g<3>', word)
         for word in s]
    s = (' '.join(s)).split()

    brackets = [['(', ')'], ['[', ']'], ['{', '}'], ['<', '>']]
    for b in brackets:
        # "(word) [word] {word} <word>" becomes "( word ) [ word ] { word } < word >"
        if any(e in s_lowered for e in b):
            s = [re.sub('(^\{0})(?=[a-z0-9])+|(?<=[a-z0-9])(\{1}$)'.format(b[0], b[1]), ' ' + r'\g<0>' + ' ', word)
                 for word in s]
            s = (' '.join(s)).split()

    # "charles&keith" becomes "charles & keith"
    if any(e in s_lowered for e in ['&', '/']):
        s = [re.sub('(?<=[a-z0-9])([\&\/])(?=[a-z0-9])', ' ' + r'\1' + ' ', word)
             for word in s]
        s = (' '.join(s)).split()

    # Other
    if any(e in s_lowered for e in list(string.punctuation)):
        s = [re.sub(  # pass('charles&keith', 's/d', 'ac//dc', 'literalstring') not('vp-link', 'game_boy')
            '(?<=[a-z])[^a-z0-9\&\/]+(?=[a-z])|' +
            # pass('s/d', 'ac/dc') not('s//d', 'ac///dc')
            '(?<=[a-z])[\&\/]{2,}(?=[a-z])|' +
            # not('90&june', '12_may')
            '(?<=[0-9])[^a-z0-9]+(?=[a-z])|' +
            # not('june&90', 'may_2017')
            '(?<=[a-z])[^a-z0-9]+(?=[0-9])|' +
            # pass('90.55', '90-55' '172.31.0.155') not('93?21', '11_11', '12|34')
            '(?<=[0-9])[^a-z0-9\.\,\:\-\/]+(?=[0-9])|' +
            # not('90..55', '90--55')
            '(?<=[0-9])[\.\,\:\-\/]{2,}(?=[0-9])|' +
            # pass('$$50', '$promo', '#55' ,'##promo') not('!promo', '!!!promo', '(promo', '((promo')
            '^[^a-z0-9\$\#]+(?=[a-z0-9])|' +
            # pass('#promo', '$12') not('$$12', '##50', '$$$100', '##promo', !@#$!promo)
            '^[^a-z0-9]{2,}(?=[a-z0-9])|' +
            # not('$promo', '$abc', '#promo')
            '^[\$\#](?=[a-z])|' +
            # pass('55%', '55%%') not('50?', 'apa?' 'baju.')
            '(?<=[a-z0-9])[^a-z0-9\%]+$|' +
            # pass('55%', '55%%', 'ab%%', 'c%') not('50?', 'apa?' 'baju.', 'ajfn@#$%^&')
            '(?<=[a-z0-9])[^a-z0-9]{2,}$|' +
            # not('ap%', 'bc%')
            '(?<=[a-z])[\%]$|' +
            # pass('.', '!', '/') not('...', '!!', '//')
            '^[^a-z0-9]{2,}$', ' ', word)
            for word in s]
        s = (' '.join(s)).split()

    s = ' '.join(s)

    return s


# strip accent (diacritic) in string
def strip_accents(s):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(n_words - 1))
    dictionary = dict()
    temp_count = 0
    
    for word, _ in count:
        dictionary[word] = temp_count
        temp_count +=1
    
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary