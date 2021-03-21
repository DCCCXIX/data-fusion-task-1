
import pickle
import random
import re
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


#preproccesses the dataframe, generates features
def preproccess(dataset):
     replace_dict = {
          "w" : "ш",
          "e" : "е",
          "y" : "у",
          "u" : "и",
          "o" : "о",
          "p" : "р",
          "a" : "а",
          "h" : "н",
          "k" : "к",
          "x" : "х",
          "c" : "с",
          "b" : "в",
          "m" : "м",
          "r" : "г",
          "a" : "a",
          "a" : "a",
     }
     #replaces latin letters in cyrillic words
     def replace_latin(text):
          if len(text) > 0:
               text = text.split()
               words = []
               for word in text:
                    if re.match(r"^[a-z]+$", word):
                         pass
                    else:
                         for lat, ru in replace_dict.items():
                              word = word.replace(lat, ru)
                    words.append(word)

               return " ".join(words)
          else:
               return "none"
     #separates numbers and punctuation from words
     def pretokenize_test(text):
          text = text.replace("-", "")
          text = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", text).strip()
          text = "".join(c if c not in string.punctuation else f" {c} " for c in text )
          text = "".join(c if c not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] else f" {c} " for c in text)
          #text = re.sub(r'[0-9]', ' 1 ', text)
          text =  " ".join(w.strip() for w in text.split())

          return text
     #removes all numbers and punctuation
     def clean_text(text):
          text = re.sub(r'[?|!"|#|.|%|,|$|~|^|`|;|@|╣|)|(|\\|\||-|[|\]|{|}|:|<|>|\'|+|&|=|°|/|№|\-|*|_]', r" ", text)
          text = re.sub(r'[0-9]', '', text)
          return text
     #removes all words shorter than 2 letters
     def keep_long_words(text):
          words = []
          for word in text.split():
               if len(word) > 2:
                    words.append(word)
          if len(words) == 0:
               return "none"
          return " ".join(words)

     #split time into hours and minutes
     time_list = [x.split(":") for x in dataset['receipt_time'].values]
     dataset['receipt_time_hours'] = [row[0] for row in time_list]
     dataset['receipt_time_minutes'] = [row[1] for row in time_list]
     #drop unneccessary "receipt time" column
     dataset.drop(["receipt_time"], inplace = True, axis = 1)

     #creating features
     #order of execution is important
     dataset["string_length"] = dataset["item_name"].apply(lambda x: len(x))
     dataset["punctuation_count"] = dataset["item_name"].apply(lambda comment: sum(comment.count(w) for w in '.,;:-/\\'))
     dataset["uppercase_amount"] = dataset["item_name"].apply(lambda comment: sum(1 for c in comment if c.isupper()))

     dataset["item_name"] = dataset["item_name"].apply(lambda x: x.lower())
     dataset["digit_amount"] = dataset["item_name"].apply(lambda x: sum(c.isdigit() for c in x))

     dataset["item_name"] = dataset["item_name"].apply(lambda x: pretokenize_test(x))

     dataset["latin_amount"] = dataset["item_name"].apply(lambda x: len(re.findall(r"[a-z]", x)))
     dataset["cyrillic_amount"] = dataset["item_name"].apply(lambda x: len(re.findall(r"[а-я]", x)))

     dataset["item_name"] = dataset["item_name"].apply(lambda x: x.replace("ё", "е"))
     dataset["item_name"] = dataset["item_name"].apply(lambda x: x.replace("й", "и"))
     dataset["item_name"] = dataset["item_name"].apply(lambda x: x.replace("ъ", "ь"))

     dataset["item_name"] = dataset["item_name"].apply(lambda x: x.replace("tpk", "трк"))
     dataset["item_name"] = dataset["item_name"].apply(lambda x: x.replace("трк", "топливо"))
     dataset["item_name"] = dataset["item_name"].apply(lambda x: x.replace(" аи ", " бензин "))
     dataset["item_name"] = dataset["item_name"].apply(lambda x: x.replace("раф", "кофе"))

     dataset["item_name"] = dataset["item_name"].apply(lambda x: replace_latin(x))

     dataset["temp_col"] = dataset["item_name"].apply(lambda x: clean_text(x))

     dataset["has_kcal"] = dataset["temp_col"].apply(lambda x: 1 if "ккал" in x.split() else 0)
     dataset["has_kg"] = dataset["temp_col"].apply(lambda x: 1 if "кг" in x.split() else 0)
     dataset["has_g"] = dataset["temp_col"].apply(lambda x: 1 if "г" in x.split() else 0)
     dataset["has_ml"] = dataset["temp_col"].apply(lambda x: 1 if "мл" in x.split() else 0)
     dataset["has_l"] = dataset["temp_col"].apply(lambda x: 1 if "л" in x.split() else 0)
     dataset["has_sht"] = dataset["temp_col"].apply(lambda x: 1 if "шт" in x.split() else 0)
     dataset["has_tab"] = dataset["temp_col"].apply(lambda x: 1 if "таб" in x.split() else 0)
     dataset["has_sb"] = dataset["temp_col"].apply(lambda x: 1 if "сб" in x.split() else 0)

     dataset["has_fl"] = dataset["temp_col"].apply(lambda x: 1 if "фл" in x.split() else 0)
     dataset["has_up"] = dataset["temp_col"].apply(lambda x: 1 if "уп" in x.split() else 0)
     dataset["has_cm"] = dataset["temp_col"].apply(lambda x: 1 if "см" in x.split() else 0)
     dataset["has_m"] = dataset["temp_col"].apply(lambda x: 1 if "м" in x.split() else 0)
     dataset["has_mm"] = dataset["temp_col"].apply(lambda x: 1 if "мм" in x.split() else 0)

     dataset["temp_col"] = dataset["temp_col"].apply(lambda x: keep_long_words(x))

     dataset["first_word"] = dataset["temp_col"].apply(lambda x: x.split()[0] if len(x.split()) > 0 else "none")
     dataset["last_word"] = dataset["temp_col"].apply(lambda x: x.split()[-1] if len(x.split()) > 0 else "none")

     dataset["second_first_word"] = dataset["temp_col"].apply(lambda x: x.split()[1] if len(x.split()) > 1 else "none")
     dataset["second_last_word"] = dataset["temp_col"].apply(lambda x: x.split()[-2] if len(x.split()) > 1 else "none")

     dataset.drop("temp_col", axis = 1, inplace = True)

     return dataset

#lower dtypes to save memory
def lower_dtypes(dataset):

    dtype_dict = {
    "has_kcal" : np.int8,
    "has_kg" : np.int8,
    "has_g" : np.int8,
    "has_ml" : np.int8,
    "has_l" : np.int8,
    "has_sht" : np.int8,
    "has_tab" : np.int8,
    "has_sb" : np.int8,
    "has_fl" : np.int8,
    "has_up" : np.int8,
    "has_cm" : np.int8,
    "has_m" : np.int8,
    "has_mm" : np.int8,
    "receipt_id" : np.int32,
    "id" : np.int32,
    "receipt_dayofweek" : np.int8,
    "one_char_amount" : np.int8,
    "two_char_amount" : np.int8,
    "item_name" : str,
    "item_name_modified" : str,
    "item_quantity" : np.int8,
    "item_price" : np.int8,
    "item_nds_rate" : np.int8,
    "category_id" : np.int8,
    "receipt_time_hours" : np.int8,
    "receipt_time_minutes" : np.int8,
    "string_length" : np.float16,
    "punctuation_count" : np.int8,
    "uppercase_amount" : np.int8,
    "FULLCAPS_COUNT" : np.float16,
    "digit_amount" : np.int8,
    "relative_digit_amount" : np.float16,
    "latin_amount" : np.int8,
    "cyrillic_amount" : np.int8,
    "relative_latin_amount" : np.float16,
    "relative_cyrillic_amount" : np.float16,
    "first_word" : str,
    "last_word" : str,
    "second_first_word" : str,
    "second_last_word" : str,
    "item_name_list" : str,
    "two_letter_features" : str,
    "one_letter_features" : str,
    "brands" : str
    }

    for column in dataset.columns:
        dataset[column] = dataset[column].astype(dtype_dict[column])

    return dataset

#splitting features into categorical, continuous and text
cont_features = [
                "string_length",
                "punctuation_count",
                "uppercase_amount",
                "digit_amount",
                "latin_amount",
                "cyrillic_amount",
                "item_quantity",
                "item_price"
                ]

cat_features = [
                "has_kcal",
                "has_kg",
                "has_g",
                "has_ml",
                "has_l",
                "has_sht",
                "has_tab",
                "has_sb",
                "has_fl",
                "has_up",
                "has_cm",
                "has_m",
                "has_mm",
                "item_nds_rate",
                "receipt_time_hours",
                "receipt_time_minutes",
                "receipt_dayofweek",
                #"receipt_id"
                ]

text_features = ["item_name", "first_word", "last_word", "second_first_word", "second_last_word"]
target = ["category_id"]
features = text_features + cat_features + cont_features

#custom text proccessing
#text processing differs for item_name and other text columns
text_processing = {
        "tokenizers" : [
          {
            "tokenizer_id" : "Space",
            "separator_type" : "ByDelimiter",
            "delimiter" : " ",
            "number_process_policy" : "Replace",
            "number_token" : "@",
          },
        ],

        "dictionaries" : [
            {
              "dictionary_id" : "Unigram",
              "token_level_type": "Letter",
              #"max_dictionary_size" : "500",
              "occurrence_lower_bound" : "1",
              "gram_order" : "1"
            },
            {
            "dictionary_id" : "Bigram",
            "token_level_type": "Letter",
            #"max_dictionary_size" : "500",
            "occurrence_lower_bound" : "1",
            "gram_order" : "2"
            },
            {
            "dictionary_id" : "Trigram",
            #"max_dictionary_size" : "500",
            "token_level_type": "Letter",
            "occurrence_lower_bound" : "1",
            "gram_order" : "3"
            },
            {
            "dictionary_id" : "Fourgram",
            #"max_dictionary_size" : "500",
            "token_level_type": "Letter",
            "occurrence_lower_bound" : "1",
            "gram_order" : "4"
            },
            {
            "dictionary_id" : "Fivegram",
            #"max_dictionary_size" : "500",
            "token_level_type": "Letter",
            "occurrence_lower_bound" : "1",
            "gram_order" : "5"
            },
            {
            "dictionary_id" : "Sixgram",
            #"max_dictionary_size" : "500",
            "token_level_type": "Letter",
            "occurrence_lower_bound" : "1",
            "gram_order" : "6"
            },
        ],

        "feature_processing" : {
            "item_name" : [
                  {
                    "dictionaries_names" : [
                      #"Unigram",
                      #"Bigram",
                      "Trigram",
                      "Fourgram",
                      "Fivegram",
                      #"Sixgram"
                      ],
                    "feature_calcers" : ["BoW"],
                    "tokenizers_names" : ["Space"]
                  },
                  {
                    "dictionaries_names" : [
                      #"Unigram",
                      #"Bigram",
                      "Trigram",
                      "Fourgram",
                      "Fivegram",
                      #"Sixgram"
                      ],
                    "feature_calcers" : ["NaiveBayes"],
                    "tokenizers_names" : ["Space"]
                  },
                  {
                    "dictionaries_names" : [
                      #"Unigram",
                      #"Bigram",
                      "Trigram",
                      "Fourgram",
                      "Fivegram",
                      #"Sixgram"
                      ],
                    "feature_calcers" : ["BM25"],
                    "tokenizers_names" : ["Space"]
                  },
            ],

            "default" : [
                  {
                    "dictionaries_names" : [
                      #"Unigram",
                      "Bigram",
                      "Trigram",
                      "Fourgram",
                      "Fivegram",
                      #"Sixgram"
                      ],
                    "feature_calcers" : ["BoW"],
                    "tokenizers_names" : ["Space"]
                  },
                  {
                    "dictionaries_names" : [
                      #"Unigram",
                      "Bigram",
                      "Trigram",
                      "Fourgram",
                      "Fivegram",
                      #"Sixgram"
                      ],
                    "feature_calcers" : ["NaiveBayes"],
                    "tokenizers_names" : ["Space"]
                  },
                  {
                    "dictionaries_names" : [
                      #"Unigram",
                      "Bigram",
                      "Trigram",
                      "Fourgram",
                      "Fivegram",
                      #"Sixgram"
                      ],
                    "feature_calcers" : ["BM25"],
                    "tokenizers_names" : ["Space"]
                  },
            ],
        }
    }

def main():
    dataset = pd.read_parquet('data/task1_test_for_user.parquet')

    with open("model.clf", 'rb') as f:
        model = pickle.loads(f.read())

    with open("encoder.pkl", "rb") as f:
        labelecoder = pickle.loads(f.read())

    dataset = preproccess(dataset)
    dataset = lower_dtypes(dataset)

    predictions = model.predict(dataset[features])
    predictions = labelecoder.inverse_transform(predictions)
    result = pd.DataFrame(predictions, columns=['pred'])
    result['id'] = dataset['id'].values
    result[['id', 'pred']].to_csv('answers.csv', index=None)

if __name__ == "__main__":
    main()
