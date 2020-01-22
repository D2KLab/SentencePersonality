import numpy as np
import os
import sys
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
import spacy
from spacymoji import Emoji

#in the following tweets stand for Facebook post

line_done = 0
#multi_cased_L-12_H-768_A-12
#uncased_L-12_H-768_A-12
bert_model_dir = 'pretrained_bert/multi_cased_L-12_H-768_A-12'

check_empty = ""
output_file = open("train_whole_lines.csv", "a")

args = get_args_parser().parse_args(['-model_dir',bert_model_dir,
'-port', '5555', '-port_out', '5556',
'-max_seq_len','NONE', '-mask_cls_sep','-cpu','-num_worker=1', '-pooling_strategy', 'CLS_TOKEN'])


server = BertServer(args)
server.start()
bc = BertClient(ip = 'localhost')

#large vocabulary used in final solution
#nlp = spacy.load("en_core_web_lg")
#small vocabulary is used for testing purpose
print("spaCy en_core_web_sm loading...")
nlp = spacy.load("en_core_web_sm")
print("spaCy loaded")
# we use this library to translate image unicode of emoji ":)" into words "smiling face"
emoji = Emoji(nlp) 
nlp.add_pipe(emoji, first=True)

user_tweets = []

first_status_line = ""
statuses_file = open("myPersonalitySmall/statuses_unicode.txt", "r")
big5_scores_file = open("myPersonalitySmall/big5labels.txt")

progress_line = 0

tweet_aslist = []
lines_skipped = []
for now_big5 in big5_scores_file.readlines():
    if progress_line<line_done:
        progress_line = progress_line+1
        now_status = statuses_file.readline()
        continue
    
    progress_line = progress_line+1
    now_status = statuses_file.readline()
    user_tweets.append(now_status)
    for tweet in user_tweets:
        try:
            tweet_doc = nlp(tweet)
        except:
            print("cannot nlp spacy this post")
            user_tweets = []                
            tweet_aslist = []
            break
        cleaned_tweet = ""
        for token in tweet_doc:
            if(token.text.startswith("@")or "http" in token.text or "." in token.text or token.is_punct):
                continue
            if token._.is_emoji:
                emoji_description = token._.emoji_desc
                emoji_doc = nlp(emoji_description)
                for emoji_token in emoji_doc:
                    cleaned_tweet = cleaned_tweet+emoji_token.text+" "
                continue
            else:
                if token.text == "\n":
                    continue
                else:
                    cleaned_tweet = cleaned_tweet +token.text +" "
        if(len(cleaned_tweet)>0):
            tweet_aslist.append(str(cleaned_tweet.lower()))   

    if (len(tweet_aslist) == 0):
        lines_skipped.append(str(progress_line-1))
    else:
        print(tweet_aslist)
        check_empty = tweet_aslist[0].strip()
        print(check_empty)
        print(len(check_empty))
        if not check_empty:
            print("empty")
            print("check_empty", check_empty)
        else:
            print("encoding...")
            tweetEmbeddings = bc.encode(tweet_aslist)
            for elem in tweetEmbeddings[0]:
                output_file.write(str(elem)+",")
            big5_of_the_user = now_big5.rstrip("\n").split(" ")
            for elem in big5_of_the_user[:-1]:
                output_file.write(str(elem)+",")
            output_file.write(str(big5_of_the_user[-1])+"\n")
                
        user_tweets = []
        tweet_aslist = []

output_file.close()
statuses_file.close()
big5_scores_file.close()

myCmd = 'bert-serving-terminate -port 5555'
os.system(myCmd)
bc.close()
print("created train.csv file with one user per line and cls token strategy")
print("example line: 768d array + 5 scores")
f = open("lines_skipped.csv","w")
for elem in lines_skipped:
    f.write(elem+"\n")
f.close()
