from ibm_watson import PersonalityInsightsV3
from ibm_watson import ApiException
import json

personality_insights = PersonalityInsightsV3(
    version='2017-10-13',
    iam_apikey='',
    url='https://gateway-fra.watsonplatform.net/personality-insights/api'
)


user_tweets = ""
first_status_line = ""
statuses_file = open("myPersonalitySmall/statuses_unicode.txt", "r")
first_status_line = statuses_file.readline()

print("first_status_line", first_status_line)
big5_scores_file = open("myPersonalitySmall/big5labels.txt")
first_big5_line = big5_scores_file.readline()
print("first_big5_line", first_big5_line)

progress_line = 0
new_user = 0


for now_big5 in big5_scores_file.readlines():
    print("progress_line", progress_line)
    progress_line = progress_line+1
    now_status = statuses_file.readline()
    if now_big5 == first_big5_line:
        #first_status_line = first_status_line.rstrip("\n")+" "+now_status
        user_tweets = user_tweets + " " + now_status
    else:
        print("new_user", new_user)        
        fo = open("./watson/_"+str(new_user)+"_watson_out.json","w")
        new_user = new_user + 1
        try:
            # Invoke a Personality Insights method
            profile = personality_insights.profile(user_tweets, 'application/json', content_language=None,
            accept_language=None, raw_scores=True, csv_headers=None,
            consumption_preferences=None, content_type=None).get_result()
            #print(json.dumps(profile, indent=2))
            fo.write(json.dumps(profile, indent=2))

        except ApiException as ex:
            print("Method failed with status code " + str(ex.code) + ": " + ex.message)
        fo.close()
                   
        first_status_line = now_status
        user_tweets = first_status_line
        first_big5_line = now_big5
        

#output_file.close()
statuses_file.close()
big5_scores_file.close()


fi.close()
fo.close()
