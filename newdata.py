import requests
import os
import json
import tweepy
import time
import datetime
from predict import Predict
from preprocessing import raw_dataset_path, save_preprocessed_data_path
from county_polygon import counties_bounding_boxes
# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAALJtVwEAAAAALk1nggzOKVlKVv2MRWx1qlq0VMM%3DYgSB0YDShF4dhzqdScp4SvSi63xt0ES8PAhEotY06R27bnkofh'

search_url = "https://api.twitter.com/2/tweets/search/all"
# search_url = "https://api.twitter.com/1.1/tweets/search/fullarchive/"

counties = ['Queens', 'Staten Island']
ever_14_day = [20210101,20210115,20210129,20210212,20210226]
# ever_14_day = [20210101,20210115,20210129]
ever_14_day = [str(i) for i in ever_14_day]
#set up the time: from now to 20200101 500 rquest every day every county

query = 'has:geo (libtard OR black OR white OR woman OR racist OR politics OR liberal OR allahsoil OR hate) lang:en place:'
# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
query_params = {
                'tweet.fields': 'created_at',
                'max_results' : '500',
                'expansions' : 'geo.place_id',
                'start_time' :'2020-12-31T00:00:00Z',
                'place.fields' : 'country,name'
                }


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r


def connect_to_endpoint(url, params):

    response = requests.request("GET", search_url, auth=bearer_oauth, params=params)
    # if response.status_code != 200:
    #     raise Exception(response.status_code, response.text)
    return response.json(), response.status_code


def main():
    predict = Predict('used')
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Starts")
    with open(os.path.join(save_preprocessed_data_path, "hate_num2.csv"), "a+") as fp:
        for county in counties:
            query1 = query + county
            query_params.update({'query':query1})
            res_str = []
            for period in ever_14_day:
                date = datetime.datetime(int(period[:4]),int(period[4:6]),int(period[6:8]),22)
                date_after_14 = date + datetime.timedelta(days=14)
                time_format = date.strftime('%Y-%m-%dT%H:%M:%SZ')
                query_params.update({'end_time' : time_format})
                accumulate_hate = 0
                acc_t = 0
                while date < date_after_14:
                    json_response, response_status_code = connect_to_endpoint(search_url, query_params)
                    if response_status_code != 200:
                        time.sleep(1)
                        continue
                    date = date + datetime.timedelta(days=1)
                    time_format = date.strftime('%Y-%m-%dT%H:%M:%SZ')
                    query_params.update({'end_time': time_format})
                    for data in json_response['data']:
                        acc_t += 1
                        if predict.predict(data['text']) :
                            accumulate_hate += 1
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} County: {county}, Date: {period}, "
                      f"Total tweets retrieved: {acc_t}, Detected hate speech tweets: {accumulate_hate}")
                res_str.append(str(accumulate_hate / acc_t))
            res_str = ",".join(res_str)
            fp.write(f"{res_str}\n")



if __name__ == "__main__":
    main()

