# encoding: utf-8

import pandas as pd
import random
import numpy as np

imdb = pd.read_csv("/home/zhang//devworks/ir.hit.edu.cn/dataset/massaged/imdb.train.txt.ss", sep='\t+')

users = set()
products = set()
user_review_records = {}
product_review_records = {}

for index, review_record in imdb.iterrows():
    user = review_record['user']
    product = review_record['product']
    users.add(user)
    products.add(product)

    if not user in user_review_records or (user_review_records[user] == None):
        user_review_records[user] = []

    user_review_records[user].append(review_record)

    if not product in product_review_records or (product_review_records[product] == None):
        product_review_records[product] = []
        
    product_review_records[product].append(review_record)

users = list(users)
products = list(products)
num_users = len(users)
num_products = len(products)

measure_same = []
measure_diff = []

random.seed()

num_iterations = 50

for iteration in range(0, num_iterations):
    current_measure_same = 0.0
    current_measure_diff = 0.0
    for user in users:
        num_user_reviews = len(user_review_records[user])

        try:
            rand_review_id = random.randrange(0, num_user_reviews)
            user_score = user_review_records[user][rand_review_id]['score']
            
            rand_review_id1 = random.randrange(0, num_user_reviews)
            same_user_score = user_review_records[user][rand_review_id1]['score']

            current_measure_same += abs(same_user_score - user_score)
            measure_same.append(abs(same_user_score - user_score))

            diff_user = user
            while user == diff_user:
                diff_user_index = random.randrange(0, num_users)
                diff_user = users[diff_user_index]

            diff_user_review_id = random.randrange(0, len(user_review_records[diff_user]))
            diff_user_score = user_review_records[diff_user][diff_user_review_id]['score']

            current_measure_diff += abs(diff_user_score - user_score)
            measure_diff.append(abs(diff_user_score - user_score))
        except IndexError as err:
            print(err)

    current_measure_diff /= len(users)
    # measure_diff.append(current_measure_diff)

    current_measure_same /= len(users)
    # measure_same.append(current_measure_same)

same_variance = np.var(measure_same)
diff_variance = np.var(measure_diff)

same_ev = sum(measure_same) / len(measure_same)
diff_ev = sum(measure_diff) / len(measure_diff)

print("similarity, EV: {0}, var: {1}".format(same_ev, same_variance))
print("difference, EV: {0}, var: {1}".format(diff_ev, diff_variance))