# coding: utf-8

import random

import pandas as pd
import numpy as np
from numba import njit

dataset_url = "/home/zhang//devworks/ir.hit.edu.cn/dataset/massaged/imdb.train.txt.ss"


class Review:
    def __init__(self, user, product, score, comment):
        self.user = user
        self.product = product
        self.score = score
        self.comment = comment

    def get_score(self):
        return self.score

    def get_user(self):
        return self.user

    def get_product(self):
        return self.product

    def get_comment(self):
        return self.score


class ReviewRelative(object):
    all_names = {}

    def __init__(self, name):
        self.name = name
        self.reviews = []
        self.__class__.all_names[name] = self

    def add_review(self, review):
        self.reviews.append(review)

    def num_reviews(self):
        return len(self.reviews)

    def get_random_review(self, existing_review_id):
        rand_review_id = existing_review_id
        if (existing_review_id >= 0) and (self.num_reviews() == 1):
            raise RuntimeError(
                "{0} has too few reviews(#1) to selecting another randome review.".
                format(self.name))
            exit(-1)

        while rand_review_id == existing_review_id:
            rand_review_id = random.randrange(0, self.num_reviews())
            # print("existing id {0}, new random id: {1}".format(
            #     existing_review_id, rand_review_id))

        return (rand_review_id, self.reviews[rand_review_id])

    def get_name(self):
        return self.name


class User(ReviewRelative):
    def __init__(self, name):
        super().__init__(name)


class Product(ReviewRelative):
    def __init__(self, name):
        super().__init__(name)


imdb = pd.read_csv(
    dataset_url,
    sep='\t+')

# users = {}
# products = {}
# user_review_records = {}
# product_review_records = {}

for index, review_record in imdb.iterrows():
    user_name = review_record['user']
    product_name = review_record['product']

    review = Review(user_name, product_name, review_record['score'],
                    review_record['review'])

    user = None
    if user_name not in User.all_names:
        user = User(user_name)
    else:
        user = User.all_names[user_name]
    user.add_review(review)

    product = None
    if product_name not in Product.all_names:
        product = Product(product_name)
    else:
        product = Product.all_names[product_name]
    product.add_review(review)

# # print(User.all_names)
# s = [(k, d[k]) for k in sorted(User.all_names, key=User.all_names.get, reverse=True)]
# for (name, user) in s:
#     print("user name: {0}, num of reviews: {1}".format(name, user.num_reviews()))

print(">>> Done loading data. <<<")

list_users = User.all_names
list_products = Product.all_names
num_users = len(list_users)
num_products = len(list_products)

measure_same = []
measure_diff = []

random.seed()

num_iterations = 50

for iteration in range(0, num_iterations):
    current_measure_same = 0.0
    current_measure_diff = 0.0
    for user_name in User.all_names:
        user = User.all_names[user_name]
        num_user_reviews = user.num_reviews()

        if 1 == num_user_reviews:
            continue

        try:
            first_user_review_id0, first_user_review0 = user.get_random_review(
                -1)
            first_user_review_id1, first_user_review1 = user.get_random_review(
                first_user_review_id0)

            same_user_review_score_delta = abs(first_user_review1.get_score() -
                                               first_user_review0.get_score())
            current_measure_same += same_user_review_score_delta
            # measure_same.append(same_user_review_score_delta)

            # select another user
            second_user_name = user.get_name()
            second_user = user
            while second_user_name == user.get_name():
                second_user_name = random.choice(list(User.all_names.keys()))
                second_user = User.all_names[second_user_name]

            second_user_review_id, second_user_review = second_user.get_random_review(
                -1)
            diff_user_review_score_delta = abs(second_user_review.get_score() -
                                               first_user_review0.get_score())
            current_measure_diff += diff_user_review_score_delta
            # measure_diff.append(abs(diff_user_review_score_delta)

            print("first user: {0}, base review id: {1}, same user review id {2}, different user: {3} review id: {4}"
                   .format(first_user_review0.get_user(),
                           first_user_review_id0,
                           first_user_review_id1,
                           second_user_review.get_user(),
                           second_user_review_id))
        except IndexError as err:
            print(err)

    current_measure_diff /= len(list_users)
    measure_diff.append(current_measure_diff)

    current_measure_same /= len(list_users)
    measure_same.append(current_measure_same)

print("number of same's: {0}".format(len(measure_same)))
print("number of diff's: {0}".format(len(measure_diff)))

same_variance = np.var(measure_same)
diff_variance = np.var(measure_diff)

same_ev = sum(measure_same) / len(measure_same)
diff_ev = sum(measure_diff) / len(measure_diff)

print("similarity, EV: {0}, var: {1}".format(same_ev, same_variance))
print("difference, EV: {0}, var: {1}".format(diff_ev, diff_variance))
