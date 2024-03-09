# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field

import numpy as np
import re

import random

from porter_stemmer import PorterStemmer


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'Chatty Botter'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        stemmer = PorterStemmer()
        self.sentiment_stemmed = {stemmer.stem(word): score for word, score in self.sentiment.items()}

        self.user_ratings = np.zeros_like(ratings[:,0])
        self.recommendation_indices = []

        self.affirmative_regex = re.compile(r"\b(yes|okay|sure|definitely|certainly|absolutely|positive|accept|approve|consent|yeah)\b", re.IGNORECASE)
        self.negation_regex = re.compile(r"\b(no|never|deny|reject|refuse|decline|negative|nay|nope)\b", re.IGNORECASE)

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hello! I am Chatty Botter your personal movie recommender bot. Tell me about a movie you like!"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. """ +\
        """You can help users find movies they like and provide information about movies, but not TV shows or any other topic.""" +\
        """Your name is Chatty Botter. You can help users find movies they like and provide information about movies. They do not necessarily have to only mention moves they like, they could also mention movies they dislike. Your response must satisfy the following criterion. """ +\
"""(1) Detect which movie the user is talking about. All movies will be mentioned in quotes. Then, detect if the sentiment is positive or negative. Respond such that you are affirming the sentiment and the movie. Make sure to explicitly mention the name of the movie each time if you know about the movie. If you don't, say that you couldn't find the movie in your database. Also, if the user has not put the movie in quotation marks, tell them to enclose movies in quotation marks."""+\
""" Also, acknowledge whether the user liked or disliked the movie. If it is unclear whether the user liked or disliked the movie, please ask for clarification. Here is an example """ +\
"""User : I enjoyed "The Notebook"."""+\
"""Chatty Botter :  Ok, you liked "The Notebook"! Tell me what you thought of another movie."""+\
"""At the end, make sure to ask the user to tell you about another movie if there have been less than 5 movies mentioned so far."""+\
"""(2) Keep a count of the unique names of movies mentioned by the user. Once the user has mentioned 5 unique movies, ask if they would like a reccomendation and wait for their response before giving one. Only if the user would like one, reccomend a movie based on the movies (and sentiments towards each movie) mentioned thus far."""+\
"""(3) If the user mentions anything other than a movie, stay focused on movies and remember that your role is that of a moviebot assistant. Here is an example: """+\
"""User: Can we talk about cars instead?"""+\
"""Chatty Botter: As a moviebot assistant my job is to help you with only your movie related needs!  Anything film related that you'd like to discuss?"""+\
"""Make sure to only talk about movies for as long as you are talking to the user. Even after you have made a recommendation, you cannot at any point talk about anything that is not directly related to movies."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################

        def first_part_of_response():
            titles = self.extract_titles(line)

            if titles == []:
                return self.no_movies_provided(), False
            
            if len(titles) > 1:
                return self.one_movie_at_a_time(), False
            
            title = titles[0]
            
            movie_indices = self.find_movies_by_title(title)

            if movie_indices == []:
                return self.not_found_in_db(title), False
            
            if len(movie_indices) > 1:
                return self.many_found_in_db(title, movie_indices), False
            
            if len(movie_indices) == 1:
                index = movie_indices[0]
                sentiment = self.extract_sentiment(line)

                self.user_ratings[index] = sentiment

                if sentiment < 0:
                    return self.negative_sentiment(title), True
                elif sentiment == 0:
                    return self.neutral_sentiment(title), False
                else:
                    return self.positive_sentiment(title), True

        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
        else:
            response = ""
            successful = False
            
            if np.count_nonzero(self.user_ratings) >= 5:
                if self.affirmative_regex.search(line):
                    response += self.new_recommendation()
                elif self.negation_regex.search(line):
                    response += self.no_more_recommendations()
            else:
                response, successful = first_part_of_response()

            if successful:
                if np.count_nonzero(self.user_ratings) >= 5:
                    self.recommendation_indices = self.recommend(self.user_ratings, self.ratings, 10)

                    response += " " + self.new_recommendation()
                else:
                    response += " " + self.ask_another_movie()

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    def no_movies_provided(self):
        options = [
            "I'm sorry, but you haven't provided any movie titles yet. Could you please tell me about a movie you've seen?",
            "I'm afraid I can't make a recommendation without knowing your movie preferences first. Please tell me about a movie you've watched.",
            "To get started, could you share your thoughts on a movie you've seen recently?"
        ]
        return random.choice(options)

    def one_movie_at_a_time(self):
        options = [
            "Thanks for sharing your thoughts on that movie. Could you tell me about another one you've seen?",
            "I appreciate the feedback. Let me know about another movie you've watched, and I'll continue refining my recommendations.",
            "Got it! To better understand your preferences, could you provide details on another movie you've seen?"
        ]
        return random.choice(options)

    def not_found_in_db(self, title):
        options = [
            f"I'm afraid I don't have any information about \"{title}\" in my database. Could you tell me about a different one you've seen?",
            f"It seems like \"{title}\" isn't in my database. Please provide details on another movie you've watched.",
            f"Unfortunately, I don't have access to \"{title}\". Could you share your thoughts on a different movie instead?"
        ]
        return random.choice(options)

    def many_found_in_db(self, title, indices):
        matched_titles = [self.titles[i][0] for i in indices]

        options = [
            f"I found multiple movies matching \"{title}\": {', '.join(matched_titles)}. Could you please also provide the release year to help me identify the specific movie?",
            f"There are a few movies with the title \"{title}\" in my database: {', '.join(matched_titles)}. To narrow it down, can you also include the release year?",
            f"It looks like there are a few movies called \"{title}\": {', '.join(matched_titles)}. Can you give me the title with the release year so I can find the right one?"
        ]
        return random.choice(options)
        
    def negative_sentiment(self, title):
        options = [
            f"Noted, you didn't like \"{title}\".",
            f"Got it, \"{title}\" was not a favorite of yours.",
            f"Understood, you had a negative opinion of \"{title}\"."
        ]
        return random.choice(options)

    def neutral_sentiment(self, title):
        options = [
            f"I couldn't quite tell if you liked \"{title}\" or not. Could you clarify your thoughts on the movie?",
            f"It's a bit unclear whether you enjoyed \"{title}\" or not. Could you provide some more details on your opinion?",
            f"I'm afraid I didn't fully understand your sentiment towards \"{title}\". Would you mind explaining your thoughts a bit more?"
        ]
        return random.choice(options)

    def positive_sentiment(self, title):
        options = [
            f"Noted, you liked \"{title}\".",
            f"Got it, \"{title}\" was a positive experience for you.",
            f"Understood, you had a favorable opinion of \"{title}\"."
        ]
        return random.choice(options)
    
    def new_recommendation(self):
        options = []

        if len(self.recommendation_indices) == 0:
            options = [
                "That's all the recommendations I have based on your preferences. Your movie preferences have been reset, so feel free to provide new movies you've seen.",
                "I've exhausted my list of recommendations for you. Your movie preferences have been cleared, so we can start fresh with new movies you've watched.",
                "I don't have any more recommendations based on the movies you've mentioned. I've cleared your preferences, so please share some new movies you've seen."
            ]
            self.user_ratings = np.zeros_like(self.user_ratings)
        else:
            recommended_title = self.titles[self.recommendation_indices[0]][0]
            self.recommendation_indices = self.recommendation_indices[1:]
            options = [
                f"Based on your preferences, I recommend you watch \"{recommended_title}\". Would you like another recommendation?",
                f"My next recommendation for you is \"{recommended_title}\". Let me know if you'd like me to suggest another movie.",
                f"You might enjoy \"{recommended_title}\" based on the movies you've mentioned. Shall I provide another recommendation?"
            ]

        return random.choice(options)
    
    def no_more_recommendations(self):
        options = [
            "Okay, no problem. Your movie preferences have been cleared.",
            "Got it. I've reset your movie preferences, so feel free to provide new movies you've seen.",
            "Understood. I've cleared your movie preferences, so we can start fresh whenever you're ready to share more movies."
        ]
        self.user_ratings = np.zeros_like(self.user_ratings)
        return random.choice(options)

    def ask_another_movie(self):
        options = [
            "Tell me about another movie you've seen.",
            "Could you share your thoughts on another movie?",
            "To continue improving my recommendations, please tell me about another movie you've watched."
        ]
        return random.choice(options)


    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text
    
    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        return []

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        pattern = r'"([^"]+)"'
        matches = re.findall(pattern, preprocessed_input)
        return matches

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        def is_same_movie(db_title, input_title) -> bool:
            end_article_regex = re.compile(r',\s(the|an|a)', re.IGNORECASE)
            match = end_article_regex.search(db_title)
            
            if match:
                db_title = end_article_regex.sub('', db_title)
                db_title = f"{match.group(1)} {db_title}"
            
            input_title = input_title.strip()

            input_year_regex = r'\(\d+\)$'
            input_year_match = re.search(input_year_regex, input_title)

            db_year_regex = rf'{re.escape(input_year_match.group())}' if input_year_match else r'\(\d+\)'

            input_title_movie = input_title[:input_year_match.start()].strip() if input_year_match else input_title

            whole_movie_regex = rf'^{input_title_movie}\s*{db_year_regex}$'
            return re.match(whole_movie_regex, db_title) is not None
        
        return [i for i in range(len(self.titles)) if is_same_movie(self.titles[i][0], title)]

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        movie_regex = r'"([^"]+)"'
        preprocessed_input = re.sub(movie_regex, '', preprocessed_input)

        punctuation_regex = r'[^\w\s\d\']'  
        preprocessed_input = re.sub(punctuation_regex, '', preprocessed_input)

        words = preprocessed_input.split()

        sentiment_score = 0

        negation_mode = 1
        negation_words = [
            'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing', 'nowhere', 'nobody',
            'hardly', 'scarcely', 'barely', 'seldom', 'rarely', 'little', 'few', 'lack',
        ]
        negation_regex = r'n\'t'

        stemmer = PorterStemmer()
        for word in words:
            if word in negation_words or re.search(negation_regex, word, re.IGNORECASE):
                negation_mode *= -1
            
            stemmed = stemmer.stem(word)
            if stemmed in self.sentiment_stemmed:
                word_sentiment = self.sentiment_stemmed[stemmed]

                if word_sentiment == 'pos':
                    sentiment_score += 1 * negation_mode
                if word_sentiment == 'neg':
                    sentiment_score -= 1 * negation_mode
        
        if sentiment_score > 0:
            return 1
        elif sentiment_score < 0:
            return -1
        else:
            return 0

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)

        binarized_ratings[ratings > threshold] = 1
        binarized_ratings[ratings <= threshold] = -1
        binarized_ratings[ratings == 0] = 0

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            similarity = 0
        else:
            similarity = dot_product / (norm_u * norm_v)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        def cosine_similarity_matrix(matrix, watched_indices, not_watched_indices):
            m, _ = matrix.shape
            similarity_matrix = np.zeros((m, m))
            
            for i in watched_indices:
                for j in not_watched_indices:
                    u = matrix[i]
                    v = matrix[j]
                    similarity_score = self.similarity(u, v)
                    similarity_matrix[i, j] = similarity_score
                    similarity_matrix[j, i] = similarity_score
            
            return similarity_matrix
        
        not_watched_indices = np.where(user_ratings == 0)[0]
        watched_indices = np.where(user_ratings != 0)[0]

        similarity_matrix = cosine_similarity_matrix(ratings_matrix, watched_indices, not_watched_indices)

        scores = np.zeros(ratings_matrix.shape[0])

        for i in not_watched_indices:
            scores[i] = np.sum([similarity_matrix[i, j] * user_ratings[j] for j in watched_indices])

        recommendations = np.argsort(-scores)[:k]

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
