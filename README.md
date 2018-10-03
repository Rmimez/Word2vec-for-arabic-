Arabic is a rich language, but there have not been many opportunities to use it in the field of scientific research. For this we suggested a way to implement the #Word2Vec algorithm to improve Arabic #Questions_Answering_System, it was tested on a set of questions written in Arabic.
#Word2Vec created by Google in 2013, #Word2Vec use as an entry a large or small text and then produce a vector space usually of several hundred dimensions.
#Word2vec has been used in several domains, from these domains #Questions_Answering_System with different language. But we noted that these works are not very applied to our mother language "Arabic language".
We suggested a way for implement #Word2Vec algorithm to improve Arabic #Questions_Answering_System, it was tested on a set of questions written in Arabic.
We suggested a way for implement #Word2Vec algorithm to improve Arabic #Questions_Answering_System, it was tested on a set of questions written in Arabic.
Our method is to improve questions in Arabic language, then apply the #Word2Vec algorithm with its models, on this questions after doing a preprocessing to extract similar words in context or meaning, then try to find the answers to the reformulated questions via Google API. Only the relevant answers are kept for each question.
Other times, we take the initial questions without preprocessing (which means in the original state) and try to find the answers also via Google API, and we keep just the relevant
answers.
From the results we get after doing the two parts, we compare between the two results (initial and reformulated), to see if there is an improvement or not according to the percentage.
