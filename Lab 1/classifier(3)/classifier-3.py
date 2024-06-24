import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
from util import * #fixes the imports, just importing util did not work
import random

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here
    #note q_d is for HAM, p_d is for spam
    #initialize variables used to return
    #start by building dictionary of total word occurences by category
    #note, depending on how dataset works, may need to pad out dictionaries to have 0 entries for words with no frequency
    spam_freq = get_word_freq(file_lists_by_category[0])
    ham_freq = get_word_freq(file_lists_by_category[1])
    #count total number of words per category
    #also need to compute total vocab size D (dictSize) for purpose of laplace smoothing
    dictSize = 0
    spamTotal = 0
    for key in spam_freq:
        spamTotal += spam_freq[key]
        #increment dict size
        dictSize +=1
    #repeat for HAM
    hamTotal = 0
    for key in ham_freq:
        hamTotal += ham_freq[key]
        #increment dict size if word not yet counted
        if key not in spam_freq:
            dictSize+=1
    #initialize dicts to return - use counter
    p_d = Counter()
    q_d = Counter()

    #the dictionaries need smoothed entries for entries with missing keys so the probablity can be computed in q2 without a math error
    #loop through dicts computing vals
    for key in spam_freq:
        #compute value then add to dict
        value = (spam_freq[key]+1)/(spamTotal+dictSize)
        p_d[key] = value
        #laplace smooth q_d
        q_d[key] = 1/(hamTotal+dictSize)
    #repeat for HAM
    for key in ham_freq:
        #compute value then add to dict
        value = (ham_freq[key]+1)/(hamTotal+dictSize)
        q_d[key] = value
        #laplace smooth p_d
        if p_d[key] == 0:
            p_d[key] = 1/(spamTotal+dictSize)
    #return tuple
    probabilities_by_category = (p_d, q_d)

    print(p_d)
    print(q_d)
    
    
    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category, decision_boundary):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    #using formula from lecture, compute Ai (as a list) and Bi (also as a list)
    #then use MAP formula to compute to the 2 values to compare, find prediction based on them (I think the 2 values computed are the log posterior probs to return)
    #get sentence length
    sentence = get_words_in_file(filename)
    #get dict of word freq
    word_freq = Counter()
    for word in sentence:
        word_freq[word] += 1
    #intialize A and B
    #intialize chance of both cases variable
    #p(y|x) = p(x|y)p(y)p(x). goal is argmax(y), so the p(x) term can be ignored
    #load p(y) values 
    pSpam = prior_by_category[0]
    pHam = prior_by_category[1]
    #initialize variables used to compare probabilities - initialized to log of prior - see written answer for why
    pSpamGivenX = np.log(pSpam)
    pHamGivenX = np.log(pHam)
    #perform the summation of log probability
    for key in probabilities_by_category[0]:
        pSpamGivenX += word_freq[key] * np.log(probabilities_by_category[0][key])
        pHamGivenX += word_freq[key] * np.log(probabilities_by_category[1][key])
    #make classification decision - new decision rule implemented to allow trade-off between error types
    #print(pSpamGivenX)
    #print(pHamGivenX)
    if pSpamGivenX - pHamGivenX > decision_boundary:
        classification = "spam"
    else:
        classification = "ham"
    #print(classification)
    #format output
    listProb = [pSpamGivenX, pHamGivenX]
    classify_result = (classification, listProb)
    
    return classify_result

def select_files(directory, fraction=0.75):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    random.shuffle(all_files)
    num_files = int(len(all_files) * fraction)
    return all_files[:num_files]
if __name__ == '__main__':

    ############################CHANGE YOUR STUDENT ID###############################
    student_number = 1008333028  # Replace with the actual student number
    random.seed(student_number)
    # folder for training and testing
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    if student_number % 2 == 0:
        test_folder = "data/testing2"
    else:
        test_folder = "data/testing1"

    # generate the file lists for training
    file_lists = []
    file_lists = [select_files(folder) for folder in (spam_folder, ham_folder)]

        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category, 0)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1


    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    #use np.arange to loop through decision boundary values
    domain = np.arange(-10, 50.1, 5)
    #make lists for error tracking and plotting
    type1 = np.zeros(13)
    type2 = np.zeros(13)
    #loop through domain of decision boundaries
    for i in range(13):
        #reset totals
        totals[0] = 0
        totals[1] = 0
        performance_measures[0, 0] = 0
        performance_measures[0, 1] = 0
        performance_measures[1, 0] = 0
        performance_measures[1, 1] = 0
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                    probabilities_by_category,
                                                    priors_by_category, domain[i], 0)
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1


        #template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        #print("Boundary val: ", i)
        #print(template % (correct[0],totals[0],correct[1],totals[1]))
        #store the type 1 errors and type 2 errors
        type1[i] = totals[0] - correct[0]
        type2[i] = totals[1] - correct[1]
    #plot the data
    plt.plot(type1, type2)
    #label axes and title
    plt.xlabel('Type 1 Errors')
    plt.ylabel('Type 2 Errors')
    plt.title('Type 2 vs Type 1 errors')
    plt.show()
   
