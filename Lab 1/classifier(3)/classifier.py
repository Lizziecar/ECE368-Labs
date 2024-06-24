import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
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

    # Steps
    # Open file lists
    # Count number of words in file lists
    # Sums total number of words
    # Calculate estimated probabilities

    ### Process data:
    # Spam:
    spam_word_freq = util.get_word_freq(file_lists_by_category[0])
    # Get totals:
    spam_freq_total = 0
    word_total = 0
    for word in spam_word_freq:
        spam_freq_total += spam_word_freq[word]
        word_total += 1

    # Ham:
    ham_word_freq = util.get_word_freq(file_lists_by_category[1])
    # Get totals:
    ham_freq_total = 0
    for word in ham_word_freq:
        ham_freq_total += ham_word_freq[word]
        if word not in spam_word_freq:
            word_total += 1

    ### Calcualte probability:
    p_d = util.Counter()
    q_d = util.Counter()

    for word in spam_word_freq: # Going through each word
        probability = (spam_word_freq[word] + 1) / (spam_freq_total + word_total)
        p_d[word] = probability

        # Fill in probability if word is not present in the other list
        q_d[word] = 1/(ham_freq_total + word_total)

    for word in ham_word_freq: # Going through each word
        probability = (ham_word_freq[word] + 1) / (ham_freq_total + word_total)
        q_d[word] = probability

        if p_d[word] == 0:
            p_d[word] = 1/(spam_freq_total + word_total)
    
    # Return result
    probabilities_by_category = (p_d, q_d)
    
    #print(p_d)
    #print(q_d)

    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category, decision):
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

    ### calcualte values for decision rule
    words = util.get_words_in_file(filename)

    x_nds = util.Counter()
    for word in words:
        x_nds[word] += 1

    p_d, q_d = probabilities_by_category
    
    # Spam:
    p_d_sum = np.log(prior_by_category[0])
    for word in q_d:
        p_d_sum += x_nds[word] * np.log(p_d[word])

    # Ham:
    q_d_sum = np.log(prior_by_category[1])
    for word in q_d:
        q_d_sum += x_nds[word] * np.log(q_d[word])

    ### check decision rule to decide spam or ham
    #print(p_d_sum)
    #print(q_d_sum)
    if p_d_sum - q_d_sum > decision:
        result = 'spam'
    else:
        result = 'ham'
    #print(result)
    
    ### calculate p(x|y=1) and p(x|y=0)
    # Ignoring the coeff because it is the same across all distributions
    p_y_1 = p_d_sum
    p_y_0 = q_d_sum
    
    probabilties = [p_y_1, p_y_0]

    classify_result = result, probabilties
    return classify_result

def select_files(directory, fraction=0.75):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    random.shuffle(all_files)
    num_files = int(len(all_files) * fraction)
    return all_files[:num_files]
if __name__ == '__main__':

    ############################CHANGE YOUR STUDENT ID###############################
    student_number = 1007930820  # Replace with the actual student number
    random.seed(student_number)
    # folder for training and testing
    spam_folder = "classifier(3)/data/spam"
    ham_folder = "classifier(3)/data/ham"
    if student_number % 2 == 0:
        test_folder = "classifier(3)/data/testing2"
    else:
        test_folder = "classifier(3)/data/testing1"

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
    
    # Arrange trade_off values
    trade_offs = np.arange(-10, 50.1, 3)

    # Initialize Error
    type1_error = np.zeros(20)
    type2_error = np.zeros(20)

    # loop through
    for i in range(20):

        total_type_errors = [0, 0]
        performance_measures = np.zeros([2,2])

        for filename in (util.get_files_in_folder(test_folder)):
        # Classify
            label,log_posterior = classify_new_email(filename,
                                                    probabilities_by_category,
                                                    priors_by_category, trade_offs[i])
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1
        
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    
    type1_error[i] = total_type_errors[0] - correct[0]
    type2_error[i] = total_type_errors[1] - correct[1]

# Plot data
plt.plot(type1_error, type2_error)
plt.title("Type 1 vs Type 2 error")
plt.xlabel("Type 1 Error")
plt.ylabel("Type 2 Error")
plt.show()



 