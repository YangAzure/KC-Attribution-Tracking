# Model without synthetic data
# Specify arguments
KC_index = 1

# # KC 0
# LC_A = 0.3324
# LC_B = 0.2050

# KC 1
LC_A = 0.5887
LC_B = 0.2574

# # KC 2
# LC_A = 0.4527
# LC_B = 0.1643

max_epochs = 60 
resample_times = 10
lr = 0.0002
batch_size = 64 

LC_WEIGHT = 0.0001

import pandas as pd
kc_df = pd.read_csv("../data/CWOSyntheticNewProbSolProblem.csv")

tracking_df = kc_df
tracking_df = tracking_df.fillna(0).set_index('ProblemID')
tracking_df = tracking_df.drop(['ProblemDec', 'Hussle in ChatGPT'], axis=1)
dropping = []
for c in tracking_df.columns:
    column = tracking_df[c]
    if sum(column)/len(column)<0.25 or sum(column)/len(column)>0.75:
        dropping.append(c)
        
        
tracking_df = tracking_df.drop(dropping, axis=1)

rand_seed = 0
code_df = pd.read_csv("../data/paths_with_functions_synthetic.tsv", sep='\t')

# code_df = code_df[code_df['AssignmentID'] == 439] 
# Note: Already filtered the assignments so no need additional filtering

correct_df = code_df[code_df['Score'] == 1]

problems = pd.unique(code_df['ProblemID'])
synthetic_problems = [p for p in problems if p > 1000]
original_problems = [p for p in problems if p < 1000]

from sklearn.model_selection import train_test_split, KFold


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import sem
from matplotlib import pyplot as plt

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
setup_seed(rand_seed)

def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary

    """
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword

def convert_to_idx(sample, node_word_index, path_word_index):
    """
    Converting to the index 
    Input:
    sample: list. One single training sample, which is a code, represented as a list of neighborhoods.
    node_word_index: dict. The node to word index dictionary.
    path_word_index: dict. The path to word index dictionary.

    """
    sample_index = []
    for line in sample:
        components = line.split(",")
        if components[0] in node_word_index:
            starting_node = node_word_index[components[0]]
        else:
            starting_node = node_word_index['UNK']
        if components[1] in path_word_index:
            path = path_word_index[components[1]]
        else:
            path = path_word_index['UNK']
        if components[2] in node_word_index:
            ending_node = node_word_index[components[2]]
        else:
            ending_node = node_word_index['UNK']
        
        sample_index.append([starting_node,path,ending_node])
    return sample_index


# Pre-processing of the sequential data
# Cheat removal
all_students = pd.unique(code_df['SubjectID'])
code_df['ServerTimestamp'] = pd.to_datetime(code_df['ServerTimestamp'], format="%Y-%m-%dT%H:%M:%S")
def sec_diff(late, early):
    return (late - early)/ pd.Timedelta(seconds=1)


processed_df_list = []
for s in all_students:
    if pd.isna(s):
        continue
    student_df = code_df[code_df['SubjectID'] == s].reset_index()
    past_row = student_df.iloc[0]
    start_row = student_df.iloc[0]
    dropping_index = []
    cheat_flag = 0
    problem_count = 1
    current_problem_attempts = 1
    attempt_average = 0
    for i, row in student_df.iterrows():
        # Init removal time
        remove_flag = 0
        
        # Calculate different in submission time
        second_diff = sec_diff(row['ServerTimestamp'], past_row['ServerTimestamp'])
        
        # Cheat handling
        # New problem
        if row['ProblemID'] != past_row['ProblemID'] and i > 0:
            # Cheat checker (simple version)
            
            if (current_problem_attempts < 3 and # Not struggled
                attempt_average - current_problem_attempts > 8 and # Historical struggle
                problem_count > 3):
                
                time_used = sec_diff(past_row['ServerTimestamp'], start_row['ServerTimestamp'])
                if time_used < 90: # Short and fast submission
                    cheat_flag = 1
                    break
                    
            # Update to new problem
            attempt_average = (attempt_average*problem_count + current_problem_attempts)/(problem_count + 1)
            problem_count += 1
            current_problem_attempts = 1
            start_row = row
        # Old problem
        else:
            current_problem_attempts += 1
            

#         # Confusion handling
#         if i > 0:
#             if row['Code'] == past_row['Code']:
#                 remove_flag = 1
#             if second_diff < 15 and row['Score'] != 1:
#                 remove_flag = 1

        # Partial handling (really simple)
        if len(row['Code'].split("\n")) <= 5:
            remove_flag = 1
        
        # Update row
        past_row = row

        if remove_flag == 1:
            dropping_index.append(i)
    student_df = student_df.drop(dropping_index)
    if not cheat_flag:
        processed_df_list.append(student_df)
processed_df = pd.concat(processed_df_list)

import javalang
def program_parser(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree

# Keep first
first_df_list = []
students = pd.unique(processed_df["SubjectID"])
for s in students:
    student_df = processed_df[processed_df['SubjectID'] == s].reset_index()
    past_ps = []
    dropping_index = []
    for i, row in student_df.iterrows():
        remove_flag = 0
        
        # ONLY KEEP FIRST ATTEMPT!
        if row['ProblemID'] not in past_ps:
            try:
                parsed = program_parser(row['Code'])
                past_ps.append(row['ProblemID'])
            except:
                parsed = "Uncompilable"
                remove_flag = 1
        else:
            remove_flag = 1
            
        
        # Update row
        past_row = row

        if remove_flag == 1:
            dropping_index.append(i)
    student_df = student_df.drop(dropping_index)
    first_df_list.append(student_df)
first_df = pd.concat(first_df_list)

processed_df['First'] = processed_df['CodeStateID'].isin(first_df['CodeStateID'])

# processed_df = first_df

def read_seq_df(df, subjects, KC_index):
    """
    Reading extracted path data frame and return the paths and the label.

    """
    separated_paths_label = []
    separated_paths_data = []
    separated_paths_problem = []
    separated_paths_score = []
    separated_paths_first = []
    separated_paths_code = []
    for s in subjects:
        student_df = df[df['SubjectID'] == s]
        s_separated_paths_label = []
        s_separated_paths_data = []
        s_separated_paths_problem = []
        s_separated_paths_score = []
        s_separated_paths_first = []
        s_separated_paths_code = []
        for index, row in student_df.iterrows():
            if type(row['RawASTPath']) == str:
                s_separated_paths_label.append(row[tracking_df.columns[KC_index]])
                s_separated_paths_data.append(row['RawASTPath'].split("\U0001f972"))
                s_separated_paths_problem.append(row['ProblemID'])
                s_separated_paths_score.append(row['Score'])
                s_separated_paths_first.append(row['First'])
                s_separated_paths_code.append(row['Code'])
        separated_paths_label.append(s_separated_paths_label)
        separated_paths_data.append(s_separated_paths_data)
        separated_paths_problem.append(s_separated_paths_problem)
        separated_paths_score.append(s_separated_paths_score)
        separated_paths_first.append(s_separated_paths_first)
        separated_paths_code.append(s_separated_paths_code)


    return separated_paths_label, separated_paths_data, separated_paths_problem, separated_paths_score, separated_paths_first, separated_paths_code


class KCAttributorLC(nn.Module):
    """
    Defining the network. 

    """
    def __init__(self, node_count, path_count, P_number):
        super(KCAttributorLC, self).__init__()
        KC_number = len(tracking_df.columns)

        self.embed_nodes = nn.Embedding(node_count+2, 100) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, 100) # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(300,100)
        self.attention_layer = nn.Linear(100,1)

        self.prediction_layer = nn.Linear(300,1)
        self.attention_softmax = nn.Softmax(dim=2)

        self.fc_skill = nn.Linear(300, 1)

        self.leakyReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.device = device
        
        self.classification_layer = nn.Sigmoid()
    def forward(self, starting_node_index, ending_node_index, path_index, evaluating=False):
        '''
        Parameters:
        batch: b
        codes: c (non-unified)
        node_embeddings: ne
        path_embeddings: pe
        
        '''

        starting_node_embed = self.embed_nodes(starting_node_index) # (b,l,c,1) -> (b,l,c,ne)
        ending_node_embed = self.embed_nodes(ending_node_index) # (b,l,c,1) -> (b,l,c,ne)
        path_embed = self.embed_paths(path_index) # (b,l,c,1) -> (b,l,c,pe)
        
        full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed), dim=3) # (b,l,c,2ne+pe+q)
        if not evaluating:
            full_embed = self.embed_dropout(full_embed) # (b,l,c,2ne+pe+2q)
        
        full_embed_transformed = torch.tanh(self.path_transformation_layer(full_embed)) # (b,l,c,2ne+pe+2q)
        context_weights = self.attention_layer(full_embed_transformed) # (b,l,c,1)
        attention_weights = self.attention_softmax(context_weights) # (b,l,c,1)
        code_vectors = torch.sum(torch.mul(full_embed,attention_weights),dim=2) # (b,l,2ne+pe+2q)
        out = self.sig(self.fc_skill(code_vectors))
        
        return out, code_vectors
        
    def get_minibatches_idx(self, n, minibatch_size, shuffle=False, droplast=False):
        """
        Getting the minibatches given the training set

        """
        idx_list = np.arange(len(n))

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        if droplast:
            for i in range(math.floor(len(n) // minibatch_size)):
                minibatches.append(idx_list[minibatch_start : min(minibatch_start + minibatch_size,len(n))])
                minibatch_start += minibatch_size
        else:
            for i in range(math.ceil(len(n) // minibatch_size)):
                minibatches.append(idx_list[minibatch_start : min(minibatch_start + minibatch_size,len(n))])
                minibatch_start += minibatch_size
        return zip(range(len(minibatches)), minibatches)
        
    def padding(self, index_list, max_length):
        """
        Padding the paths and the code.

        """
        padder_index = 0

        padded_list = []
        for sample in index_list:

            if max_length > len(sample):
                # padding the paths, or just get the first "max_gram_length" of paths
                padding_length = max_length - len(sample)
                padded_list.append(sample + [padder_index]*padding_length)
            else:
                padded_list.append(sample[:max_length])
            
        return padded_list
    
    def padding_sequence(self, student_list, num_questions):
        max_length = 100
        padder_index = [0]*max_length
        padded_list = []
        for index_list in student_list:

            if num_questions > len(index_list):
                # padding the paths, or just get the first "max_gram_length" of paths
                padding_length = num_questions - len(index_list)
                padded_list.append(self.padding(index_list, max_length) + [padder_index]*padding_length)
            else:
                padded_list.append(self.padding(index_list[:num_questions], max_length))
            
        return padded_list
    
    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)
        
    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))
        
def zeros2end(x):
    zeros_mask = x == 0
    out = torch.stack(
        [torch.cat([x[_, :][~zeros_mask[_, :]], x[_, :][zeros_mask[_, :]]], dim=0) for _ in range(x.size()[0])])
    return out
    
class lossFunc(nn.Module):
    def __init__(self, num_of_questions, bs, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCEWithLogitsLoss()
        self.MSE = nn.MSELoss()
        self.num_of_questions = num_of_questions
        self.batch_size = bs
        self.device = device
    
    
    
    def forward(self, pred, labels, scores, problems, practicing_problems, first, code, test=False):
        
        
        first_error_rate = LC_A
        learning_rate = LC_B
        LC_weight = LC_WEIGHT
        
        loss = 0
        labeled_loss = 0
        checked_code = 0
        binary_preds = torch.tensor([])
        ground_truths = torch.tensor([])
        code_list = []
        pred = pred.to('cpu')

        for student in range(pred.shape[0]):
            
            for prob_ind in range(len(scores[student])):
                if not test:
                    if scores[student][prob_ind] == 1: # The submission is correct, but not necessary practiced.

                        binary_preds = torch.cat([binary_preds,pred[student][prob_ind]]) 
                        ground_truths = torch.cat([ground_truths, torch.tensor([labels[student][prob_ind]])])
                        code_list.append(code[student][prob_ind])
                        checked_code += 1
                else:
                    if problems[student][prob_ind] in practicing_problems and scores[student][prob_ind] == 0: # We only evaluate on relavant problems
                        binary_preds = torch.cat([binary_preds,pred[student][prob_ind]]) 
                        ground_truths = torch.cat([ground_truths, torch.tensor([labels[student][prob_ind]])])
                        code_list.append(code[student][prob_ind])
                        checked_code += 1
                    

        labeled_loss = self.crossEntropy(binary_preds, ground_truths)
        labeled_loss = labeled_loss/checked_code
        
        lc_mask = sum(problems == i for i in practicing_problems)
        prediction = torch.squeeze(pred) * lc_mask * first
        opportunities = prediction.shape[1]
        prediction_lc = zeros2end(prediction)
        fitting_lc = first_error_rate*np.arange(1,opportunities+1)**-learning_rate
        
        lc_loss = 0
        lc_preds = []
        lc_target = []
        
            
        for i in range(prediction.shape[1]):
            preds = torch.mean(prediction_lc[:,i][prediction_lc[:,i]!=0]).to(self.device)
            lcs = torch.tensor(fitting_lc[i], dtype=torch.float).to(self.device)
            if not preds.isnan():
#                 print(preds, lcs)
                digit_lc_loss = self.MSE((1-preds), lcs) # Reverse the predicts
                lc_loss += digit_lc_loss
                lc_preds.append((1-preds))
                lc_target.append(lcs)
                
        total_loss = (1-LC_weight)*labeled_loss + LC_weight*torch.sqrt(lc_loss)
        
                
        return total_loss, binary_preds, ground_truths, code_list
    
    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


label1_df = pd.read_csv("../data/439_yang_labeling_data_1.csv")
label2_df = pd.read_csv("../data/439_yang_labeling_data_2.csv")

label_df = pd.concat([label1_df, label2_df])

problems = pd.unique(processed_df['ProblemID'])
synthetic_problems = [p for p in problems if p > 1000]
original_problems = [p for p in problems if p < 1000]


test_f1 = []
test_auc = []
test_acc = []
total_preds = []

practiced_problems = list(tracking_df[tracking_df[tracking_df.columns[KC_index]] == 1].index)

test_subjects = pd.unique(label_df['SubjectID'])
train_subjects = pd.unique(processed_df[~processed_df['SubjectID'].isin(test_subjects)]['SubjectID'])


for i in range(resample_times):
    
    original_processed_df = processed_df[processed_df['ProblemID'].isin(original_problems)]
    original_processed_df = original_processed_df.sort_values('ServerTimestamp')
    
    
    train_df = original_processed_df[original_processed_df['SubjectID'].isin(train_subjects)]
    test_df = original_processed_df[original_processed_df['SubjectID'].isin(test_subjects)]
    test_df = test_df[test_df['CodeStateID'].isin(label_df['CodeStateID'])]
    
    for c in tracking_df.columns:
        train_df[c] = [tracking_df.loc[x[1]['ProblemID'], c] for x in train_df.iterrows()]
        test_df[c] = [1-label_df[label_df['CodeStateID'] == x[1]['CodeStateID']][c].iloc[0] for x in test_df.iterrows()] # Readin labels

    train_run_label, train_run_data, train_run_problem, train_run_score, train_run_first, train_run_code = read_seq_df(train_df, train_subjects, KC_index)
    test_label, test_data, test_problem, test_score, test_first, test_code = read_seq_df(test_df, test_subjects, KC_index)


    node_hist = {}
    path_hist = {}
    for student_paths in train_run_data:
        for paths in student_paths:
            starting_nodes = [p.split(",")[0] for p in paths]
            path = [p.split(",")[1] for p in paths]
            ending_nodes = [p.split(",")[2] for p in paths]
            nodes = starting_nodes + ending_nodes
            for n in nodes:
                if not n in node_hist:
                    node_hist[n] = 1
                else:
                    node_hist[n] += 1
            for p in path:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1

    node_count = len(node_hist)
    path_count = len(path_hist)

    # small frequency then abandon, for node and path
    valid_node = [node for node, count in node_hist.items()]
    valid_path = [path for path, count in path_hist.items()]

    # create ixtoword and wordtoix lists
    node_word_index, node_index_word = create_word_index_table(valid_node)
    path_word_index, path_index_word = create_word_index_table(valid_path)
    
    train_converted_data = [[convert_to_idx(sample, node_word_index, path_word_index) for sample in student_sample] for student_sample in train_run_data]
    test_converted_data = [[convert_to_idx(sample, node_word_index, path_word_index) for sample in student_sample] for student_sample  in test_data]
    
#     val_converted_data += train_converted_data
    
    P_number = len(problems)
    
    model = KCAttributorLC(node_count, path_count, P_number).to(device)
    
    loss_func = lossFunc(10, batch_size, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(max_epochs):
        model.train()

        mini_batches = model.get_minibatches_idx(train_converted_data, batch_size, shuffle=True, droplast=True)
        epoch_loss = 0
        for _, batch_index in mini_batches:
            
            batch_samples = [train_converted_data[t] for t in batch_index]
            raw_starting_node_index = [[[p[0] for p in single_sample] for single_sample in student_samples] for student_samples in batch_samples]
            raw_ending_node_index = [[[p[2] for p in single_sample] for single_sample in student_samples] for student_samples in batch_samples]
            raw_path_index = [[[p[1] for p in single_sample] for single_sample in student_samples] for student_samples in batch_samples]
            batch_max_len = max(len(student_samples) for student_samples in batch_samples)
            # Padding to max paths in a file
            starting_node_index = torch.LongTensor(model.padding_sequence(raw_starting_node_index, num_questions=batch_max_len)).to(device)
            ending_node_index = torch.LongTensor(model.padding_sequence(raw_ending_node_index, num_questions=batch_max_len)).to(device)
            path_index = torch.LongTensor(model.padding_sequence(raw_path_index, num_questions=batch_max_len)).to(device)

            batch_labels = [train_run_label[t] for t in batch_index]
            
            batch_scores = [train_run_score[t] for t in batch_index]
            
            batch_code = [train_run_code[t] for t in batch_index]
            
            batch_problem = torch.nn.utils.rnn.pad_sequence([torch.tensor(train_run_problem[t]) for t in batch_index], batch_first=True, padding_value=0)
            batch_first_attempt = torch.nn.utils.rnn.pad_sequence([torch.tensor(train_run_first[t]) for t in batch_index], batch_first=True, padding_value=0)
            
            predictions, code_vectors = model(starting_node_index,ending_node_index,path_index)
            
#             print(predictions)

            loss, binary_preds, ground_truths, code_used = loss_func(predictions, batch_labels, batch_scores, batch_problem, practiced_problems, batch_first_attempt, batch_code)

            epoch_loss += loss.item()*len(batch_index)
    
            # Model optimization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(train_converted_data)
        
    #test
    model.eval()
    test_mini_batches = model.get_minibatches_idx(test_converted_data, min(batch_size,len(test_converted_data)), shuffle=False)
    test_epoch_loss = 0
    test_epoch_lc_loss = 0
    for _, test_batch_index in test_mini_batches:
        batch_samples = [test_converted_data[t] for t in test_batch_index]
        raw_starting_node_index = [[[p[0] for p in single_sample] for single_sample in student_samples] for student_samples in batch_samples]
        raw_ending_node_index = [[[p[2] for p in single_sample] for single_sample in student_samples] for student_samples in batch_samples]
        raw_path_index = [[[p[1] for p in single_sample] for single_sample in student_samples] for student_samples in batch_samples]
        batch_max_len = max(len(student_samples) for student_samples in batch_samples)
        
        # Padding to max paths in a file
        starting_node_index = torch.LongTensor(model.padding_sequence(raw_starting_node_index, num_questions=batch_max_len)).to(device)
        ending_node_index = torch.LongTensor(model.padding_sequence(raw_ending_node_index, num_questions=batch_max_len)).to(device)
        path_index = torch.LongTensor(model.padding_sequence(raw_path_index, num_questions=batch_max_len)).to(device)

        batch_labels = [test_label[t] for t in test_batch_index]

        batch_scores = [test_score[t] for t in test_batch_index]
        
        batch_code = [test_code[t] for t in test_batch_index]

        batch_problem = torch.nn.utils.rnn.pad_sequence([torch.tensor(test_problem[t]) for t in test_batch_index], batch_first=True, padding_value=0)
        batch_first_attempt = torch.nn.utils.rnn.pad_sequence([torch.tensor(test_first[t]) for t in test_batch_index], batch_first=True, padding_value=0)

        predictions, code_vectors = model(starting_node_index,ending_node_index,path_index)

        loss, binary_preds, ground_truths, code_used = loss_func(predictions, batch_labels, batch_scores, batch_problem, practiced_problems, batch_first_attempt, batch_code, test=True)
        
        total_preds.append(list(binary_preds.detach().numpy()))

        test_epoch_loss += loss.item()*len(test_batch_index)
        

    # Calculating the prediction scores and the classification results.
    test_loss = test_epoch_loss / len(test_converted_data)
#     print(nn.Sigmoid()(binary_preds))
#     print(ground_truths)
        
#     print(epoch, "val loss:", val_loss, "val lc loss:", val_lc_loss)

    test_auc.append(metrics.roc_auc_score(ground_truths.detach().numpy(), (binary_preds).detach().numpy()))
    test_acc.append(metrics.accuracy_score(ground_truths.detach().numpy(), (binary_preds).detach().numpy().round())) 
    test_f1.append(metrics.f1_score(ground_truths.detach().numpy(), (binary_preds).detach().numpy().round()))
    
    print("Round:", i)
    
    print("avg auc:", np.mean(test_auc), "std auc:", np.std(test_auc), "stderr auc:", sem(test_auc))
    print("avg acc:", np.mean(test_acc), "std acc:", np.std(test_acc), "stderr acc:", sem(test_acc))
    print("avg f1:", np.mean(test_f1), "std f1:", np.std(test_f1), "stderr f1:", sem(test_f1))

pred_avg = np.mean(total_preds, axis=0)
pred_std = np.std(total_preds, axis=0)
pred_err = sem(total_preds, axis=0)
pd.DataFrame(data={"code":code_used,
                   "pred_avg":pred_avg,
                   "pred_std":pred_std,
                   "pred_err":pred_err,
                   "GT":ground_truths.detach().numpy()}).to_csv("lc_case_" +str(KC_index)+ ".csv")
# np.save('results/'+str(KC_index)+'LC.npy', [test_auc,test_acc,test_f1])