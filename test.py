import pickle

with open('data/halfcheetah/expert_data_HalfCheetah-v4.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)