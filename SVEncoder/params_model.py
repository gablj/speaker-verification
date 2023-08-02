'''
"GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION" - https://arxiv.org/pdf/1710.10467.pdf [1]
'''
##Model parameters
model_hidden_size = 256 #The paper [1] used 768, however, this implementation is on a smaller dataset  
model_embedding_size = 64 
model_num_layers = 3

##Training parameters
learning_rate_init = 1e-4 
speakers_per_batch = 64 
utterances_per_speaker = 10 

##Testing parameters 
speakers_per_test_batch = 64 
utterances_per_test_speaker = 10