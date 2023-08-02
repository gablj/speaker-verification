The .json files in this directory are from the data retreived during the training of the model, during the trainign,
every $10$ epochs the training and testing loss, accuracy and eer were averaged and stored on the corresponding .json file, 
with a "key:dictionary" format.

The format of the content of these files is as follows:

For the "training.json" file:
{"epoch_number": {"loss": x.xxxxxx, "eer": x.xxxxxx, "accuracy": x.xxxxxx}, ..., "epoch_number": {"loss": x.xxxxxx, "eer": x.xxxxxx, "accuracy": x.xxxxxx}}
For the "testing.json" file:
{"epoch_number": {"loss_test": x.xxxxxx, "eer_test": x.xxxxxx, "test_accuracy": x.xxxxxx}, ..., "epoch_number": {"loss_test": x.xxxxxx, "eer_test": x.xxxxxx, "test_accuracy": x.xxxxxx}}.
