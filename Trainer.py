from ModelData import *
from Model import *

data = ModelData.ModelData(path=r'your/path/here/', file_format='csv', on_error='raise')
print(data().head(10))

data.preprocess(train_test_split=0.8, label_col='label_col', word_col='words_col')
print(data.X_train)
print(data.X_test)
print(data.y_train)
print(data.y_test)

model = Model(data=data, ensemble=False)
model.instantiate(model_type='BiLSTM',
                  hidden_layers=2,
                  neurons_per_layer=256,
                  dropout_threshold=0.2,
                  validation_split=0.2,
                  epochs=100,
                  patience=10)
model.fit()
model.evaluate(batch_size=1)
model.predict('predictions')
