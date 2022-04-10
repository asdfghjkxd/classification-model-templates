from data import *
from model import *

if __name__ == '__main__':
    model_data = ModelData(path=r'your/path/here/', file_format='csv', on_error='raise')
    print(model_data().head(10))

    model_data.preprocess(train_test_split=0.8, label_col='label_col', word_col='words_col')
    print(model_data.X_train)
    print(model_data.X_test)
    print(model_data.y_train)
    print(model_data.y_test)

    tf_model = Model(m_data=model_data, ensemble=False)
    tf_model.instantiate(model_type='BiLSTM',
                         hidden_layers=2,
                         neurons_per_layer=256,
                         dropout_threshold=0.2,
                         validation_split=0.2,
                         epochs=100,
                         patience=10)
    tf_model.fit()
    tf_model.evaluate(batch_size=1)
    tf_model.predict('predictions')
