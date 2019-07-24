from numpy import newaxis

def predict_multiple(model, data, window_size, prediction_size):
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
    
def plot_multiple(predicted_data, true_data, prediction_size):
    plt.figure(figsize=(25, 20))
    plt.plot(true_data, label='True Data')
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
    plt.show()
    
def create_model_data(data):
    data = data[['Date']+[metric for metric in ['Close','Volume']]]
    data = data.sort_values(by='Date')
    return data

def create_inputs(data, window_len=window_len):
    norm_cols = [metric for metric in ['Close', 'Volume']]
    inputs = []
    for i in range(len(data) - window_len):
        temp_set = data[i:(i + window_len)].copy()
        inputs.append(temp_set)
        for col in norm_cols:
            inputs[i].loc[:, col] = inputs[i].loc[:, col] / inputs[i].loc[:, col].iloc[0] - 1  
    return inputs

def create_outputs(data, window_len=window_len):
    return (data['Close'][window_len:].values / data['Close'][:-window_len].values) - 1

def to_array(data):
    x = [np.array(data[i]) for i in range (len(data))]
    return np.array(x)

def build_model(inputs, output_size, 
                neurons, 
                activ_func=activation_function, 
                dropout=dropout, 
                loss=loss, 
                optimizer=optimizer):
    
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]), activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=True, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    model.summary()
    return model
    
def plot_results(history, model, Y_target, coin):
    plt.figure(figsize=(25, 20))
    plt.subplot(311)
    plt.plot(history.epoch, history.history['loss'], )
    plt.plot(history.epoch, history.history['val_loss'])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title(coin + ' Model Loss')
    plt.legend(['Training', 'Test'])

    plt.subplot(312)
    plt.plot(Y_target)
    plt.plot(model.predict(X_train))
    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title(coin + ' Single Point Price Prediction on Training Set')
    plt.legend(['Actual','Predicted'])

    ax1 = plt.subplot(313)
    plt.plot(test_set['Close'][window_len:].values.tolist())
    plt.plot(((np.transpose(model.predict(X_test)) + 1) * test_set['Close'].values[:-window_len])[0])
    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title(coin + ' Single Point Price Prediction on Test Set')
    plt.legend(['Actual','Predicted'])