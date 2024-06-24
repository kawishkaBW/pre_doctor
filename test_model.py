import pickle

def test_model():
    filename = 'model/diabetes_predict_model.pickle'
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Loaded model type: {type(model)}")
        if hasattr(model, 'predict'):
            print("Model has a predict method.")
        else:
            print("Model does NOT have a predict method.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    test_model()
