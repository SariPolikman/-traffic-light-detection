class FilterLight:
    def __init__(self, model):
        from tensorflow.keras.models import load_model
        self.loaded_model = load_model(model)

    def predict_model(self, images):
        predictions = self.loaded_model.predict(images)

        return predictions
