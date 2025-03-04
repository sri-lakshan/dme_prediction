from models.opthamologist import train_model

if __name__ == '__main__':
    train_model("data/train_data.csv", "data/images", num_epochs=10, batch_size=16, learning_rate=1e-4, save_path="../model_weights.pth")