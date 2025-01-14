import torch
import typer
from elines_pakke.data import corrupt_mnist
from elines_pakke.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#app = typer.Typer()

#@app.command()
def evaluate(model_checkpoint: str) -> None:
    
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = MyAwesomeModel().to(DEVICE)
    path = f"models/{model_checkpoint}"
    model.load_state_dict(torch.load(path))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    #app()
    typer.run(evaluate)


# import argparse
# import torch
# from elines_pakke.data import corrupt_mnist
# from elines_pakke.model import MyAwesomeModel

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# def evaluate(model_checkpoint: str) -> None:
#     """Evaluate a trained model."""
#     print("Evaluating like my life depended on it")
#     print(model_checkpoint)

#     model = MyAwesomeModel().to(DEVICE)
#     path = f"models/{model_checkpoint}"
#     model.load_state_dict(torch.load(path))

#     _, test_set = corrupt_mnist()
#     test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

#     model.eval()
#     correct, total = 0, 0
#     for img, target in test_dataloader:
#         img, target = img.to(DEVICE), target.to(DEVICE)
#         y_pred = model(img)
#         correct += (y_pred.argmax(dim=1) == target).float().sum().item()
#         total += target.size(0)
#     print(f"Test accuracy: {correct / total}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate a trained model.")
#     parser.add_argument("model_checkpoint", type=str, help="Path to the model checkpoint")
#     args = parser.parse_args()
#     evaluate(args.model_checkpoint)